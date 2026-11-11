"""Frozen one-step counterfactual actor credit for shared-system HAPPO.

Enumerate each actor's complete command grid while retaining the other original
commands. Exact daily physics/reward and the old critic give Q = r + gamma V.
The actor's credit is Q(selected) minus its old-policy-weighted mean Q. This
opt-in model-assisted estimator replaces actor GAE, while critic GAE targets
and the shared-system reward remain unchanged.
"""
import numpy as np
import torch

from harl.envs.cascade_reservoir.batched_physics import batched_transition
from harl.envs.cascade_reservoir.reward import validate_discount_factor


ACTOR_CREDIT_MODES = ('team_gae', 'counterfactual')
COUNTERFACTUAL_NORMALIZATION = 'common_controllable_rms_without_centering'


def validate_credit_assignment(mode):
    if not isinstance(mode, str) or mode not in ACTOR_CREDIT_MODES:
        raise ValueError('happo_credit_assignment must be one of '+', '.join(ACTOR_CREDIT_MODES))
    return mode


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(name+' must be a positive integer')
    return int(value)


def _finite_array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(name+' must be a finite array with shape '+str(shape))
    return result


def system_rewards_from_dispatch(operation_reward, dispatch):
    """Batch equivalent of OperationReward's scalar system reward, not legacy."""
    power = np.asarray(dispatch['power'], dtype=np.float64)
    release = np.asarray(dispatch['release'], dtype=np.float64)
    if (power.ndim != 2 or power.shape[1] != 5 or release.shape != power.shape or
            not np.isfinite(power).all() or not np.isfinite(release).all() or
            (power < 0.).any() or (release < 0.).any()):
        raise ValueError('Dispatch power and release must be finite nonnegative five-column batches')
    capacity = _finite_array(operation_reward.power_cap, (5,), 'installed capacity')
    if (capacity <= 0.).any():
        raise ValueError('Installed capacities must be positive')
    local = power/capacity
    settings = operation_reward.settings
    if settings['enabled']:
        if getattr(operation_reward, 'requirements', None) is None:
            raise ValueError('Counterfactual service penalties require loaded requirements')
        for term, actual in (('guaranteed_output', power), ('ecological_flow', release)):
            threshold = _finite_array(operation_reward.thresholds[term], (5,), term+' threshold')
            if (threshold < 0.).any():
                raise ValueError('Service thresholds must be nonnegative')
            shortfall = np.maximum(threshold-actual, 0.)
            shortfall = np.where(shortfall > 1e-10*np.maximum(threshold, 1.), shortfall, 0.)
            relative = np.divide(shortfall, threshold, out=np.zeros_like(shortfall), where=threshold > 0.)
            local = local-settings[term+'_weight']*relative
            if term == 'guaranteed_output':
                local = local-settings.get('guaranteed_output_violation_weight', 0.)*(shortfall > 0.)
    result = .5*local.sum(axis=1)+.5*power.sum(axis=1)/capacity.sum()
    if not np.isfinite(result).all():
        raise FloatingPointError('Non-finite counterfactual shared-system rewards')
    return result


def normalize_counterfactual_advantages(raw_advantages, controls):
    """Apply one sign-preserving RMS scale over all controllable actor rows."""
    raw = np.asarray(raw_advantages, dtype=np.float64)
    active = np.asarray(controls)
    if raw.shape != active.shape or raw.ndim != 3 or raw.shape[-1] != 5:
        raise ValueError('Counterfactual advantages and controls must have shape [time, lanes, 5]')
    if not np.isfinite(raw).all() or not np.isfinite(active).all() or not np.isin(active, (0., 1.)).all():
        raise ValueError('Counterfactual advantages must be finite and controls must be binary')
    selected = raw[active > 0.]
    rms = float(np.sqrt(np.mean(np.square(selected)))) if len(selected) else 0.
    scale = rms+1e-5
    normalized = np.where(active > 0., raw/scale, 0.).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise FloatingPointError('Non-finite normalized counterfactual advantages')
    return normalized, scale


@torch.no_grad()
def compute_counterfactual_credit(physics, operation_reward, critic, value_normalizer,
        context, next_states, policy_probabilities, continuation_masks, gamma, batch_size=4096):
    """Return frozen Q grids, baselines, per-actor credit and one common scale.

    Context arrays storage_start/external/actions/storage_lower/storage_upper
    and optional controls are [T,N,5]; actions are original integer commands.
    Extra context provenance fields are ignored. Next states are [T,N,18/38],
    old probabilities [T,N,5,K], continuation masks [T,N,1]. Call before any
    actor, critic or ValueNorm update. Candidate batches never exceed batch_size.

    Candidate next-state templates retain the actual next-date forcing/calendar
    and bound rows; only storage and previous-release features are replaced.
    Terminal rows and gamma=0 never query the critic. No environment, optimizer,
    normalizer or random state is advanced. Approximate one-step values do not
    establish long-term optimality or guaranteed performance improvement.
    """
    gamma = validate_discount_factor(gamma)
    batch_size = _positive_integer(batch_size, 'counterfactual batch_size')
    if not isinstance(context, dict):
        raise ValueError('Counterfactual context must be a mapping')
    probabilities = np.asarray(policy_probabilities, dtype=np.float64)
    if probabilities.ndim != 4 or probabilities.shape[2] != 5 or probabilities.shape[3] < 2:
        raise ValueError('Old-policy probabilities must have shape [time, lanes, 5, action_bins>=2]')
    time, lanes, agents, bins = probabilities.shape
    if time < 1 or lanes < 1:
        raise ValueError('Counterfactual rollout must have positive time and lane dimensions')
    if (not np.isfinite(probabilities).all() or (probabilities < 0.).any() or
            (probabilities > 1.+1e-7).any() or
            not np.allclose(probabilities.sum(axis=-1), 1., rtol=0., atol=1e-5)):
        raise ValueError('Old-policy probabilities must be finite normalized categorical distributions')
    # Remove float32 summation drift on a private copy, without changing policy.
    probabilities = probabilities/probabilities.sum(axis=-1, keepdims=True)
    shape = (time, lanes, agents)
    values = {}
    for name in ('storage_start', 'external', 'actions', 'storage_lower', 'storage_upper'):
        if name not in context:
            raise ValueError('Missing counterfactual context: '+name)
        values[name] = _finite_array(context[name], shape, name).reshape(-1, agents)
    actions = values['actions']
    if ((actions != np.floor(actions)) | (actions < 0) | (actions >= bins)).any():
        raise ValueError('Counterfactual context actions must be original integer command indices')
    actions = actions.astype(np.int64)
    controls = _finite_array(context.get('controls', np.ones(shape)), shape, 'controls')
    if not np.isin(controls, (0., 1.)).all():
        raise ValueError('Counterfactual controls must be binary')
    states = np.asarray(next_states)
    if (states.ndim != 3 or states.shape[:2] != (time, lanes) or states.shape[-1] not in (18, 38)
            or not np.isfinite(states).all()):
        raise ValueError('Counterfactual next states must be finite [time, lanes, 18/38] arrays')
    states = states.astype(np.float32, copy=False).reshape(-1, states.shape[-1])
    masks = _finite_array(continuation_masks, (time, lanes, 1), 'continuation masks').reshape(-1)
    if not np.isin(masks, (0., 1.)).all():
        raise ValueError('Continuation masks must be binary')
    data = physics.data
    span = _finite_array(data.vmax-data.vmin, (5,), 'storage span')
    caps = _finite_array(data.release_cap, (5,), 'release capacity')
    if (span <= 0.).any() or (caps <= 0.).any():
        raise ValueError('Storage spans and release capacities must be positive')
    rows = time*lanes
    storage_constraints = context.get('storage_constraints')
    if storage_constraints is not None:
        storage_constraints = _finite_array(storage_constraints,
            (time, lanes, 6, 6), 'storage_constraints').reshape(-1, 6, 6)
    total = rows*agents*bins
    q_flat = np.empty(total, dtype=np.float64)
    predict = getattr(critic, 'get_values', critic)
    for start in range(0, total, batch_size):
        stop = min(start+batch_size, total)
        indices = np.arange(start, stop)
        row = indices//(agents*bins)
        actor = (indices//bins) % agents
        command = indices % bins
        normalized = actions[row].astype(np.float64)/(bins-1)
        normalized[np.arange(len(indices)), actor] = command/(bins-1)
        options = ({} if storage_constraints is None else
                   dict(storage_constraints=storage_constraints[row]))
        dispatch = batched_transition(physics, values['storage_start'][row], values['external'][row],
            normalized, values['storage_lower'][row], values['storage_upper'][row], **options)
        q = system_rewards_from_dispatch(operation_reward, dispatch)
        continuing = masks[row] > 0.
        if gamma > 0. and continuing.any():
            candidate = states[row[continuing]].copy()
            candidate[:, :5] = (dispatch['storage_end'][continuing]-data.vmin)/span
            candidate[:, 10:15] = dispatch['release'][continuing]/caps
            prediction = predict(candidate)
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.detach().cpu().numpy()
            prediction = np.asarray(prediction)
            if prediction.shape != (int(continuing.sum()), 1) or not np.isfinite(prediction).all():
                raise ValueError('Counterfactual credit requires one finite scalar shared-team value per state')
            if value_normalizer is not None:
                prediction = value_normalizer.denormalize(prediction)
            prediction = np.asarray(prediction, dtype=np.float64)
            if not np.isfinite(prediction).all():
                raise FloatingPointError('Non-finite denormalized counterfactual values')
            q[continuing] += gamma*prediction[:, 0]
        q_flat[start:stop] = q
    q_values = q_flat.reshape(time, lanes, agents, bins)
    baseline = np.sum(probabilities*q_values, axis=-1)
    q_actual = np.take_along_axis(q_values, actions.reshape(shape)[..., None], axis=-1)[..., 0]
    raw = np.where(controls > 0., q_actual-baseline, 0.)
    normalized, scale = normalize_counterfactual_advantages(raw, controls)
    best = np.argmax(q_values, axis=-1).astype(np.int64)
    selected = actions.reshape(shape)
    controllable = controls > 0.
    gaps = np.max(q_values, axis=-1)-q_actual
    active_count = int(controllable.sum())
    summary = dict(estimator='one_step_model_assisted_counterfactual',
        objective='shared_system_reward', baseline='old_policy_probability_weighted_one_actor_counterfactual_q',
        normalization=COUNTERFACTUAL_NORMALIZATION, normalization_scale=scale,
        rollout_rows=rows, actor_rows=int(raw.size), controllable_rows=active_count,
        counterfactual_candidates=total, critic_candidate_batch_limit=batch_size,
        positive_advantage_fraction=(float(np.mean(raw[controllable] > 0.)) if active_count else 0.),
        negative_advantage_fraction=(float(np.mean(raw[controllable] < 0.)) if active_count else 0.),
        selected_best_action_fraction=(float(np.mean(selected[controllable] == best[controllable])) if active_count else 0.),
        mean_best_action_q_gap=(float(gaps[controllable].mean()) if active_count else 0.),
        mean_selected_q=float(q_actual.mean()), mean_baseline_q=float(baseline.mean()),
        critic_bootstrap='frozen rollout critic and ValueNorm; zero at terminal and when gamma is zero',
        scope='same starting state and other original commands; all downstream physics and team rewards recomputed',
        limitation='approximate one-step action values; long-term performance improvement is not established')
    return dict(q_values=q_values, q_actual=q_actual, baseline=baseline,
        raw_advantages=raw, normalized_advantages=normalized,
        best_action_indices=best, advantage_scale=scale, summary=summary)

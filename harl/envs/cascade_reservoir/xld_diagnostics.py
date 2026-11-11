"""Read-only, same-day Xiluodu dispatch diagnostics.

Counterfactuals keep the actual starting state, current external inflows, daily
storage bounds, active terminal reachability constraints and the other four
actors' chosen actions fixed. They do not
assess future water value or declare that a same-day alternative is a better
multi-day policy. No counterfactual is executed in the real environment.
"""
import csv

import numpy as np

from harl.envs.cascade_reservoir.data_loader import IDS


_XLD = IDS.index('XLD')
CLASSIFICATION_DEFINITIONS = {
    'requirement_inactive': 'XLD guaranteed-output requirement is zero.',
    'no_violation': 'Executed XLD power meets the effective requirement within reward tolerance.',
    'same_day_policy_alternative': 'Another available discrete XLD action meets the requirement from the same state; future value is unassessed.',
    'discrete_grid_gap': 'A continuous XLD action meets the requirement but none on the configured discrete grid does.',
    'same_state_capacity_limit': 'Even global maximum modeled head cannot meet the requirement at turbine/installed capacity.',
    'same_state_water_limit': 'Water releasable above today\'s next-state storage floor is insufficient even at global maximum modeled head and without turbine limits.',
    'same_state_dispatch_limit': 'The current feasible release ceiling cannot meet the requirement even at global maximum modeled head.',
    'same_state_head_limit': 'Water and dispatch optimistic bounds permit the requirement, but actual head/storage coupling prevents every continuous same-day XLD action from meeting it.',
}

XLD_DIAGNOSTIC_FIELDS = (
    'phase', 'seed', 'evaluation_round', 'window_id', 'episode_step', 'date',
    'storage_start_m3', 'storage_end_m3', 'level_start_m', 'level_end_m',
    'storage_lower_m3', 'storage_upper_m3',
    'upstream_release_m3s', 'external_inflow_m3s', 'inflow_m3s',
    'release_m3s', 'turbine_m3s', 'spill_m3s',
    'net_head_m', 'reference_head_m', 'power_mw', 'guaranteed_output_mw',
    'shortfall_mw', 'required_turbine_flow_at_actual_head_m3s',
    'required_head_at_actual_turbine_m', 'power_at_reference_head_same_turbine_mw',
    'sampled_action_index', 'action_bins', 'forced_action',
    'feasible_release_low_m3s', 'feasible_release_high_m3s',
    'water_budget_upper_m3s', 'downstream_upper_m3s', 'outlet_capacity_m3s',
    'turbine_capacity_m3s', 'installed_capacity_mw',
    'violating', 'classification', 'head_contribution_at_executed_release',
    'water_limit_proven', 'dispatch_limit_proven',
    'best_grid_action_index', 'best_grid_power_mw', 'best_grid_release_m3s',
    'best_grid_head_m', 'meeting_grid_action_count',
    'minimum_meeting_action_index', 'minimum_meeting_release_m3s',
    'meeting_release_delta_m3s',
    'continuous_max_power_mw', 'continuous_best_release_m3s',
    'continuous_best_head_m',
    'best_grid_other_guaranteed_violations', 'actual_other_guaranteed_violations',
    'best_grid_other_ecological_violations', 'actual_other_ecological_violations',
)
_TEXT_FIELDS = {'phase', 'window_id', 'date', 'classification'}
_BOOLEAN_FIELDS = {'forced_action', 'violating', 'head_contribution_at_executed_release',
                   'water_limit_proven', 'dispatch_limit_proven'}
_INTEGER_FIELDS = {'seed', 'evaluation_round', 'episode_step', 'sampled_action_index',
                   'action_bins', 'best_grid_action_index', 'meeting_grid_action_count',
                   'minimum_meeting_action_index', 'best_grid_other_guaranteed_violations',
                   'actual_other_guaranteed_violations', 'best_grid_other_ecological_violations',
                   'actual_other_ecological_violations'}
_OPTIONAL_FIELDS = {
    'required_head_at_actual_turbine_m', 'sampled_action_index',
    'best_grid_action_index', 'best_grid_power_mw', 'best_grid_release_m3s',
    'best_grid_head_m', 'meeting_grid_action_count', 'minimum_meeting_action_index',
    'minimum_meeting_release_m3s', 'meeting_release_delta_m3s',
    'continuous_max_power_mw', 'continuous_best_release_m3s', 'continuous_best_head_m',
    'best_grid_other_guaranteed_violations', 'actual_other_guaranteed_violations',
    'best_grid_other_ecological_violations', 'actual_other_ecological_violations',
}


class XLDDispatchDiagnosticError(RuntimeError):
    """A diagnostic failed; this is not a failed actual physical trajectory."""


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(name+' must be an integer >= '+str(minimum))
    return int(value)


def _number(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name+' must be a finite number')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name+' must be a finite number') from exc
    if not np.isfinite(value):
        raise ValueError(name+' must be a finite number')
    return value


class XLDDispatchWriter:
    """One compact scalar CSV row per executed XLD evaluation day."""

    def __init__(self, stream):
        self.stream = stream
        self.writer = csv.writer(stream, lineterminator='\n')
        self.writer.writerow(XLD_DIAGNOSTIC_FIELDS)
        self.stream.flush()

    def write(self, record):
        if set(record) != set(XLD_DIAGNOSTIC_FIELDS):
            raise ValueError('XLD diagnostic row has missing or unknown fields')
        values = []
        for name in XLD_DIAGNOSTIC_FIELDS:
            value = record[name]
            if value is None:
                if name not in _OPTIONAL_FIELDS:
                    raise ValueError('Missing required XLD diagnostic value: '+name)
                value = ''
            elif name in _TEXT_FIELDS:
                if not isinstance(value, str) or not value:
                    raise ValueError(name+' must be nonempty text')
            elif name in _BOOLEAN_FIELDS:
                if not isinstance(value, (bool, np.bool_)):
                    raise ValueError(name+' must be boolean')
                value = int(value)
            elif name in _INTEGER_FIELDS:
                value = _integer(value, name)
            else:
                value = format(_number(value, name), '.10g')
            values.append(value)
        if record['classification'] not in CLASSIFICATION_DEFINITIONS:
            raise ValueError('Unknown XLD diagnostic classification')
        self.writer.writerow(values)
        self.stream.flush()


class XLDDispatchDiagnostics:
    """Observe executed transitions without changing environment or RNG state.

    A vectorized local calculation considers every configured action, while
    selected witnesses are independently replayed through ReservoirPhysics for
    all five reservoirs. Continuous maxima also consider stage/storage segment
    breakpoints and interior quadratic stationary points. Maximum release is
    therefore never assumed to maximize power.
    """

    def __init__(self, env, writer=None, *, seed, evaluation_round, window_id, phase):
        self.env, self.data, self.physics = env, env.data, env.physics
        self.writer = writer
        self.seed = _integer(seed, 'seed')
        self.evaluation_round = _integer(evaluation_round, 'evaluation_round')
        if not isinstance(window_id, str) or not window_id:
            raise ValueError('window_id must be nonempty text')
        if phase not in ('training_evaluation', 'evaluation'):
            raise ValueError('XLD diagnostics are restricted to evaluation episodes')
        self.window_id, self.phase = window_id, phase
        self.bins = _integer(env.action_bins, 'action_bins', 2)
        operation = getattr(env, 'operation_reward', None)
        requirements = getattr(operation, 'requirements', None)
        if requirements is None:
            requirements = getattr(self.data, 'service_requirements', None)
        if requirements is None:
            raise ValueError('XLD diagnostics require loaded service requirements')
        self.guaranteed = self._vector(requirements.guaranteed_output_mw, 'guaranteed_output_mw')
        self.ecological = self._vector(requirements.ecological_flow_m3s, 'ecological_flow_m3s')
        if (self.guaranteed < 0.).any() or (self.ecological < 0.).any():
            raise ValueError('Service requirements cannot be negative')
        self.threshold = float(self.guaranteed[_XLD])
        self.tolerance = 1e-10*max(self.threshold, 1.)
        self.steps = self.violations = self.active_days = 0
        self.counts = dict.fromkeys(CLASSIFICATION_DEFINITIONS, 0)
        self.head_contribution_days = 0
        self.max_shortfall = 0.
        self.shortfall_sum = 0.
        if self.physics.head_model == 'fixed':
            self.reference_head = self.maximum_head = float(self.data.head[_XLD])
        elif self.physics.head_model == 'storage_sensitivity':
            sensitivity = self.data.head_sensitivity
            self.reference_head = float(sensitivity['reference_net_head_m'][_XLD])
            lower, upper = self.data.levels(self.data.vmin), self.data.levels(self.data.vmax)
            heads = [self.reference_head+float(sensitivity['head_slope_m_per_m'][_XLD])*
                     (level[_XLD]-float(sensitivity['assumed_reference_forebay_level_m'][_XLD]))
                     for level in (lower, upper)]
            self.maximum_head = float(max(heads))
        else:
            raise ValueError('Unsupported XLD head model')
        if not np.isfinite([self.reference_head, self.maximum_head]).all() or min(self.reference_head, self.maximum_head) <= 0.:
            raise ValueError('XLD diagnostic head references must be positive')

    @staticmethod
    def _vector(value, name):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (len(IDS),) or not np.isfinite(value).all():
            raise ValueError(name+' must be a finite five-vector')
        return value.copy()

    def _local_generation(self, row, releases):
        """Exact XLD scalar/vector equivalent of the shared generation model."""
        releases = np.asarray(releases, dtype=np.float64)
        end_storage = float(row['storage_start'][_XLD])+self.physics.dt*(float(row['inflow'][_XLD])-releases)
        if self.physics.head_model == 'fixed':
            heads = np.full(releases.shape, self.reference_head)
        else:
            anchors = self.data.anchors['XLD']
            start_storage = np.clip(float(row['storage_start'][_XLD]), self.data.vmin[_XLD], self.data.vmax[_XLD])
            start_level = np.interp(start_storage, anchors[:, 1], anchors[:, 0])
            end_level = np.interp(np.clip(end_storage, self.data.vmin[_XLD], self.data.vmax[_XLD]),
                                  anchors[:, 1], anchors[:, 0])
            sensitivity = self.data.head_sensitivity
            heads = self.reference_head+float(sensitivity['head_slope_m_per_m'][_XLD])*(
                (start_level+end_level)/2.-float(sensitivity['assumed_reference_forebay_level_m'][_XLD]))
        if not np.isfinite(heads).all() or (heads <= 0.).any():
            raise XLDDispatchDiagnosticError('Candidate has an invalid effective head')
        coefficient = self.physics.rho_g_eta*heads/1e6
        caps = np.minimum(self.data.turbine_cap[_XLD], self.data.power_cap[_XLD]/coefficient)
        powers = np.minimum(releases, caps)*coefficient
        return powers, heads

    def _continuous_candidates(self, row, low, high):
        if high-low <= 1e-8:
            return np.array([float(row['release'][_XLD])])
        points = [low, high]
        turbine_cap = float(self.data.turbine_cap[_XLD])
        if low < turbine_cap < high:
            points.append(turbine_cap)
        if self.physics.head_model == 'storage_sensitivity':
            # End storage is affine in release. Include every interpolation
            # kink, including global clipping boundaries used by generation.
            anchors = self.data.anchors['XLD'][:, 1]
            storages = np.concatenate([anchors, [self.data.vmin[_XLD], self.data.vmax[_XLD]]])
            breaks = float(row['inflow'][_XLD])+(float(row['storage_start'][_XLD])-storages)/self.physics.dt
            points.extend(float(q) for q in breaks if low < q < high)
        points = np.unique(np.asarray(points, dtype=np.float64))
        stationary = []
        _, heads = self._local_generation(row, points)
        coefficients = self.physics.rho_g_eta*heads/1e6
        for left, right, first, last in zip(points[:-1], points[1:], coefficients[:-1], coefficients[1:]):
            if right <= left or left >= turbine_cap:
                continue
            slope = (last-first)/(right-left)
            intercept = first-slope*left
            # Before turbine saturation, P(q)=intercept*q+slope*q**2.
            if slope < 0.:
                vertex = -intercept/(2.*slope)
                if left < vertex < right:
                    stationary.append(vertex)
        return np.unique(np.concatenate([points, np.asarray(stationary)]))

    def _chosen_actions(self, row):
        sampled = row.get('sampled_actions')
        if sampled is not None:
            sampled = self._vector(sampled, 'sampled_actions')
            if ((sampled != np.floor(sampled)) | (sampled < 0.) | (sampled >= self.bins)).any():
                raise ValueError('Sampled counterfactual actions must lie on the configured grid')
            return sampled/(self.bins-1)
        # Once a forced downstream action becomes controllable after changing
        # XLD release, its executed .5 value does not identify the actor's
        # original choice. Such rows must retain their sampled indices.
        if not np.all(self._vector(row['controls'], 'controls') == 1.):
            raise ValueError('Forced-action counterfactuals require original sampled_actions')
        normalized = self._vector(row['normalized_actions'], 'normalized_actions')
        if ((normalized < 0.) | (normalized > 1.)).any():
            raise ValueError('Normalized counterfactual actions must lie in [0, 1]')
        return normalized

    def _replay(self, row, normalized_xld, expected_release, expected_power):
        actions = self._chosen_actions(row)
        actions[_XLD] = normalized_xld
        constraints = row.get('storage_constraints')
        options = {} if constraints is None else dict(storage_constraints=constraints)
        replay = self.physics.transition(
            self._vector(row['storage_start'], 'storage_start'),
            self._vector(row['external'], 'external'), actions,
            self._vector(row['storage_lower_m3'], 'storage_lower_m3'),
            self._vector(row['storage_upper_m3'], 'storage_upper_m3'), **options)
        if (not np.isclose(replay['release'][_XLD], expected_release, rtol=1e-10, atol=1e-7) or
                not np.isclose(replay['power'][_XLD], expected_power, rtol=1e-10, atol=1e-7)):
            raise XLDDispatchDiagnosticError('Local candidate disagrees with the complete cascade replay')
        return replay

    def _other_violations(self, row, term):
        requirement = self.guaranteed if term == 'guaranteed' else self.ecological
        actual = np.asarray(row['power'] if term == 'guaranteed' else row['release'])
        mask = requirement > 0.
        mask[_XLD] = False
        return int(((requirement-actual > 1e-10*np.maximum(requirement, 1.)) & mask).sum())

    def update(self, transition, episode_step):
        """Return/write one observation; diagnostic errors cannot become physical failures."""
        step = _integer(episode_step, 'episode_step', 1)
        if step != self.steps+1:
            raise ValueError('XLD diagnostic episode steps must be consecutive from one')
        try:
            record = self._record(transition, step)
            if self.writer is not None:
                self.writer.write(record)
        except Exception as exc:
            raise XLDDispatchDiagnosticError(
                'XLD diagnostic failed; seed={} window={} date={} step={}; cause={}'.format(
                    self.seed, self.window_id, transition.get('date', 'unknown'), step, exc)) from exc
        self.steps += 1
        self.active_days += int(self.threshold > 0.)
        self.violations += int(record['violating'])
        self.counts[record['classification']] += 1
        self.head_contribution_days += int(record['head_contribution_at_executed_release'])
        self.shortfall_sum += record['shortfall_mw']
        self.max_shortfall = max(self.max_shortfall, record['shortfall_mw'])
        return record

    def _record(self, row, step):
        for name in ('storage_start', 'storage_end', 'storage_lower_m3', 'storage_upper_m3',
                     'external', 'inflow', 'release', 'turbine', 'spill', 'power', 'net_head_m',
                     'low', 'high', 'controls'):
            self._vector(row[name], name)
        power, head, turbine = (float(row[key][_XLD]) for key in ('power', 'net_head_m', 'turbine'))
        if min(power, turbine) < 0. or head <= 0.:
            raise ValueError('Actual XLD power/flow must be nonnegative and head positive')
        low, high = float(row['low'][_XLD]), float(row['high'][_XLD])
        if low > high+1e-8 or low < -1e-8:
            raise ValueError('Invalid executed XLD release interval')
        coefficient = self.physics.rho_g_eta*head/1e6
        raw_shortfall = max(self.threshold-power, 0.)
        shortfall = raw_shortfall if raw_shortfall > self.tolerance else 0.
        reference_power = min(float(self.data.power_cap[_XLD]),
                              self.physics.rho_g_eta*self.reference_head*turbine/1e6)
        water_upper = float(row['inflow'][_XLD])+(float(row['storage_start'][_XLD])-float(row['storage_lower_m3'][_XLD]))/self.physics.dt
        _, downstream_upper = self.physics.downstream_intervals(
            self._vector(row['storage_start'], 'storage_start'), self._vector(row['external'], 'external'),
            self._vector(row['storage_lower_m3'], 'storage_lower_m3'), self._vector(row['storage_upper_m3'], 'storage_upper_m3'))
        levels_start = self.physics._forebay_levels(self._vector(row['storage_start'], 'storage_start'))
        levels_end = self.physics._forebay_levels(self._vector(row['storage_end'], 'storage_end'))
        sampled = row.get('sampled_actions')
        if sampled is not None:
            chosen_actions = self._chosen_actions(row)
            sampled_index = int(round(chosen_actions[_XLD]*(self.bins-1)))
        else:
            sampled_index = None
        record = dict.fromkeys(XLD_DIAGNOSTIC_FIELDS)
        record.update(phase=self.phase, seed=self.seed, evaluation_round=self.evaluation_round,
            window_id=self.window_id, episode_step=step, date=str(row['date']),
            storage_start_m3=float(row['storage_start'][_XLD]), storage_end_m3=float(row['storage_end'][_XLD]),
            level_start_m=float(levels_start[_XLD]), level_end_m=float(levels_end[_XLD]),
            storage_lower_m3=float(row['storage_lower_m3'][_XLD]), storage_upper_m3=float(row['storage_upper_m3'][_XLD]),
            upstream_release_m3s=float(row['release'][_XLD-1]), external_inflow_m3s=float(row['external'][_XLD]),
            inflow_m3s=float(row['inflow'][_XLD]), release_m3s=float(row['release'][_XLD]),
            turbine_m3s=turbine, spill_m3s=float(row['spill'][_XLD]), net_head_m=head,
            reference_head_m=self.reference_head, power_mw=power, guaranteed_output_mw=self.threshold,
            shortfall_mw=shortfall, required_turbine_flow_at_actual_head_m3s=self.threshold/coefficient,
            required_head_at_actual_turbine_m=(self.threshold*1e6/(self.physics.rho_g_eta*turbine) if turbine else None),
            power_at_reference_head_same_turbine_mw=reference_power, sampled_action_index=sampled_index,
            action_bins=self.bins, forced_action=bool(row['controls'][_XLD] <= 0.),
            feasible_release_low_m3s=low, feasible_release_high_m3s=high,
            water_budget_upper_m3s=water_upper, downstream_upper_m3s=float(downstream_upper[_XLD]),
            outlet_capacity_m3s=float(self.data.release_cap[_XLD]), turbine_capacity_m3s=float(self.data.turbine_cap[_XLD]),
            installed_capacity_mw=float(self.data.power_cap[_XLD]), violating=bool(shortfall > 0.),
            classification='requirement_inactive' if not self.threshold else 'no_violation',
            head_contribution_at_executed_release=bool(shortfall and self.threshold-reference_power <= self.tolerance),
            water_limit_proven=False, dispatch_limit_proven=False)
        if not shortfall:
            return record

        forced = high-low <= 1e-8
        grid_actions = np.linspace(0., 1., self.bins)
        releases = (np.full(self.bins, float(row['release'][_XLD])) if forced else
                    low+grid_actions**2*(high-low))
        powers, heads = self._local_generation(row, releases)
        best = int(np.argmax(powers))
        best_replay = self._replay(row, float(grid_actions[best]), float(releases[best]), float(powers[best]))
        best_power = float(best_replay['power'][_XLD])
        meeting = np.flatnonzero(self.threshold-powers <= self.tolerance)
        record.update(best_grid_action_index=best, best_grid_power_mw=best_power,
            best_grid_release_m3s=float(best_replay['release'][_XLD]),
            best_grid_head_m=float(best_replay['net_head_m'][_XLD]), meeting_grid_action_count=int(len(meeting)),
            best_grid_other_guaranteed_violations=self._other_violations(best_replay, 'guaranteed'),
            actual_other_guaranteed_violations=self._other_violations(row, 'guaranteed'),
            best_grid_other_ecological_violations=self._other_violations(best_replay, 'ecological'),
            actual_other_ecological_violations=self._other_violations(row, 'ecological'))
        if len(meeting):
            first = int(meeting[0])
            witness = self._replay(row, float(grid_actions[first]), float(releases[first]), float(powers[first]))
            if self.threshold-float(witness['power'][_XLD]) > self.tolerance:
                raise XLDDispatchDiagnosticError('A minimum-meeting witness failed its power requirement')
            record.update(minimum_meeting_action_index=first,
                minimum_meeting_release_m3s=float(witness['release'][_XLD]),
                meeting_release_delta_m3s=float(witness['release'][_XLD]-row['release'][_XLD]))
        candidates = self._continuous_candidates(row, low, high)
        continuous_powers, _ = self._local_generation(row, candidates)
        index = int(np.argmax(continuous_powers))
        release = float(candidates[index])
        normalized = .5 if forced else float(np.sqrt(np.clip((release-low)/(high-low), 0., 1.)))
        continuous = self._replay(row, normalized, release, float(continuous_powers[index]))
        continuous_power = float(continuous['power'][_XLD])
        record.update(continuous_max_power_mw=continuous_power,
            continuous_best_release_m3s=float(continuous['release'][_XLD]),
            continuous_best_head_m=float(continuous['net_head_m'][_XLD]))
        max_coefficient = self.physics.rho_g_eta*self.maximum_head/1e6
        hardware_upper = min(float(self.data.power_cap[_XLD]), max_coefficient*float(self.data.turbine_cap[_XLD]))
        water_power_upper = max(water_upper, 0.)*max_coefficient
        dispatch_power_upper = min(float(self.data.power_cap[_XLD]), max_coefficient*min(max(high, 0.),float(self.data.turbine_cap[_XLD])))
        record['water_limit_proven'] = bool(self.threshold-water_power_upper > self.tolerance)
        record['dispatch_limit_proven'] = bool(self.threshold-dispatch_power_upper > self.tolerance)
        if self.threshold-best_power <= self.tolerance:
            classification = 'same_day_policy_alternative'
        elif self.threshold-continuous_power <= self.tolerance:
            classification = 'discrete_grid_gap'
        elif self.threshold-hardware_upper > self.tolerance:
            classification = 'same_state_capacity_limit'
        elif record['water_limit_proven']:
            classification = 'same_state_water_limit'
        elif record['dispatch_limit_proven']:
            classification = 'same_state_dispatch_limit'
        else:
            classification = 'same_state_head_limit'
        record['classification'] = classification
        return record

    def summary(self):
        return dict(reservoir_id='XLD', schema='xld_dispatch_v1',
            seed=self.seed, evaluation_round=self.evaluation_round,
            window_id=self.window_id, phase=self.phase,
            guaranteed_output_mw=self.threshold, action_bins=self.bins,
            head_model=self.physics.head_model,
            completed_steps=self.steps, active_days=self.active_days, violation_count=self.violations,
            violation_rate=self.violations/self.active_days if self.active_days else None,
            classification_counts=dict(self.counts),
            classification_definitions=dict(CLASSIFICATION_DEFINITIONS),
            max_shortfall_mw=self.max_shortfall,
            shortfall_energy_mwh=self.shortfall_sum*self.physics.dt/3600.,
            head_contribution_at_executed_release_days=self.head_contribution_days,
            counterfactual_scope='same starting state, current external inflows and next-state bounds; '
                'only XLD action changes; other actors retain original sampled actions when available '
                '(executed normalized actions are a fallback only for unforced raw physics fixtures)',
            candidate_verification='all discrete actions evaluated with exact local head/storage mapping; '
                'best grid, minimum-meeting and continuous maximum witnesses replayed through full cascade physics',
            limitation='Same-day attainability does not establish long-horizon optimality or attribute a poor starting state '
                'to exogenous hydrology. Water/head restrictions can coexist; proof flags supplement the ordered classifications. '
                'Reference-head power is an accounting comparison, not an executed or guaranteed-feasible higher-head state.',
            row_scope='one row per executed XLD evaluation day; detailed alternative-action search only on violating days',
            numeric_tolerance='same requirement-relative 1e-10 tolerance as OperationReward; CSV formatting only uses 10 significant digits')

_XLD_CONTINUOUS_LIMIT_CLASSES = (
    'same_state_capacity_limit', 'same_state_water_limit',
    'same_state_dispatch_limit', 'same_state_head_limit',
)
_XLD_AGGREGATE_GROUPS = {
    'actor_missed_feasible_discrete': ('same_day_policy_alternative',),
    'discrete_grid_gap': ('discrete_grid_gap',),
    'same_state_continuous_unattainable': _XLD_CONTINUOUS_LIMIT_CLASSES,
}
_XLD_AGGREGATE_SCHEMA = 'xld_dispatch_aggregate_v1'


def _summary_fraction(value, expected, name):
    if expected is None:
        if value is not None:
            raise ValueError(name+' must be null when its denominator is zero')
    elif not np.isclose(_number(value, name), expected, rtol=1e-12, atol=1e-12):
        raise ValueError(name+' disagrees with its counts')


def _summary_identity_list(values, name, *, text=False):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(name+' must be a nonempty list')
    if text:
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(name+' must contain nonempty text')
        return list(values)
    return [_integer(value, name) for value in values]


def aggregate_xld_summaries(summaries):
    """Pool complete XLD evaluation summaries without changing any trajectory.

    A raw collector summary does not contain the requested episode horizon.
    Callers must supply only successful complete windows. Explicit failure or
    incomplete markers, and any supplied episode_horizon_days, are validated.
    Existing aggregate summaries can be pooled again without losing counts.
    """
    try:
        if not isinstance(summaries, (list, tuple)) or not summaries:
            raise ValueError('XLD aggregation requires a nonempty list of complete summaries')
        identity = None
        totals = dict(completed_steps=0, active_days=0, violation_count=0)
        counts = dict.fromkeys(CLASSIFICATION_DEFINITIONS, 0)
        component_count, window_ids, seeds, rounds = 0, [], set(), set()
        for summary in summaries:
            if not isinstance(summary, dict):
                raise ValueError('Every XLD summary must be a nonnull dictionary')
            schema = summary['schema']
            if schema not in ('xld_dispatch_v1', _XLD_AGGREGATE_SCHEMA):
                raise ValueError('Unknown XLD summary schema')
            if summary.get('status', 'ok') != 'ok':
                raise ValueError('Failed or partial XLD summaries cannot be pooled as complete windows')
            for name in ('complete', 'completed'):
                if name in summary and (not isinstance(summary[name], (bool, np.bool_)) or
                                        not bool(summary[name])):
                    raise ValueError('Incomplete XLD summaries cannot be pooled')
            threshold = _number(summary['guaranteed_output_mw'], 'guaranteed_output_mw')
            if threshold < 0.:
                raise ValueError('guaranteed_output_mw cannot be negative')
            bins = _integer(summary['action_bins'], 'action_bins', 2)
            head_model, phase = summary['head_model'], summary['phase']
            if summary['reservoir_id'] != 'XLD':
                raise ValueError('Only XLD summaries can be pooled')
            if head_model not in ('fixed', 'storage_sensitivity'):
                raise ValueError('Unknown XLD head model')
            if phase not in ('training_evaluation', 'evaluation'):
                raise ValueError('XLD summaries must identify one evaluation phase')
            current_identity = ('XLD', threshold, bins, head_model, phase)
            if identity is None:
                identity = current_identity
            elif identity != current_identity:
                raise ValueError('XLD summaries have inconsistent reservoir, threshold, grid, head model or phase')
            days = _integer(summary['completed_steps'], 'completed_steps', 1)
            active = _integer(summary['active_days'], 'active_days')
            violations = _integer(summary['violation_count'], 'violation_count')
            if not 0 <= violations <= active <= days:
                raise ValueError('XLD day and violation counts are inconsistent')
            if 'episode_horizon_days' in summary:
                if _integer(summary['episode_horizon_days'], 'episode_horizon_days', 1) != days:
                    raise ValueError('XLD summary is shorter than its complete episode horizon')
            classes = summary['classification_counts']
            if not isinstance(classes, dict) or set(classes) != set(CLASSIFICATION_DEFINITIONS):
                raise ValueError('XLD classification counts have missing or unknown classes')
            classes = {name: _integer(value, 'classification_counts.'+name)
                       for name, value in classes.items()}
            if sum(classes.values()) != days:
                raise ValueError('XLD classifications do not reconcile with completed steps')
            if active != (days if threshold > 0. else 0):
                raise ValueError('XLD active-day count disagrees with the effective requirement')
            if classes['requirement_inactive'] != days-active or classes['no_violation'] != active-violations:
                raise ValueError('XLD inactive/nonviolating classifications disagree with service counts')
            groups = {name: sum(classes[term] for term in terms)
                      for name, terms in _XLD_AGGREGATE_GROUPS.items()}
            if sum(groups.values()) != violations:
                raise ValueError('XLD violating classifications do not reconcile with violation count')
            _summary_fraction(summary['violation_rate'],
                              violations/active if active else None, 'violation_rate')
            if schema == _XLD_AGGREGATE_SCHEMA or 'group_counts' in summary:
                supplied = summary['group_counts']
                if not isinstance(supplied, dict) or set(supplied) != set(groups):
                    raise ValueError('XLD aggregate groups have missing or unknown classes')
                if any(_integer(supplied[name], 'group_counts.'+name) != groups[name] for name in groups):
                    raise ValueError('XLD aggregate groups disagree with classification counts')
            if schema == _XLD_AGGREGATE_SCHEMA or 'group_fractions_among_violations' in summary:
                fractions = summary['group_fractions_among_violations']
                if not isinstance(fractions, dict) or set(fractions) != set(groups):
                    raise ValueError('XLD aggregate fractions have missing or unknown groups')
                for name in groups:
                    _summary_fraction(fractions[name], groups[name]/violations if violations else None,
                                      'group_fractions_among_violations.'+name)
            if schema == _XLD_AGGREGATE_SCHEMA:
                components = _integer(summary['component_count'], 'component_count', 1)
                windows = _summary_identity_list(summary['window_ids'], 'window_ids', text=True)
                if len(windows) != components:
                    raise ValueError('XLD aggregate window identities do not match its component count')
                source_seeds = _summary_identity_list(summary['seeds'], 'seeds')
                source_rounds = _summary_identity_list(summary['evaluation_rounds'], 'evaluation_rounds')
            else:
                components = 1
                windows = _summary_identity_list([summary['window_id']], 'window_id', text=True)
                source_seeds = [_integer(summary['seed'], 'seed')]
                source_rounds = [_integer(summary['evaluation_round'], 'evaluation_round')]
            component_count += components
            window_ids.extend(windows)
            seeds.update(source_seeds)
            rounds.update(source_rounds)
            totals['completed_steps'] += days
            totals['active_days'] += active
            totals['violation_count'] += violations
            for name in counts:
                counts[name] += classes[name]

        groups = {name: sum(counts[term] for term in terms)
                  for name, terms in _XLD_AGGREGATE_GROUPS.items()}
        violations, active = totals['violation_count'], totals['active_days']
        return dict(schema=_XLD_AGGREGATE_SCHEMA, status='ok', reservoir_id=identity[0],
            guaranteed_output_mw=identity[1], action_bins=identity[2], head_model=identity[3],
            phase=identity[4], component_count=component_count, window_ids=window_ids,
            seeds=sorted(seeds), evaluation_rounds=sorted(rounds), **totals,
            violation_rate=violations/active if active else None,
            classification_counts=counts, classification_definitions=dict(CLASSIFICATION_DEFINITIONS),
            group_counts=groups,
            group_fractions_among_violations={name: value/violations if violations else None
                                             for name, value in groups.items()},
            aggregation='pooled counts over complete supplied XLD windows; rates are ratios of pooled counts',
            group_denominator='observed XLD guaranteed-output violations, not all XLD days or all-reservoir days',
            counterfactual_scope='same starting state, current external inflows and next-state bounds; '
                'only XLD command changes and the other four original sampled commands stay fixed; '
                'XLD inflow stays fixed, while downstream realized releases and power may change',
            limitation='A feasible discrete alternative is not proof of a globally wrong action or a better '
                'multi-day policy. Same-state continuous unattainability is conditional, not a proof of '
                'unavoidable hydrological shortage or impossibility under earlier storage decisions or '
                'different upstream cooperation. Alternative commands may change other service violations. '
                'The four physical classes are ordered attribution labels; overlapping proof flags are not added.')
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise XLDDispatchDiagnosticError('Invalid complete XLD summary aggregation: '+str(exc)) from exc

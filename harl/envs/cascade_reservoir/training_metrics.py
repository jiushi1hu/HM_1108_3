"""Read-only accounting of sampled episodes, separate from rewards and updates."""
import csv
import json
import numpy as np
from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.reward import PENALTY_TERMS, REWARD_AGGREGATION, REWARD_DEFINITION


STEP_METRIC_FIELDS = (
    'phase', 'seed', 'episode', 'episode_step', 'date', 'training_steps',
    'reward', 'generation_reward', 'guaranteed_output_penalty', 'ecological_flow_penalty',
    'generation_MWh', 'guaranteed_output_shortfall_MW', 'guaranteed_output_shortfall_MWh',
    'ecological_flow_shortfall_m3s', 'ecological_flow_deficit_m3',
)
_STEP_VALUE_FIELDS = STEP_METRIC_FIELDS[6:]
_OPTIONAL_STEP_FIELDS = STEP_METRIC_FIELDS[11:]


def _step_integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(name+' must be an integer >= '+str(minimum))
    return int(value)


def _step_number(value, name):
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(name+' must be a finite scalar') from exc
    if not np.isfinite(number):
        raise ValueError(name+' must be a finite scalar')
    return number


def _step_sum(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(IDS),) or not np.isfinite(values).all() or (values < 0.).any():
        raise ValueError(name+' must be a finite nonnegative five-vector')
    return _step_number(values.sum(), name)


def step_record(transition, *, seed, episode, episode_step, phase, training_steps):
    """Key team metrics for one executed day, without a physical trajectory.

    Reward and weighted penalty components retain the legacy undiscounted team
    metric. HAPPO learns from system_reward; MAPPO retains mixed_rewards.
    Both are recorded separately in episode summaries. Generation and physical shortfalls are sums
    across the five reservoirs. An unavailable service measurement is None;
    a measured zero shortfall remains zero, even with reward penalties disabled.
    Values retain full precision until the CSV writer serializes them.
    """
    if phase not in ('training', 'evaluation', 'training_evaluation'):
        raise ValueError('Unknown daily log phase: '+str(phase))
    costs = transition.get('reward_penalty_components', {})
    record = dict(phase=phase, seed=_step_integer(seed, 'seed'),
        episode=_step_integer(episode, 'episode'),
        episode_step=_step_integer(episode_step, 'episode_step', 1),
        date=str(transition['date']), training_steps=_step_integer(training_steps, 'training_steps'),
        reward=_step_number(transition['reward'], 'reward'),
        generation_reward=_step_number(transition.get('generation_reward', transition['reward']),
                                       'generation_reward'),
        guaranteed_output_penalty=_step_number(costs.get('guaranteed_output', 0.),
                                               'guaranteed_output_penalty'),
        ecological_flow_penalty=_step_number(costs.get('ecological_flow', 0.),
                                             'ecological_flow_penalty'),
        generation_MWh=_step_sum(transition['energy'], 'generation_MWh'))
    measurements = (
        ('guaranteed_output_shortfall_MW', 'service_shortfalls', 'guaranteed_output'),
        ('guaranteed_output_shortfall_MWh', 'service_shortfall_integrals', 'guaranteed_output'),
        ('ecological_flow_shortfall_m3s', 'service_shortfalls', 'ecological_flow'),
        ('ecological_flow_deficit_m3', 'service_shortfall_integrals', 'ecological_flow'),
    )
    for name, source, term in measurements:
        values = transition.get(source, {}).get(term)
        record[name] = None if values is None else _step_sum(values, name)
    return record


class StepMetricsWriter:
    """A fixed, scalar CSV table; write one header and flush every daily row.

    Ten significant digits limit storage while preserving useful precision in
    the logged metrics. Formatting never changes the transition or training.
    Missing service measurements are empty CSV cells, distinct from zero.
    """

    def __init__(self, stream):
        self.stream = stream
        self.writer = csv.writer(stream, lineterminator='\n')
        self.writer.writerow(STEP_METRIC_FIELDS)
        self.stream.flush()

    def write(self, record):
        if set(record) != set(STEP_METRIC_FIELDS):
            raise ValueError('Daily CSV rows must contain exactly STEP_METRIC_FIELDS')
        row = []
        for name in STEP_METRIC_FIELDS:
            value = record[name]
            if name in _STEP_VALUE_FIELDS:
                if value is None and name in _OPTIONAL_STEP_FIELDS:
                    value = ''
                else:
                    value = format(_step_number(value, name), '.10g')
            row.append(value)
        self.writer.writerow(row)
        self.stream.flush()


def log_step(record, writer):
    """Persist one compact daily row without printing it or retaining history."""
    if writer is not None:
        writer.write(record)


def episode_record(env, rewards, energy_by_reservoir, seed, episode, gamma,
                   gae_lambda, failed_episodes, failure=None):
    """Describe real executed days; a rejected transition contributes no energy.

    Failure rates count failed episode attempts, not soft reservoir-day events.
    The current runner still stops on a hard failure after emitting this record.
    """
    days = env.index-env.start
    completed = failure is None and env.index == env.stop
    failed = failure is not None
    if not (completed or failed):
        raise ValueError('Episode accounting requires completion or an explicit failure')
    energy = np.asarray(energy_by_reservoir, dtype=np.float64)
    if energy.shape != (5,) or not np.isfinite(energy).all() or (energy < 0.).any():
        raise ValueError('Generation must be a finite nonnegative five-vector')
    totals = rewards.summary()
    if rewards.gamma != gamma or rewards.steps != days:
        raise ValueError('Episode accounting must use the training gamma and executed days')
    power = totals['episode_generation_reward']
    costs = {k: totals['episode_penalty_'+k] for k in PENALTY_TERMS}
    if not np.isclose(rewards.undiscounted_generation_reward,
                      energy.sum()/env.physics.reference_energy, rtol=1e-10, atol=1e-9):
        raise ValueError('Power Reward and actual generation disagree')
    if not np.isclose(totals['episode_reward'], power-sum(costs.values()), rtol=1e-10, atol=1e-9):
        raise ValueError('Reward components do not sum to Total Reward')
    percentages = {k: 100.*v/power if power > 0. else None for k,v in costs.items()}
    percentages['total'] = 100.*sum(costs.values())/power if power > 0. else None
    requirements = getattr(env.data, 'service_requirements', None)
    return dict(record_type='training_episode', seed=int(seed), episode=int(episode),
        window_start_year=int(env.data.dates[env.start].year), completed=completed,
        source_years=list(dict.fromkeys(day.year for day in env.data.dates[env.start:env.stop])),
        episode_horizon_days=int(env.stop-env.start),
        start_date=env.data.dates[env.start].isoformat(),
        end_date=env.data.dates[env.stop-1].isoformat(),
        calendar_days=int(env.stop-env.start), simulated_days=int(days),
        actual_generation_MWh=float(energy.sum()), actual_generation_TWh=float(energy.sum()/1e6),
        episode_generation_TWh=float(energy.sum()/1e6) if completed else None,
        generation_MWh_by_reservoir=energy.tolist(), reservoir_ids=list(IDS),
        power_reward=power, **{k+'_penalty':v for k,v in costs.items()},
        total_soft_penalty=sum(costs.values()),
        total_reward=totals['episode_reward'], penalty_to_power_percent=percentages,
        learning_rewards=rewards.learning_summary(),
        hard_constraint_failed=failed, hard_constraint_failure_rate=float(failed),
        hard_constraint_failure_rate_so_far=failed_episodes/float(episode),
        attempted_episodes=int(episode), failed_episodes_so_far=int(failed_episodes), failure=failure,
        service_requirements_available=requirements is not None,
        service_shortfalls=rewards.shortfall_summary(),
        service_requirements=None if requirements is None else requirements.metadata(),
        gamma=float(gamma), gae_lambda=float(gae_lambda),
        reward_aggregation=REWARD_AGGREGATION, reward_definition=REWARD_DEFINITION,
        **env.observation_metadata(),
        reward_penalties=dict(env.reward_settings), reference_energy_MWh=env.physics.reference_energy,
        release_bounds_scope=('episode_terminal_reachability' if getattr(
            env, 'terminal_storage_equal_initial', False) else 'current_transition'))


def log_episode(record, stream=None):
    """Always print JSON; optionally flush one JSONL record to an explicit file."""
    encoded = json.dumps(record, allow_nan=False)
    if stream is not None:
        stream.write(encoded+'\n')
        stream.flush()
    print('EPISODE_METRICS '+encoded, flush=True)

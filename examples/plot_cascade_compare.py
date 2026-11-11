"""Three matched cascade figures, with separate HAPPO and MAPPO seed means."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from examples import plot_cascade_parallel as legacy


SCHEMA = 'cascade_comparison_v1'
COLORS = {'happo': '#c0392b', 'mappo': '#2563eb'}
SHARED_FIELDS = (
    'episode_horizon_days', 'parallel_environments', 'episodes_per_seed',
    'gamma', 'action_bins', 'seeds', 'training_years', 'evaluation_source_years',
    'window_sampling', 'physical_config', 'periodic_windows', 'evaluation_interval',
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def _normalized_context(context):
    """Sharing is the intended architecture difference, not a task difference."""
    result = copy.deepcopy(context)
    settings = result.get('shared_settings', {})
    if isinstance(settings, dict) and isinstance(settings.get('algo'), dict):
        settings['algo'].pop('share_param', None)
    return result


def context_from_configuration(metadata):
    """Extract matched settings without output paths or algorithm labels."""
    if 'comparison_context' in metadata:
        return _normalized_context(metadata['comparison_context'])
    config = metadata['config']
    experiment = config['experiment']
    shared = dict(
        episode_horizon_days=metadata['episode_horizon_days'],
        parallel_environments=metadata['parallel_environments'],
        episodes_per_seed=metadata['episodes_per_seed'],
        gamma=config['algo']['gamma'], action_bins=config['model']['action_bins'],
        seeds=experiment['seeds'], training_years=metadata['training_years'],
        evaluation_source_years=metadata['evaluation_source_years'],
        window_sampling=experiment.get('window_sampling', 'coverage_balanced'),
        physical_config=metadata['physical_config'],
        periodic_windows=metadata['periodic_evaluation']['windows'],
        evaluation_interval=experiment.get('eval_interval', 1),
    )
    architecture = metadata.get('learning_reward_architecture', {})
    if 'actor_temporal_weighting' in architecture:
        shared['actor_temporal_weighting'] = architecture['actor_temporal_weighting']
    for key in ('device', 'train', 'model', 'algo'):
        if key in config:
            shared[key] = config[key]
    return _normalized_context(dict(shared_settings=shared, data_hashes=metadata['data_hashes']))


def _context(payload):
    context = payload.get('comparison_context')
    if not isinstance(context, dict):
        raise ValueError('Every algorithm payload requires comparison_context')
    settings, hashes = context.get('shared_settings'), context.get('data_hashes')
    if not isinstance(settings, dict) or any(key not in settings for key in SHARED_FIELDS):
        raise ValueError('comparison_context.shared_settings requires: '+', '.join(SHARED_FIELDS))
    if (not isinstance(hashes, dict) or not hashes or
            any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in hashes.items())):
        raise ValueError('comparison_context requires nonempty input data_hashes')
    _canonical(context)  # Reject NaN and non-JSON values before drawing.
    for key in ('episode_horizon_days', 'parallel_environments', 'episodes_per_seed',
                'action_bins', 'evaluation_interval'):
        legacy.positive_integer(settings[key], key)
    if settings['action_bins'] < 2:
        raise ValueError('action_bins must be at least two')
    if settings['episodes_per_seed'] % settings['parallel_environments']:
        raise ValueError('episodes_per_seed must be divisible by parallel_environments')
    gamma = settings['gamma']
    if isinstance(gamma, bool) or not np.isfinite(float(gamma)) or not 0 <= float(gamma) <= 1:
        raise ValueError('Comparison gamma must be finite and in [0, 1]')
    for key in ('seeds', 'training_years', 'evaluation_source_years'):
        values = settings[key]
        if (not isinstance(values, list) or not values or
                any(isinstance(v, bool) or not isinstance(v, int) for v in values) or
                len(set(values)) != len(values)):
            raise ValueError(key+' must contain unique integers')
    if set(settings['training_years']) & set(settings['evaluation_source_years']):
        raise ValueError('Training and held-out source years must be disjoint')
    if settings['window_sampling'] not in ('coverage_balanced', 'uniform'):
        raise ValueError('Unknown comparison training-window sampling mode')
    if not isinstance(settings['physical_config'], dict) or not settings['physical_config']:
        raise ValueError('Comparison physical_config must be a nonempty mapping')
    expected_signature = legacy.window_signature(settings['periodic_windows'])
    for window in settings['periodic_windows']:
        if not set(window['source_years']) <= set(settings['training_years']):
            raise ValueError('Periodic comparison windows must contain training source years only')
    if float(payload['gamma']) != float(gamma):
        raise ValueError('Plot gamma disagrees with comparison_context')
    if payload['episode_horizon_days'] != settings['episode_horizon_days']:
        raise ValueError('Plot episode horizon disagrees with comparison_context')
    if ('parallel_envs' in payload and
            payload['parallel_envs'] != settings['parallel_environments']):
        raise ValueError('Plot worker count disagrees with comparison_context')
    updates = settings['episodes_per_seed']//settings['parallel_environments']
    if 'updates_per_seed' in payload and payload['updates_per_seed'] != updates:
        raise ValueError('Plot update budget disagrees with comparison_context')
    seeds = payload['seeds']
    if (not isinstance(seeds, list) or not seeds or
            any(isinstance(v, bool) or not isinstance(v, int) for v in seeds) or
            len(set(seeds)) != len(seeds) or not set(seeds) <= set(settings['seeds'])):
        raise ValueError('Plot seeds must be a nonempty subset of configured comparison seeds')
    if payload.get('configured_seeds', settings['seeds']) != settings['seeds']:
        raise ValueError('Configured plot seeds disagree with comparison_context')
    return _normalized_context(context), expected_signature


def _validate_architecture(algorithm, run):
    """Accept legacy independent actors or explicit ID-conditioned shared MAPPO."""
    settings = run['comparison_context']['shared_settings']
    requested = settings.get('algo', {}).get('share_param')
    if requested is not None and not isinstance(requested, bool):
        raise ValueError('Actor share_param must be a boolean')
    if algorithm == 'happo' and requested:
        raise ValueError('HAPPO comparison requires independent actor parameters')
    architecture = run.get('actor_architecture')
    objective = run.get('learning_objective')
    critic_dim = 6  # Historical runs used individual mixed values plus legacy.
    if objective is not None:
        if not isinstance(objective, dict):
            raise ValueError('learning_objective must be a mapping')
        counterfactual = (algorithm == 'happo' and
            settings.get('algo', {}).get('happo_credit_assignment','team_gae') == 'counterfactual')
        team = algorithm == 'happo' and objective.get('advantage_source') in ('shared_team','counterfactual_shared_team')
        expected_objective = dict(
            advantage_source='counterfactual_shared_team' if counterfactual else 'shared_team' if team else 'individual_mixed',
            reward_formula=('0.5 * sum(local rewards) + 0.5 * global generation reward'
                            if team else '0.5 * own local reward + 0.5 * global generation reward'),
            critic_reward_layout='shared_team' if team else 'mixed_by_reservoir_then_legacy_team',
            actor_value_channels=[0]*5 if team else list(range(5)),
            normalization=('common_controllable_rms_without_centering' if counterfactual else
                           'once_over_common_rollout_rows' if team
                           else 'per_reservoir_over_common_rollout_rows'
                           if settings.get('algo', {}).get('normalize_shared_advantages', False)
                           else 'per_reservoir_over_controllable_rows'))
        if objective != expected_objective:
            raise ValueError('Learning objective does not match the algorithm reward/value layout')
        critic_dim = 1 if team else 6
    if architecture is None:
        if requested:
            raise ValueError('Shared MAPPO requires explicit actor_architecture metadata')
        return  # Older labeled MAPPO and HAPPO logs used five independent actors.
    if not isinstance(architecture, dict):
        raise ValueError('actor_architecture must be a mapping')
    shared = architecture.get('parameter_sharing')
    if not isinstance(shared, bool) or (algorithm == 'happo' and shared):
        raise ValueError('Only MAPPO may use an ID-conditioned shared actor')
    if requested is not None and requested != shared:
        raise ValueError('Actor architecture disagrees with share_param')
    raw_dim = 15 if settings['physical_config'].get('use_daily_level_limits', True) else 11
    if 'five_reservoir_next7d_mean_inflow_forecast_1965_2024.csv' in run['comparison_context']['data_hashes']:
        raw_dim += 1
    expected = dict(network_count=1 if shared else 5, optimizer_count=1 if shared else 5,
        raw_observation_dim=raw_dim, network_observation_dim=raw_dim+5 if shared else raw_dim,
        identity_features='one_hot_reservoir_id' if shared else 'implicit_in_separate_actor',
        critic_output_dim=critic_dim)
    if any(architecture.get(key) != value for key, value in expected.items()):
        raise ValueError('Actor architecture dimensions/counts do not match the cascade observations')


def validate_comparison(payload):
    """Validate identity and each legacy history; return a normalized outer payload."""
    if not isinstance(payload, dict) or payload.get('schema') != SCHEMA:
        raise ValueError('Expected comparison schema '+SCHEMA)
    runs = payload.get('runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError('Comparison requires at least one algorithm run')
    normalized, names, reference = [], set(), None
    for entry in runs:
        if not isinstance(entry, dict) or not isinstance(entry.get('algorithm'), str):
            raise ValueError('Every run requires an algorithm name and legacy payload')
        algorithm = entry['algorithm'].lower()
        if algorithm not in COLORS or algorithm in names:
            raise ValueError('Algorithms must be unique HAPPO/MAPPO entries')
        names.add(algorithm)
        run = entry.get('payload')
        if not isinstance(run, dict):
            raise ValueError('Every algorithm run requires a legacy plotting payload')
        if 'algorithm' in run and str(run['algorithm']).lower() != algorithm:
            raise ValueError('Algorithm label disagrees with its plotting payload')
        context, signature = _context(run)
        _validate_architecture(algorithm, run)
        identity = _canonical(context)
        if reference is not None and identity != reference:
            raise ValueError('Comparison settings or input data_hashes differ between algorithms')
        reference = identity
        legacy.available_histories(run, 'histories')
        for rows in run.get('histories', []):
            steps = [legacy.positive_integer(r['training_steps'], 'training_steps') for r in rows]
            if any(b <= a for a, b in zip(steps, steps[1:])):
                raise ValueError('Training steps must increase without duplicates')
            if any(not np.isfinite(float(r[key])) for r in rows
                   for key in ('episode_reward', 'episode_energy_TWh')):
                raise ValueError('Training curve metrics must be finite')
        _, histories, _ = legacy.periodic_histories(run)
        for rows in histories:
            for row in rows:
                legacy.validate_periodic_row(row, signature)
                if 'gamma' in row and float(row['gamma']) != float(run['gamma']):
                    raise ValueError('Periodic gamma differs from plot gamma')
        if not legacy.violation_metrics_available(run):
            raise ValueError('Three comparison figures require actual periodic violation counts')
        normalized.append(dict(algorithm=algorithm, payload=run))
    return dict(payload, runs=normalized)


def load_comparison(happo_log=None, mappo_log=None):
    """Load separate logs: legacy integer seed identifiers remain isolated."""
    paths = [Path(path).resolve() for path in (happo_log, mappo_log) if path is not None]
    if len(set(paths)) != len(paths):
        raise ValueError('The same log cannot represent two different algorithms')
    runs = []
    for algorithm, path in (('happo', happo_log), ('mappo', mappo_log)):
        if path is None:
            continue
        run = legacy.load_episode_log(path)
        metadata = None
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('record_type') == 'run_configuration':
                    metadata = row
                    break
        if metadata is None:
            raise ValueError('A run_configuration is required for comparison')
        recorded = metadata.get('algorithm', metadata.get('algorithm_name'))
        if recorded is None and algorithm == 'mappo':
            raise ValueError('A MAPPO log requires explicit algorithm metadata; unlabeled legacy logs are HAPPO')
        if recorded is not None and str(recorded).lower() != algorithm:
            raise ValueError('The supplied '+algorithm+' log records a different algorithm')
        run['comparison_context'] = context_from_configuration(metadata)
        if 'learning_objective' in metadata:
            run['learning_objective'] = metadata['learning_objective']
        if 'actor_architecture' in metadata:
            run['actor_architecture'] = copy.deepcopy(metadata['actor_architecture'])
        runs.append(dict(algorithm=algorithm, payload=run))
    return validate_comparison(dict(schema=SCHEMA, runs=runs))


def _label(algorithm, count, run):
    total = len(run['comparison_context']['shared_settings']['seeds'])
    method = algorithm.upper()
    if run['comparison_context']['shared_settings'].get('algo',{}).get('happo_credit_assignment') == 'counterfactual':
        method += ' counterfactual'
    return method+(' mean' if count == total else
                              ' available-seed mean (n={})'.format(count))


def _line_and_band(ax, grid, aligned, algorithm, run):
    valid = np.isfinite(aligned).all(axis=0)
    mean = np.full(len(grid), np.nan)
    mean[valid] = aligned[:, valid].mean(axis=0)
    color = COLORS[algorithm]
    ax.plot(grid, mean, color=color, lw=2,
            linestyle='-' if algorithm == 'happo' else '--',
            label=_label(algorithm, len(aligned), run))
    if len(aligned) > 1:
        sd = np.full(len(grid), np.nan)
        sd[valid] = aligned[:, valid].std(axis=0, ddof=1)
        ax.fill_between(grid, mean-sd, mean+sd, color=color, alpha=.15,
                        label=algorithm.upper()+' +/-1 SD')


def _training(ax, algorithm, run, metric):
    _, histories = legacy.available_histories(run, 'histories')
    if not histories:
        return
    common = set(r['training_steps'] for r in histories[0])
    for rows in histories[1:]:
        common.intersection_update(r['training_steps'] for r in rows)
    grid = np.array(sorted(common), dtype=float)
    aligned = np.array([[{r['training_steps']: r[metric] for r in rows}[step]
                         for step in grid] for rows in histories], dtype=float)
    _line_and_band(ax, grid, aligned, algorithm, run)


def _periodic(ax, algorithm, run, metric, percent=False):
    seeds, histories, grids = legacy.periodic_histories(run)
    grid = np.unique(np.concatenate(grids))
    aligned = np.full((len(seeds), len(grid)), np.nan)
    for i, (rounds, rows) in enumerate(zip(grids, histories)):
        aligned[i, np.searchsorted(grid, rounds)] = [
            np.nan if row[metric] is None else float(row[metric])*(100. if percent else 1.)
            for row in rows]
    _line_and_band(ax, grid, aligned, algorithm, run)
    missing = ~np.isfinite(aligned).all(axis=0)
    if missing.any():
        y = .025 if algorithm == 'happo' else .055
        ax.scatter(grid[missing], np.full(int(missing.sum()), y),
                   transform=ax.get_xaxis_transform(), marker='x', color=COLORS[algorithm],
                   label=algorithm.upper()+' unavailable evaluation', clip_on=False)


def make_figures(payload):
    """Create three figures after validation, preserving five legacy metric axes."""
    payload = validate_comparison(payload)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    training, training_axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    evaluator, evaluator_ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    violations, violation_axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for entry in payload['runs']:
        algorithm, run = entry['algorithm'], entry['payload']
        for ax, metric in zip(training_axes, ('episode_reward', 'episode_energy_TWh')):
            _training(ax, algorithm, run, metric)
        _periodic(evaluator_ax, algorithm, run, 'evaluator_return')
        for ax, term in zip(violation_axes, legacy.VIOLATION_TERMS):
            _periodic(ax, algorithm, run, term+'_violation_rate', percent=True)
    shared = payload['runs'][0]['payload']['comparison_context']['shared_settings']
    for ax, label in zip(training_axes, (
            'Mean collected legacy-team return (gamma={:g})'.format(float(shared['gamma'])),
            'Mean collected episode generation (TWh)')):
        ax.set(xlabel='Collected daily environment transitions', ylabel=label,
               title='{}-day stochastic training episodes (in-sample)'.format(shared['episode_horizon_days']))
    evaluator_ax.set(ylabel='Legacy-team return (mean of three fixed training windows)',
                     title='Evaluator Return vs. Evaluation Round')
    for ax, title in zip(violation_axes, ('Ecological-flow violation rate', 'Guaranteed-output violation rate')):
        ax.set(ylabel='Violation rate (%)', title=title)
    for ax in (evaluator_ax, *violation_axes):
        ax.set(xlabel='Evaluation round', xlim=(0, shared['episodes_per_seed']//shared['parallel_environments']))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in (*training_axes, evaluator_ax, *violation_axes):
        ax.grid(alpha=.2)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8)
    violations.suptitle('Fixed-scenario evaluation: violation rates')
    objectives = {entry['algorithm']: entry['payload'].get('learning_objective', {}).get('advantage_source', 'individual_mixed')
                  for entry in payload['runs']}
    if len(set(objectives.values())) > 1:
        note = 'Different learning objectives: HAPPO shared team; MAPPO individual mixed'
        training.suptitle(note, fontsize=10)
        evaluator.suptitle(note, fontsize=10)
        violations.suptitle('Fixed-scenario violation rates\n'+note, fontsize=10)
    return dict(training=training, evaluator=evaluator, violations=violations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--happo-log', type=Path)
    parser.add_argument('--mappo-log', type=Path)
    parser.add_argument('--output', type=Path, help='Training PNG; also derives the other two paths')
    parser.add_argument('--evaluator-output', type=Path)
    parser.add_argument('--violations-output', type=Path)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.no_show and all(p is None for p in (args.output, args.evaluator_output, args.violations_output)):
        parser.error('--no-show requires an output path')
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg', force=True)
    try:
        payload = (load_comparison(args.happo_log, args.mappo_log)
                   if args.happo_log is not None or args.mappo_log is not None else json.load(sys.stdin))
        outputs = dict(training=args.output,
                       evaluator=args.evaluator_output or (legacy.evaluator_output_path(args.output) if args.output else None),
                       violations=args.violations_output or (legacy.violation_output_path(args.output) if args.output else None))
        targets = [p.resolve() for p in outputs.values() if p is not None]
        sources = {p.resolve() for p in (args.happo_log, args.mappo_log) if p is not None}
        if len(set(targets)) != len(targets) or sources.intersection(targets):
            raise ValueError('Figure outputs must be distinct and cannot overwrite input logs')
        for output in outputs.values():
            if output is not None and output.exists():
                raise ValueError('Figure output already exists: '+str(output))
        figures = make_figures(payload)
        for key, figure in figures.items():
            if outputs[key] is not None:
                outputs[key].parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(str(outputs[key]), dpi=180)
                print('Saved '+key+' comparison to '+str(outputs[key].resolve()))
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        parser.error(str(exc))
    import matplotlib.pyplot as plt
    if args.no_show:
        for figure in figures.values():
            plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()

"""Comparison plots keep algorithm identity, task provenance and failure gaps."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import plot_cascade_compare as compare


def descriptors():
    return [dict(window_id=name, window_start_year=year, source_years=[year],
                 episode_horizon_days=2, start_date=str(year)+'-01-01',
                 end_date=str(year)+'-01-02', initial_storage_m3=[1.]*5,
                 initial_level_m=[1.]*5, external_inflow_sha256='0'*64)
            for name, year in (('dry', 2000), ('median', 2001), ('wet', 2002))]


def context():
    return dict(shared_settings=dict(episode_horizon_days=2, parallel_environments=1,
        episodes_per_seed=2, gamma=1., action_bins=4, seeds=[1, 2],
        training_years=[2000, 2001, 2002], evaluation_source_years=[2003],
        window_sampling='coverage_balanced', physical_config={'head_model': 'fixed'},
        periodic_windows=descriptors(), evaluation_interval=1, device={'torch_threads': 1},
        train={'episode_length': 2, 'n_rollout_threads': 1},
        model={'action_bins': 4}, algo={'gamma': 1.}),
        data_hashes={'inflow.csv': 'a'*64})


def evaluation(seed, number, value, count=1, failed=False):
    windows = []
    for descriptor in descriptors():
        row = dict(descriptor, status='failed' if failed else 'ok',
                   evaluator_return=None if failed else value)
        for term in ('guaranteed_output', 'ecological_flow'):
            row.update({term+'_violation_count': None if failed else count,
                        term+'_active_reservoir_days': None if failed else 10,
                        term+'_violation_rate': None if failed else count/10.})
        windows.append(row)
    row = dict(seed=seed, evaluation_round=number, gamma=1.,
               status='failed' if failed else 'ok', window_count=3,
               successful_window_count=0 if failed else 3,
               evaluator_return=None if failed else value, windows=windows)
    for term in ('guaranteed_output', 'ecological_flow'):
        row.update({term+'_violation_count': None if failed else 3*count,
                    term+'_active_reservoir_days': None if failed else 30,
                    term+'_violation_rate': None if failed else count/10.})
    return row


def run(offset=0):
    return dict(gamma=1., seeds=[1, 2], configured_seeds=[1, 2],
                episode_horizon_days=2, parallel_envs=1, updates_per_seed=2,
                comparison_context=context(),
                histories=[[dict(training_steps=2*i, episode_reward=offset+seed+2*i,
                                 episode_energy_TWh=offset+10*seed+i)
                            for i in (1, 2)] for seed in (1, 2)],
                evaluation_histories=[[evaluation(seed, i, offset+seed+i, count=seed)
                                       for i in (0, 1, 2)] for seed in (1, 2)])


def payload():
    return dict(schema=compare.SCHEMA, runs=[dict(algorithm='happo', payload=run()),
                                            dict(algorithm='mappo', payload=run(100))])


def use_current_learning_objectives(data):
    """Declare the user-selected HAPPO-only team objective explicitly."""
    for entry in data['runs']:
        team = entry['algorithm'] == 'happo'
        common_rows = entry['payload']['comparison_context']['shared_settings']['algo'].get(
            'normalize_shared_advantages', False)
        entry['payload']['learning_objective'] = dict(
            advantage_source='shared_team' if team else 'individual_mixed',
            reward_formula=('0.5 * sum(local rewards) + 0.5 * global generation reward'
                            if team else '0.5 * own local reward + 0.5 * global generation reward'),
            critic_reward_layout='shared_team' if team else 'mixed_by_reservoir_then_legacy_team',
            actor_value_channels=[0]*5 if team else list(range(5)),
            normalization=('once_over_common_rollout_rows' if team
                           else 'per_reservoir_over_common_rollout_rows' if common_rows
                           else 'per_reservoir_over_controllable_rows'))
        entry['payload']['actor_architecture'] = dict(parameter_sharing=False,
            network_count=5, optimizer_count=5, raw_observation_dim=15,
            network_observation_dim=15, identity_features='implicit_in_separate_actor',
            critic_output_dim=1 if team else 6)
    return data


@unittest.skipIf(plt is None, 'Matplotlib is available in the plotting interpreter')
class ComparisonFigureTest(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_current_objectives_are_labeled_without_changing_physical_metric_curves(self):
        figures = compare.make_figures(use_current_learning_objectives(payload()))
        for figure in figures.values():
            self.assertIn('Different learning objectives', figure._suptitle.get_text())
        np.testing.assert_allclose(figures['training'].axes[0].lines[0].get_ydata(), [3.5, 5.5])
        np.testing.assert_allclose(figures['violations'].axes[1].lines[1].get_ydata(), [15., 15., 15.])

    def test_joint_figures_keep_seeds_within_algorithm(self):
        figures = compare.make_figures(payload())
        self.assertEqual(set(figures), {'training', 'evaluator', 'violations'})
        self.assertEqual([len(f.axes) for f in figures.values()], [2, 1, 2])
        for figure in figures.values():
            for ax in figure.axes:
                self.assertEqual(len(ax.lines), 2)
                self.assertNotEqual(ax.lines[0].get_color(), ax.lines[1].get_color())
        lines = figures['training'].axes[0].lines
        np.testing.assert_allclose(lines[0].get_ydata(), [3.5, 5.5])
        np.testing.assert_allclose(lines[1].get_ydata(), [103.5, 105.5])
        vertices = figures['training'].axes[0].collections[0].get_paths()[0].vertices
        band = vertices[np.isclose(vertices[:, 0], 2.), 1]
        self.assertAlmostEqual(band.min(), 3.5-np.sqrt(.5))
        self.assertAlmostEqual(band.max(), 3.5+np.sqrt(.5))
        np.testing.assert_allclose(figures['evaluator'].axes[0].lines[0].get_ydata(), [1.5, 2.5, 3.5])
        np.testing.assert_allclose(figures['violations'].axes[0].lines[0].get_ydata(), [15., 15., 15.])

    def test_failed_and_missing_seed_leave_algorithm_gaps(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                data = payload()
                rows = data['runs'][0]['payload']['evaluation_histories'][1]
                if missing:
                    del rows[1]
                else:
                    rows[1] = evaluation(2, 1, 3., failed=True)
                figures = compare.make_figures(data)
                self.assertTrue(np.isnan(figures['evaluator'].axes[0].lines[0].get_ydata()[1]))
                self.assertTrue(np.isfinite(figures['evaluator'].axes[0].lines[1].get_ydata()[1]))
                self.assertTrue(np.isnan(figures['violations'].axes[1].lines[0].get_ydata()[1]))

    def test_mismatch_is_rejected_before_drawing(self):
        for change in ('hash', 'physical', 'gamma', 'scenarios'):
            with self.subTest(change=change):
                data = payload()
                other = data['runs'][1]['payload']['comparison_context']
                if change == 'hash':
                    other['data_hashes']['inflow.csv'] = 'b'*64
                elif change == 'physical':
                    other['shared_settings']['physical_config']['head_model'] = 'storage_sensitivity'
                elif change == 'gamma':
                    other['shared_settings']['gamma'] = .99
                else:
                    other['shared_settings']['periodic_windows'][0]['initial_storage_m3'][0] = 2.
                with patch.object(plt, 'subplots', side_effect=AssertionError('Unexpected drawing')):
                    with self.assertRaises(ValueError):
                        compare.make_figures(data)

    def test_single_algorithm_before_first_training_batch(self):
        data = payload()
        data['runs'] = data['runs'][:1]
        data['runs'][0]['payload']['histories'] = [[], []]
        figures = compare.make_figures(data)
        self.assertEqual(len(figures['training'].axes[0].lines), 0)
        self.assertEqual(len(figures['evaluator'].axes[0].lines), 1)


def write_log(path, algorithm, inline_context, labeled=True):
    p = run(0 if algorithm == 'happo' else 100)
    shared = p['comparison_context']['shared_settings']
    metadata = dict(record_type='run_configuration', algorithm=algorithm,
        episode_horizon_days=2, parallel_environments=1, episodes_per_seed=2,
        updates_per_seed=2, training_years=shared['training_years'],
        evaluation_source_years=shared['evaluation_source_years'],
        physical_config=shared['physical_config'], data_hashes={'inflow.csv': 'a'*64},
        periodic_evaluation={'windows': descriptors()},
        config=dict({key: copy.deepcopy(shared[key]) for key in ('device', 'train', 'model', 'algo')},
                    experiment={'seeds': [1, 2], 'window_sampling': 'coverage_balanced'}))
    if inline_context:
        metadata['comparison_context'] = context()
    if not labeled:
        del metadata['algorithm']
    rows = [metadata]
    for seed, history, evaluations in zip(p['seeds'], p['histories'], p['evaluation_histories']):
        for i, row in enumerate(history, 1):
            rows.append(dict(record_type='training_update', seed=seed, update=i,
                training_steps=row['training_steps'], episode_horizon_days=2, parallel_envs=1,
                episode_reward_mean=row['episode_reward'],
                episode_generation_TWh_mean=row['episode_energy_TWh']))
        rows.extend(dict(row, record_type='periodic_evaluation') for row in evaluations)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')


class ComparisonIdentityTest(unittest.TestCase):
    def test_forecast_input_hash_identifies_current_and_legacy_actor_dimensions(self):
        for forecast, daily_limits, sharing in ((forecast, daily, sharing)
                for forecast in (False, True) for daily in (False, True)
                for sharing in (False, True)):
            with self.subTest(forecast=forecast, daily_limits=daily_limits, sharing=sharing):
                data = use_current_learning_objectives(payload())
                raw_dim = (16 if daily_limits else 12) if forecast else (15 if daily_limits else 11)
                for entry in data['runs']:
                    run = entry['payload']
                    context = run['comparison_context']
                    context['shared_settings']['physical_config']['use_daily_level_limits'] = daily_limits
                    if forecast:
                        context['data_hashes']['five_reservoir_next7d_mean_inflow_forecast_1965_2024.csv'] = 'f'*64
                    shared = sharing and entry['algorithm'] == 'mappo'
                    context['shared_settings']['algo']['share_param'] = shared
                    run['actor_architecture'].update(parameter_sharing=shared,
                        network_count=1 if shared else 5, optimizer_count=1 if shared else 5,
                        raw_observation_dim=raw_dim,
                        network_observation_dim=raw_dim+5 if shared else raw_dim,
                        identity_features='one_hot_reservoir_id' if shared else 'implicit_in_separate_actor')
                compare.validate_comparison(data)
                data['runs'][1]['payload']['actor_architecture']['network_observation_dim'] -= 1
                with self.assertRaisesRegex(ValueError, 'dimensions/counts'):
                    compare.validate_comparison(data)

    def test_team_objective_requires_scalar_happo_value_and_single_global_term(self):
        data = use_current_learning_objectives(payload())
        compare.validate_comparison(data)
        for flag in (False, True):
            changed = payload()
            for entry in changed['runs']:
                entry['payload']['comparison_context']['shared_settings']['algo'][
                    'normalize_shared_advantages'] = flag
            use_current_learning_objectives(changed)
            compare.validate_comparison(changed)
            self.assertEqual(changed['runs'][0]['payload']['learning_objective']['normalization'],
                             'once_over_common_rollout_rows')
            self.assertEqual(changed['runs'][1]['payload']['learning_objective']['normalization'],
                             'per_reservoir_over_common_rollout_rows' if flag else
                             'per_reservoir_over_controllable_rows')
        for field, value in (('actor_value_channels', list(range(5))),
                             ('reward_formula', 'sum(mixed rewards)'),
                             ('normalization', 'per_actor_controllable_rows')):
            changed = copy.deepcopy(data)
            changed['runs'][0]['payload']['learning_objective'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Learning objective'):
                compare.validate_comparison(changed)
        data['runs'][0]['payload']['actor_architecture']['critic_output_dim'] = 6
        with self.assertRaisesRegex(ValueError, 'architecture'):
            compare.validate_comparison(data)

    def test_shared_mappo_architecture_is_the_only_allowed_settings_exception(self):
        data = payload()
        happo, mappo = [entry['payload'] for entry in data['runs']]
        happo['comparison_context']['shared_settings']['algo']['share_param'] = False
        mappo['comparison_context']['shared_settings']['algo']['share_param'] = True
        mappo['actor_architecture'] = dict(parameter_sharing=True, network_count=1,
            optimizer_count=1, raw_observation_dim=15, network_observation_dim=20,
            identity_features='one_hot_reservoir_id', critic_output_dim=6)
        compare.validate_comparison(data)
        self.assertTrue(mappo['comparison_context']['shared_settings']['algo']['share_param'])
        altered = copy.deepcopy(data)
        altered['runs'][1]['payload']['comparison_context']['shared_settings']['algo']['gamma'] = .99
        with self.assertRaisesRegex(ValueError, 'settings'):
            compare.validate_comparison(altered)
        altered = copy.deepcopy(data)
        altered['runs'][1]['payload']['actor_architecture']['network_observation_dim'] = 15
        with self.assertRaisesRegex(ValueError, 'architecture'):
            compare.validate_comparison(altered)
        happo['actor_architecture'] = dict(mappo['actor_architecture'])
        with self.assertRaisesRegex(ValueError, 'Only MAPPO'):
            compare.validate_comparison(data)

    def test_context_fallback_and_inline_normalization_remove_only_share_param(self):
        settings = context()
        settings['shared_settings']['algo'].update(share_param=True, entropy_coef=.001)
        original = copy.deepcopy(settings)
        normalized = compare.context_from_configuration({'comparison_context': settings})
        self.assertNotIn('share_param', normalized['shared_settings']['algo'])
        self.assertEqual(normalized['shared_settings']['algo']['entropy_coef'], .001)
        self.assertEqual(settings, original)

    def test_temporal_weighting_provenance_is_preserved_and_must_match(self):
        data = payload()
        for entry in data['runs']:
            entry['payload']['comparison_context']['shared_settings']['actor_temporal_weighting'] = 'uniform_rollout_rows'
        compare.validate_comparison(data)
        data['runs'][0]['payload']['comparison_context']['shared_settings'].pop('actor_temporal_weighting')
        with self.assertRaisesRegex(ValueError, 'settings'):
            compare.validate_comparison(data)

        settings = context()['shared_settings']
        metadata = dict(episode_horizon_days=2, parallel_environments=1,
            episodes_per_seed=2, training_years=settings['training_years'],
            evaluation_source_years=settings['evaluation_source_years'],
            physical_config=settings['physical_config'], data_hashes={'inflow.csv': 'a'*64},
            periodic_evaluation={'windows': descriptors()},
            learning_reward_architecture={'actor_temporal_weighting': 'uniform_rollout_rows'},
            config=dict({key: copy.deepcopy(settings[key]) for key in ('device', 'train', 'model', 'algo')},
                        experiment={'seeds': [1, 2], 'window_sampling': 'coverage_balanced'}))
        extracted = compare.context_from_configuration(metadata)
        self.assertEqual(extracted['shared_settings']['actor_temporal_weighting'], 'uniform_rollout_rows')
        metadata.pop('learning_reward_architecture')
        self.assertNotIn('actor_temporal_weighting', compare.context_from_configuration(metadata)['shared_settings'])

    def test_unlabeled_legacy_log_cannot_be_claimed_as_mappo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'legacy.jsonl'
            write_log(path, 'happo', inline_context=False, labeled=False)
            result = compare.load_comparison(happo_log=path)
            self.assertEqual(result['runs'][0]['algorithm'], 'happo')
            with self.assertRaisesRegex(ValueError, 'explicit algorithm metadata'):
                compare.load_comparison(mappo_log=path)

    def test_duplicate_algorithm_and_missing_identity_rejected(self):
        data = payload()
        data['runs'][1]['algorithm'] = 'HAPPO'
        with self.assertRaisesRegex(ValueError, 'unique'):
            compare.validate_comparison(data)
        data = payload()
        del data['runs'][0]['payload']['comparison_context']
        with self.assertRaisesRegex(ValueError, 'comparison_context'):
            compare.validate_comparison(data)
        data = payload()
        data['runs'][0]['payload']['algorithm'] = 'mappo'
        with self.assertRaisesRegex(ValueError, 'Algorithm label'):
            compare.validate_comparison(data)

    def test_separate_jsonl_loading_and_metadata_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            h, m = Path(directory)/'happo.jsonl', Path(directory)/'mappo.jsonl'
            write_log(h, 'happo', inline_context=True)
            write_log(m, 'mappo', inline_context=False)
            result = compare.load_comparison(h, m)
            self.assertEqual([r['algorithm'] for r in result['runs']], ['happo', 'mappo'])
            for run_entry in result['runs']:
                self.assertEqual(run_entry['payload']['seeds'], [1, 2])
            self.assertEqual(result['runs'][0]['payload']['histories'][0][0]['episode_reward'], 3)
            self.assertEqual(result['runs'][1]['payload']['histories'][0][0]['episode_reward'], 103)
            with self.assertRaisesRegex(ValueError, 'same log'):
                compare.load_comparison(h, h)


if __name__ == '__main__':
    unittest.main()

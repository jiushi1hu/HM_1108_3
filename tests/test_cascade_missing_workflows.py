"""Exercise the real training, diagnostic, checkpoint and replay entry points."""
import copy
import csv
from contextlib import redirect_stdout
from datetime import date
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import yaml
from gym.spaces import Box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.envs.cascade_reservoir.reward import (
    EpisodeRewardMetrics, OperationReward, reward_metadata, reward_settings)
from harl.algorithms.critics.v_critic import VCritic
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.valuenorm import ValueNorm
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.runners.on_policy_ha_runner import OnPolicyHARunner


class GuaranteedOutputFrequencyRewardTest(unittest.TestCase):
    def setUp(self):
        self.data = SimpleNamespace(power_cap=np.full(5, 1000.),
            service_requirements=SimpleNamespace(
                guaranteed_output_mw=np.array([100., 100., 100., 100., 0.]),
                ecological_flow_m3s=np.full(5, 100.), metadata=lambda: {}))

    def transition(self, power, weight=None, enabled=True):
        penalties = dict(enabled=enabled, guaranteed_output_weight=1., ecological_flow_weight=1.)
        if weight is not None:
            penalties['guaranteed_output_violation_weight'] = weight
        settings = reward_settings(dict(reward_penalties=penalties))
        power = np.asarray(power, dtype=np.float64)
        result = dict(power=power.copy(), release=np.full(5, 100.),
            reward=float(power.sum()/self.data.power_cap.sum()),
            energy=24.*power, actions=np.arange(5),
            storage_start=np.arange(5, dtype=float), storage_end=np.arange(5, dtype=float)+1.)
        OperationReward(self.data, settings).apply(result)
        return result

    def test_frequency_weight_is_optional_and_validated(self):
        self.assertEqual(reward_settings({})['guaranteed_output_violation_weight'], 0.)
        default = self.transition([90., 100., 100., 100., 0.])
        zero = self.transition([90., 100., 100., 100., 0.], weight=0.)
        for field in ('local_rewards', 'mixed_rewards', 'local_reward_penalties',
                      'local_guaranteed_output_violation_penalties'):
            np.testing.assert_array_equal(default[field], zero[field])
        self.assertEqual(default['reward'], zero['reward'])
        self.assertAlmostEqual(default['reward'], .053)
        np.testing.assert_allclose(default['local_rewards'], [-.01, .1, .1, .1, 0.], atol=1e-15)
        for value in (-1., float('inf'), float('nan'), True, np.bool_(False), 'invalid'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                reward_settings(dict(reward_penalties=dict(guaranteed_output_violation_weight=value)))
        self.assertEqual(reward_settings(dict(reward_penalties=dict(
            guaranteed_output_violation_weight=1.25)))['guaranteed_output_violation_weight'], 1.25)

    def test_local_cost_preserves_physics_legacy_reward_and_service_counts(self):
        power = [99.999, 50., 100., 150., 0.]
        baseline, changed = self.transition(power, 0.), self.transition(power, 2.)
        frequency = np.array([2., 2., 0., 0., 0.])
        np.testing.assert_array_equal(changed['local_guaranteed_output_violation_penalties'], frequency)
        np.testing.assert_allclose(changed['mixed_rewards'], baseline['mixed_rewards']-.5*frequency)
        self.assertEqual(changed['global_reward'], baseline['global_reward'])
        self.assertAlmostEqual(changed['system_reward'], baseline['system_reward']-.5*frequency.sum())
        for field in ('reward', 'generation_reward', 'reward_penalty', 'reward_penalty_components'):
            self.assertEqual(changed[field], baseline[field])
        for field in ('power', 'release', 'energy', 'actions', 'storage_start', 'storage_end',
                      'reward_penalty_by_reservoir'):
            np.testing.assert_array_equal(changed[field], baseline[field])
        for group in ('service_shortfalls', 'normalized_service_shortfalls', 'service_active_masks',
                      'service_shortfall_integrals'):
            for term in baseline[group]:
                np.testing.assert_array_equal(changed[group][term], baseline[group][term])
        before, after = EpisodeRewardMetrics(.5), EpisodeRewardMetrics(.5)
        for _ in range(2):
            before.update(baseline)
            after.update(changed)
        self.assertEqual(before.shortfall_summary(), after.shortfall_summary())
        self.assertEqual(before.summary()['episode_reward'], after.summary()['episode_reward'])
        learning = after.learning_summary()
        np.testing.assert_array_equal(learning['episode_local_guaranteed_output_violation_penalties'],
                                      1.5*frequency)
        np.testing.assert_allclose(learning['episode_local_reward_penalties'],
                                  np.sum(list(learning['episode_local_penalty_components'].values()), axis=0))
        metadata = reward_metadata(self.data)
        self.assertTrue(metadata['actor_reward_definition'].endswith('_v2'))
        self.assertTrue(metadata['system_reward_definition'].endswith('_v2'))
        self.assertEqual(metadata['reward_definition'], 'generation_minus_mean_service_shortfalls_v1')

    def test_frequency_indicator_uses_existing_tolerance_and_inactive_requirements(self):
        power = [100., 100.-2.5e-9, 100.-4.e-8, 101., 0.]
        penalized = self.transition(power, 2.)
        disabled = self.transition(power, 2., enabled=False)
        np.testing.assert_array_equal(penalized['local_guaranteed_output_violation_penalties'],
                                      [0., 0., 2., 0., 0.])
        np.testing.assert_array_equal(disabled['local_guaranteed_output_violation_penalties'], np.zeros(5))
        np.testing.assert_array_equal(disabled['local_reward_penalties'], np.zeros(5))
        enabled_metrics, disabled_metrics = EpisodeRewardMetrics(1.), EpisodeRewardMetrics(1.)
        enabled_metrics.update(penalized)
        disabled_metrics.update(disabled)
        self.assertEqual(enabled_metrics.shortfall_summary(), disabled_metrics.shortfall_summary())
        guarantee = enabled_metrics.shortfall_summary()['guaranteed_output']
        self.assertEqual(guarantee['reservoir_day_count'], 1)
        self.assertEqual(guarantee['active_reservoir_days'], 4)
        unavailable = SimpleNamespace(power_cap=self.data.power_cap, service_requirements=None)
        transition = dict(power=np.ones(5), reward=.001)
        OperationReward(unavailable, reward_settings(dict(reward_penalties=dict(
            enabled=False, guaranteed_output_violation_weight=2.)))).apply(transition)
        np.testing.assert_array_equal(transition['local_guaranteed_output_violation_penalties'], np.zeros(5))

    def test_equal_total_deficit_is_distinguished_by_violation_frequency(self):
        fewer, more = EpisodeRewardMetrics(1.), EpisodeRewardMetrics(1.)
        for metric, powers in ((fewer, (90., 100.)), (more, (95., 95.))):
            for power in powers:
                metric.update(self.transition([power, 100., 100., 100., 0.], 2.))
        self.assertAlmostEqual(fewer.summary()['episode_reward'], more.summary()['episode_reward'])
        self.assertEqual(fewer.shortfall_summary()['guaranteed_output']['reservoir_day_count'], 1)
        self.assertEqual(more.shortfall_summary()['guaranteed_output']['reservoir_day_count'], 2)
        self.assertAlmostEqual(fewer.learning_summary()['episode_mixed_rewards'][0]-
                               more.learning_summary()['episode_mixed_rewards'][0], 1.)

    def test_shared_team_reward_counts_global_generation_once(self):
        row = self.transition([90., 100., 100., 100., 0.], weight=2.)
        # Own local rewards are (-2.01, .1, .1, .1, 0), with global=.078.
        # This is one common team reward, not the sum of five mixed rewards.
        self.assertAlmostEqual(row['system_reward'], -.816)
        self.assertAlmostEqual(float(row['mixed_rewards'].sum()), -.660)
        self.assertAlmostEqual(row['reward'], .053)
        selected = CascadeReservoirRunner._critic_rewards(
            SimpleNamespace(critic_reward_layout='shared_team'), [row])
        self.assertEqual(selected.shape, (1, 1))
        np.testing.assert_allclose(selected, [[-.816]], rtol=1e-7)


class SharedTeamAdvantageTest(unittest.TestCase):
    def test_team_advantage_is_normalized_once_and_shared_despite_own_masks(self):
        for old_normalization_flag in (False, True):
            with self.subTest(old_normalization_flag=old_normalization_flag):
                runner = OnPolicyHARunner()
                runner.num_agents, runner.fixed_order, runner.action_aggregation = 5, True, 'prod'
                runner.algo_args = dict(train=dict(episode_length=4, n_rollout_threads=1),
                    algo=dict(normalize_shared_advantages=old_normalization_flag,
                              mask_inactive_factors=True))
                raw = np.array([-3., -1., 1., 3.], dtype=np.float32).reshape(4, 1, 1)
                runner.critic_buffer = SimpleNamespace(returns=np.concatenate((raw, np.zeros((1, 1, 1)))),
                    value_preds=np.zeros((5, 1, 1), dtype=np.float32))
                runner.value_normalizer = None
                captured = []

                class Buffer:
                    def __init__(self, index):
                        self.obs = np.zeros((5, 1, 2), dtype=np.float32)
                        self.actions = np.zeros((4, 1, 1), dtype=np.float32)
                        self.available_actions = None
                        self.active_masks = np.ones((5, 1, 1), dtype=np.float32)
                        self.active_masks[index % 4] = 0.
                    def update_factor(self, factor):
                        self.factor = factor.copy()

                def train(buffer, advantages, state_type, **options):
                    self.assertEqual(state_type, 'EP')
                    self.assertTrue(options['advantages_normalized'])
                    captured.append(advantages.copy())
                    return {}

                def evaluate(observations, *unused):
                    return torch.zeros((len(observations), 1)), None, None

                runner.actor_buffer = [Buffer(i) for i in range(5)]
                runner.actor = [SimpleNamespace(train=train, evaluate_actions=evaluate) for _ in range(5)]
                runner.critic = SimpleNamespace(train=mock.Mock(return_value={'value_loss': 0.}))
                runner.train()
                expected = raw/(np.sqrt(5.)+1e-5)
                self.assertEqual(len(captured), 5)
                for advantages in captured:
                    np.testing.assert_allclose(advantages, expected, rtol=1e-7)
                runner.critic.train.assert_called_once_with(runner.critic_buffer, None,
                                                          update_normalizer_once=True)


class CriticValueNormUpdateTest(unittest.TestCase):
    def fixture(self, epochs, minibatches, channels=1):
        config = yaml.safe_load((ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())
        args = dict(config['model'], **config['algo'])
        args.update(hidden_sizes=[8, 8], critic_output_dim=channels,
                    critic_epoch=epochs, critic_num_mini_batch=minibatches)
        space = Box(-np.inf, np.inf, (3,), dtype=np.float32)
        torch.manual_seed(17)
        critic = VCritic(args, space)
        buffer = OnPolicyCriticBufferEP(dict(config['train'], **config['algo'],
            episode_length=4, n_rollout_threads=2, critic_output_dim=channels), space)
        scales = np.arange(1, channels+1, dtype=np.float32)
        offsets = np.array([0., 8., -4., 20., -12., 32.], dtype=np.float32)[:channels]
        buffer.share_obs[:-1] = np.arange(24, dtype=np.float32).reshape(4, 2, 3)/10.
        buffer.value_preds[:-1] = np.arange(8*channels, dtype=np.float32).reshape(4, 2, channels)/50.
        buffer.returns[:-1] = np.arange(1, 9, dtype=np.float32).reshape(4, 2, 1)*scales+offsets
        buffer.returns[-1] = 1e6*scales  # A terminal slot must never enter the statistics.
        buffer.masks[-1] = 0.
        normalizer = ValueNorm(channels, beta=.5)
        normalizer.update(np.array([[-4.], [0.]], dtype=np.float32)*scales+offsets)
        normalizer.update(np.array([[2.], [6.]], dtype=np.float32)*scales+offsets)
        return critic, buffer, normalizer

    def test_once_updates_full_rollout_and_freezes_scale_across_epochs_and_minibatches(self):
        reference_states = {}
        for channels, epochs, minibatches in (
                (channels, epochs, minibatches) for channels in (1, 6)
                for epochs, minibatches in ((1, 1), (5, 1), (1, 4), (5, 4))):
            with self.subTest(channels=channels, epochs=epochs, minibatches=minibatches):
                critic, buffer, normalizer = self.fixture(epochs, minibatches, channels)
                initial_model = copy.deepcopy(critic.critic.state_dict())
                initial_statistics = copy.deepcopy(normalizer.state_dict())
                observed_statistics = []
                normalize = normalizer.normalize

                def record_normalize(values):
                    observed_statistics.append(copy.deepcopy(normalizer.state_dict()))
                    return normalize(values)

                with mock.patch.object(normalizer, 'update', wraps=normalizer.update) as update, \
                        mock.patch.object(normalizer, 'normalize', side_effect=record_normalize), \
                        mock.patch.object(critic, 'update', wraps=critic.update) as optimize:
                    info = critic.train(buffer, normalizer, update_normalizer_once=True)
                update.assert_called_once()
                scales = np.arange(1, channels+1, dtype=np.float32)
                offsets = np.array([0., 8., -4., 20., -12., 32.], dtype=np.float32)[:channels]
                np.testing.assert_array_equal(update.call_args.args[0],
                    np.arange(1, 9, dtype=np.float32).reshape(8, 1)*scales+offsets)
                self.assertEqual(update.call_args.args[0].shape, (8, channels))
                self.assertEqual(optimize.call_count, epochs*minibatches)
                self.assertTrue(all(call.kwargs['update_normalizer'] is False
                                    for call in optimize.call_args_list))
                self.assertTrue(observed_statistics)
                # Before channel scaling/translation the seeded raw moments
                # (1.5, 12, .75) become (3, 18.75, .875) after one new batch.
                expected_mean = 3.*scales+.875*offsets
                expected_square = 18.75*scales**2+6.*scales*offsets+.875*offsets**2
                torch.testing.assert_close(normalizer.running_mean, torch.from_numpy(expected_mean), rtol=0., atol=0.)
                torch.testing.assert_close(normalizer.running_mean_sq, torch.from_numpy(expected_square), rtol=0., atol=0.)
                torch.testing.assert_close(normalizer.debiasing_term, torch.tensor(.875), rtol=0., atol=0.)
                mean, variance = normalizer.running_mean_var()
                torch.testing.assert_close(mean, torch.from_numpy((24./7.)*scales+offsets), rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(variance, torch.from_numpy((474./49.)*scales**2), rtol=1e-5, atol=1e-5)
                for state in observed_statistics:
                    for name, value in normalizer.state_dict().items():
                        self.assertTrue(torch.equal(state[name], value), name)
                if channels not in reference_states:
                    reference_states[channels] = copy.deepcopy(normalizer.state_dict())
                else:
                    for name, value in normalizer.state_dict().items():
                        self.assertTrue(torch.equal(value, reference_states[channels][name]), name)
                self.assertFalse(torch.equal(initial_statistics['debiasing_term'],
                                             normalizer.debiasing_term))
                self.assertTrue(any(not torch.equal(value, critic.critic.state_dict()[name])
                                    for name, value in initial_model.items()))
                self.assertTrue(all(np.isfinite(float(value)) for value in info.values()))

    def test_low_level_default_keeps_per_minibatch_compatibility(self):
        # Both cascade runners opt into once-per-rollout updates. The low-level
        # VCritic API keeps its explicit default for other existing callers.
        critic, buffer, normalizer = self.fixture(epochs=5, minibatches=2, channels=6)
        with mock.patch.object(normalizer, 'update', wraps=normalizer.update) as update:
            info = critic.train(buffer, normalizer)
        self.assertEqual(update.call_count, 10)
        batches = [np.asarray(call.args[0]) for call in update.call_args_list]
        self.assertTrue(all(batch.shape == (4, 6) for batch in batches))
        np.testing.assert_array_equal(np.sort(np.concatenate(batches), axis=0),
            np.sort(np.tile(buffer.returns[:-1].reshape(8, 6), (5, 1)), axis=0))
        self.assertEqual(float(normalizer.debiasing_term), 1.-.25*.5**10)
        self.assertTrue(all(np.isfinite(float(value)) for value in info.values()))

    def test_once_flag_without_normalizer_preserves_updates_and_rng(self):
        plain, buffer, _ = self.fixture(epochs=2, minibatches=2)
        once = copy.deepcopy(plain)
        torch.manual_seed(72)
        plain_info = plain.train(buffer, None)
        state = torch.get_rng_state().clone()
        torch.manual_seed(72)
        once_info = once.train(buffer, None, update_normalizer_once=True)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        torch.testing.assert_close(plain.critic.state_dict(), once.critic.state_dict(), rtol=0., atol=0.)
        torch.testing.assert_close(plain.critic_optimizer.state_dict(), once.critic_optimizer.state_dict(),
                                   rtol=0., atol=0.)
        torch.testing.assert_close(plain_info, once_info, rtol=0., atol=0.)


class InflowForecastObservationTest(unittest.TestCase):
    """Date-keyed supplied forecasts change actor information only."""

    @classmethod
    def setUpClass(cls):
        cls.physical = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        cls.args = yaml.safe_load((ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical, cls.args['experiment'])
        cls.columns = (
            'WDD_external_inflow_forecast_next7d_mean_m3s',
            'BHT_interval_inflow_forecast_next7d_mean_m3s',
            'XLD_interval_inflow_forecast_next7d_mean_m3s',
            'XJB_interval_inflow_forecast_next7d_mean_m3s',
            'THR_interval_inflow_forecast_next7d_mean_m3s',
        )
        with (ROOT/'data/cascade_reservoir_new/five_reservoir_next7d_mean_inflow_forecast_1965_2024.csv').open(
                encoding='utf-8-sig', newline='') as stream:
            cls.by_date = {row['date']: [float(row[column]) for column in cls.columns]
                           for row in csv.DictReader(stream)}

    @staticmethod
    def write_forecast(path, columns, rows):
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)

    def test_forecast_loader_maps_column_names_and_dates_instead_of_positions(self):
        from harl.envs.cascade_reservoir.data_loader import (
            FORECAST_COLUMNS, load_inflow_forecast)
        self.assertEqual(FORECAST_COLUMNS, self.columns)
        dates = (date(1968, 12, 31), date(1969, 1, 1))
        # Deliberately place the date in the middle and reverse station order
        # and row chronology. Values identify each station independently.
        headers = (self.columns[4], self.columns[2], 'date', self.columns[0],
                   self.columns[3], self.columns[1])
        records = [dict(date='1968-12-31', **dict(zip(self.columns, [1., 2., 3., 4., 5.]))),
                   dict(date='1969-01-01', **dict(zip(self.columns, [11., 12., 13., 14., 15.])))]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'forecast.csv'
            self.write_forecast(path, headers, [[row[column] for column in headers]
                                                for row in reversed(records)])
            forecast = load_inflow_forecast(path, dates)
        np.testing.assert_array_equal(forecast, [[1., 2., 3., 4., 5.], [11., 12., 13., 14., 15.]])
        self.assertEqual(forecast.dtype, np.float64)
        self.assertFalse(forecast.flags.writeable)

    def test_forecast_loader_rejects_missing_ambiguous_or_invalid_inputs(self):
        from harl.envs.cascade_reservoir.data_loader import load_inflow_forecast
        dates = (date(1965, 1, 1), date(1965, 1, 2))
        headers = ('date',)+self.columns
        rows = [['1965-01-01', 1., 2., 3., 4., 5.],
                ['1965-01-02', 11., 12., 13., 14., 15.]]
        cases = [
            ('missing column', headers[:-1], [row[:-1] for row in rows]),
            ('duplicate column', headers+(self.columns[0],), [row+[1.] for row in rows]),
            ('unknown station column', headers[:-1]+('TGR_interval_inflow_forecast_next7d_mean_m3s',), rows),
            ('duplicate date', headers, rows+[rows[0]]),
            ('missing date', headers, rows[:1]),
            ('outside date', headers, rows[:1]+[['1965-01-03']+rows[1][1:]]),
            ('invalid date', headers, [['1965-02-30']+rows[0][1:]]+rows[1:]),
        ]
        for value in ('', 'nan', 'inf', '-1', 'not-a-number'):
            changed = copy.deepcopy(rows)
            changed[0][3] = value
            cases.append(('invalid value '+value, headers, changed))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'forecast.csv'
            for name, columns, values in cases:
                with self.subTest(case=name):
                    self.write_forecast(path, columns, values)
                    with self.assertRaises(ValueError):
                        load_inflow_forecast(path, dates)

    def test_actor_forecast_is_last_and_uses_training_actual_inflow_statistics(self):
        from harl.envs.cascade_reservoir.episodes import episode_environment
        train = self.data.external[np.isin(self.data.years, self.data.training_years)]
        np.testing.assert_array_equal(self.data.flow_mean, train.mean(axis=0))
        np.testing.assert_array_equal(self.data.flow_std, np.maximum(train.std(axis=0), 1.))
        self.assertFalse(self.data.inflow_forecast.flags.writeable)
        for daily_limits in (True, False):
            config = dict(self.physical, use_daily_level_limits=daily_limits)
            env = episode_environment(self.data, config, [1965], days=4)
            state, observed = env.state(), env.actor_observations()
            storage, external, release = state[:5], state[5:10], state[10:15]
            upper = (env._next_state_storage_bounds(env.index)[1]-self.data.vmin)/(self.data.vmax-self.data.vmin)
            old_features = []
            for i in range(5):
                upstream = (storage[i-1], external[i-1], release[i-1]) if i else (0., 0., 0.)
                downstream = (storage[i+1], upper[i+1]-storage[i+1]) if i < 4 else (0., 1.)
                row = [storage[i], external[i], release[i], *upstream, *downstream]
                if daily_limits:
                    row.extend(state[start+i] for start in (18, 23, 28, 33))
                row.extend(state[15:18])
                old_features.append(row)
            np.testing.assert_array_equal(observed[:, :-1], np.asarray(old_features, dtype=np.float32))
            expected = (np.asarray(self.by_date['1965-01-01'])-self.data.flow_mean)/self.data.flow_std
            np.testing.assert_array_equal(observed[:, -1], expected.astype(np.float32))
            self.assertEqual(observed.shape, (5, 16 if daily_limits else 12))
            self.assertEqual(state.shape, (38 if daily_limits else 18,))
            self.assertEqual(env.observation_features[-1], 'next7d_mean_inflow_forecast_standardized')

    def test_episode_forecasts_keep_source_dates_at_leaps_joins_and_dataset_end(self):
        from harl.envs.cascade_reservoir.episodes import EpisodeData
        for years, days in (([1968, 1969], 367), (list(self.data.evaluation_years), 1825), ([2024], 366)):
            view = EpisodeData(self.data, years, days)
            expected = np.asarray([self.by_date[day.isoformat()] for day in view.dates])
            np.testing.assert_array_equal(view.inflow_forecast, expected)
            self.assertFalse(view.inflow_forecast.flags.writeable)
            if years[0] == 1968:
                self.assertEqual(view.dates[59], date(1968, 2, 29))
                self.assertEqual(view.dates[366], date(1969, 1, 1))
            elif years[0] == 1998:
                self.assertEqual(view.dates[729], date(1999, 12, 31))
                self.assertEqual(view.dates[730], date(1970, 1, 1))
            # Changing realized forcing must not recompute a forecast from its
            # next seven rows, including at artificial held-out year joins.
            changed = copy.copy(self.data)
            changed.external = np.zeros_like(self.data.external)
            np.testing.assert_array_equal(EpisodeData(changed, years, days).inflow_forecast, expected)

    def test_forecast_only_changes_neither_critic_state_nor_fixed_command_physics(self):
        from harl.envs.cascade_reservoir.episodes import episode_environment
        changed = copy.copy(self.data)
        changed.inflow_forecast = self.data.inflow_forecast+10000.*np.arange(1., 6.)
        original = episode_environment(self.data, self.physical, [1965], days=4)
        altered = episode_environment(changed, self.physical, [1965], days=4)
        for commands in (np.zeros(5), np.full(5, 99), np.array([3, 11, 45, 70, 90]), np.full(5, 48)):
            np.testing.assert_array_equal(original.state(), altered.state())
            np.testing.assert_array_equal(original.actor_observations()[:, :-1],
                                          altered.actor_observations()[:, :-1])
            self.assertTrue(np.all(original.actor_observations()[:, -1] != altered.actor_observations()[:, -1]))
            original.step(commands)
            altered.step(commands)
            left, right = original.last_transition, altered.last_transition
            for field in ('external', 'inflow', 'storage_start', 'storage_end', 'release', 'turbine',
                          'spill', 'power', 'energy', 'low', 'high', 'controls', 'local_rewards',
                          'mixed_rewards', 'local_reward_penalties', 'storage_constraints'):
                np.testing.assert_array_equal(left[field], right[field])
            for field in ('reward', 'generation_reward', 'reward_penalty', 'global_reward', 'system_reward'):
                self.assertEqual(left[field], right[field])
        np.testing.assert_array_equal(original.actor_observations(), np.zeros((5, 16), dtype=np.float32))
        np.testing.assert_array_equal(altered.actor_observations(), original.actor_observations())
        np.testing.assert_array_equal(original.state(), np.zeros(38, dtype=np.float32))

    def test_happo_actor_network_and_rollout_buffers_accept_sixteen_features(self):
        args = copy.deepcopy(self.args)
        args['train'].update(episode_length=4, n_rollout_threads=1)
        args['model'].update(hidden_sizes=[8, 8], action_bins=7)
        args['experiment'].update(episodes=1, seeds=[1])
        runner = CascadeReservoirRunner(self.data, self.physical, args, 1,
            periodic_windows=select_periodic_windows(self.data, episode_days=4))
        self.addCleanup(runner.close)
        obs = runner.envs[0].actor_observations()
        for i, actor in enumerate(runner.actor):
            first_linear = next(layer for layer in actor.actor.modules() if isinstance(layer, torch.nn.Linear))
            self.assertEqual(first_linear.in_features, 16)
            self.assertEqual(runner.actor_buffer[i].obs.shape, (5, 1, 16))
            action = actor.act(obs[i:i+1], deterministic=True)
            self.assertEqual(tuple(action.shape), (1, 1))
        first_critic = next(layer for layer in runner.critic.critic.modules() if isinstance(layer, torch.nn.Linear))
        self.assertEqual(first_critic.in_features, 38)
        self.assertEqual(runner.critic_buffer.share_obs.shape, (5, 1, 38))


class MissingWorkflowIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = yaml.safe_load((ROOT / 'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())
        cls.physical = yaml.safe_load((ROOT / 'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        cls.args['train'].update(episode_length=4, n_rollout_threads=2)
        cls.args['model'].update(hidden_sizes=[8, 8], action_bins=7)
        cls.args['algo'].update(ppo_epoch=2, critic_epoch=2)
        cls.args['experiment'].update(episodes=2, seeds=[1], diagnostics=False)
        cls.data = CascadeData(ROOT / 'data/cascade_reservoir_new', cls.physical,
                               cls.args['experiment'])
        cls.windows = select_periodic_windows(cls.data, episode_days=4)

    def assert_state_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(expected, actual)
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assert_state_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    @staticmethod
    def training_state(runner):
        return copy.deepcopy(dict(
            actors=[actor.actor.state_dict() for actor in runner.actor],
            actor_optimizers=[actor.actor_optimizer.state_dict() for actor in runner.actor],
            critic=runner.critic.critic.state_dict(),
            critic_optimizer=runner.critic.critic_optimizer.state_dict(),
            normalization=runner.value_normalizer.state_dict(),
            random_state=torch.get_rng_state(),
        ))

    def test_real_update_tracing_checkpoint_and_replay(self):
        from harl.envs.cascade_reservoir.checkpoint import load_checkpoint, ReplayTraceWriter
        from harl.envs.cascade_reservoir.trajectory_diagnostics import (
            TrajectoryDiagnosticsWriter, TRAJECTORY_DIAGNOSTIC_FIELDS)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            directory = Path(directory)
            baseline = CascadeReservoirRunner(self.data, self.physical, self.args, 1,
                                               periodic_windows=self.windows)
            self.addCleanup(baseline.close)
            initial = self.training_state(baseline)
            with mock.patch.object(baseline.value_normalizer, 'update',
                    wraps=baseline.value_normalizer.update) as update_normalizer, \
                    mock.patch.object(baseline.critic, 'train', wraps=baseline.critic.train) as train_critic:
                baseline.run(2)
            train_critic.assert_called_once_with(baseline.critic_buffer, baseline.value_normalizer,
                                                update_normalizer_once=True)
            update_normalizer.assert_called_once()
            np.testing.assert_array_equal(update_normalizer.call_args.args[0],
                baseline.critic_buffer.returns[:-1].reshape(-1, 1))
            expected = self.training_state(baseline)
            self.assertEqual(baseline.critic_reward_layout, 'shared_team')
            self.assertEqual(baseline.critic_output_dim, 1)
            self.assertEqual(baseline.actor_value_channels, (0,)*5)
            self.assertEqual(baseline.critic_buffer.rewards.shape, (4, 2, 1))
            self.assertEqual(tuple(baseline.value_normalizer.running_mean.shape), (1,))
            for lane, episode in enumerate(baseline.episode_history):
                selected_return = sum(self.args['algo']['gamma']**t*
                                      baseline.critic_buffer.rewards[t, lane, 0] for t in range(4))
                np.testing.assert_allclose(selected_return,
                    episode['learning_rewards']['episode_system_reward'], rtol=1e-6)
            self.assertTrue(any(not torch.equal(initial['critic'][name], expected['critic'][name])
                                for name in initial['critic']))
            self.assertTrue(any(not torch.equal(initial['actors'][i][name], expected['actors'][i][name])
                                for i in range(5) for name in initial['actors'][i]))

            trajectory = directory / 'trajectory.csv'
            with TrajectoryDiagnosticsWriter(trajectory) as writer:
                traced = CascadeReservoirRunner(self.data, self.physical, self.args, 1,
                    periodic_windows=self.windows, trajectory_log=writer,
                    checkpoint_dir=directory / 'checkpoints')
                self.addCleanup(traced.close)
                traced.run(2)
            self.assert_state_equal(expected, self.training_state(traced))
            with trajectory.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4 * 2 * 5)
            self.assertEqual(set(rows[0]), set(TRAJECTORY_DIAGNOSTIC_FIELDS))
            self.assertEqual(len(traced.checkpoint_history), 1)

            checkpoint = directory / 'checkpoints/update_000001.pt'
            self.assertTrue(checkpoint.is_file())
            random_before_load = torch.get_rng_state().clone()
            policy = load_checkpoint(checkpoint)
            self.assertTrue(torch.equal(random_before_load, torch.get_rng_state()))
            self.assert_state_equal(traced.critic.critic.state_dict(), policy.critic.state_dict())
            self.assert_state_equal(traced.value_normalizer.state_dict(),
                                    policy.value_normalizer.state_dict())
            output = io.StringIO()
            replay = policy.replay([1965], writer=ReplayTraceWriter(output))
            reference = evaluate_policy(traced.actor, self.data, self.physical,
                seed=1, episode=2, training_steps=8, gamma=self.args['algo']['gamma'],
                source_years=[1965], episode_days=4, diagnostics=True)
            for field in ('energy_MWh', 'episode_reward', 'episode_mixed_rewards',
                          'episode_system_reward', 'spill_volume_m3'):
                np.testing.assert_allclose(replay[field], reference[field], rtol=0., atol=1e-10)
            replay_rows = list(csv.DictReader(io.StringIO(output.getvalue())))
            self.assertEqual(len(replay_rows), 4 * 5)
            self.assertTrue(all(row['value_basis'] == 'shared_team_system_reward' for row in replay_rows))
            self.assertTrue(all(row['legacy_team_value'] == '' for row in replay_rows))

    def test_cli_help_imports_complete_workflows(self):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        for script, option in (('train_cascade.py', '--trajectory-diagnostics'),
                               ('replay_cascade_checkpoint.py', '--override-action')):
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, '-B', str(ROOT / 'examples' / script), '--help'],
                    cwd=str(ROOT), env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(option, result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)

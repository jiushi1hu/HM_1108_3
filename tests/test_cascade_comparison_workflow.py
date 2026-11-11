"""Matched algorithm dispatch, atomic output checks and a short real workflow."""
import copy
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import train_cascade as cli
from examples import train_cascade_compare as compare
from examples import train_cascade_parallel as parallel
from harl.algorithms.actors.happo import HAPPO
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.valuenorm import ValueNorm
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.reward import (
    ACTOR_REWARD_DEFINITION, SYSTEM_REWARD_DEFINITION, OperationReward, reward_settings)
from harl.runners.cascade_mappo_runner import CascadeMAPPORunner
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner


def comparison_options(**changes):
    values = dict(algorithms=['happo', 'mappo'], episode_log=None, no_plot=True,
        check_data_only=False, xld_diagnostics=False, trajectory_diagnostics=False,
        save_checkpoints=False, skip_final_evaluation=True, plot_python=sys.executable)
    values.update(changes)
    return SimpleNamespace(**values)


def assert_learning_provenance(case, record, algorithm, common_rows=True):
    team = algorithm == 'happo'
    case.assertEqual(record['algorithm'], algorithm)
    case.assertEqual(record['critic_reward_layout'],
        'shared_team' if team else 'mixed_by_reservoir_then_legacy_team')
    case.assertEqual(record['actor_value_channels'], [0]*5 if team else list(range(5)))
    case.assertEqual(record['advantage_source'], 'shared_team' if team else 'individual_mixed')
    case.assertEqual(record['actor_reward_sources'], ['system_reward']*5 if team else
                     ['mixed_rewards[{}]'.format(i) for i in range(5)])
    case.assertEqual(record['actor_reward_definition'],
                     SYSTEM_REWARD_DEFINITION if team else ACTOR_REWARD_DEFINITION)
    case.assertEqual(record['mixed_reward_definition'], ACTOR_REWARD_DEFINITION)
    case.assertEqual(record['advantage_normalization'], 'once_over_common_rollout_rows' if team else
                     'per_reservoir_over_common_rollout_rows' if common_rows else
                     'per_reservoir_over_controllable_rows')


class ComparisonDispatchTest(unittest.TestCase):
    def test_plot_only_cli_routes_three_figures_without_run_files(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            directory = Path(directory)
            stem = directory/'cycle_seeds123'
            stack.enter_context(mock.patch.object(cli, 'CascadeData', return_value=mock.sentinel.data))
            helper = stack.enter_context(mock.patch.object(compare, 'run_parallel_experiment',
                side_effect=lambda *args, **kwargs: {'algorithm': kwargs['algorithm_name']}))
            stack.enter_context(mock.patch.object(compare, 'validate_comparison'))
            plots = stack.enter_context(mock.patch.object(compare.subprocess, 'run'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            compare.main(['--seeds', '1', '2', '3', '--plot-output', str(stem)])
            self.assertEqual(helper.call_count, 2)
            for call in helper.call_args_list:
                self.assertEqual(call.args[2]['experiment']['seeds'], [1, 2, 3])
                options = call.args[3]
                self.assertIsNone(options.episode_log)
                self.assertIsNone(options.plot_output)
                self.assertTrue(options.no_plot)
                self.assertFalse(options.save_checkpoints)
                self.assertFalse(options.trajectory_diagnostics)
            self.assertEqual(plots.call_count, 2)  # Backend check, then joint plotting.
            command = plots.call_args.args[0]
            self.assertEqual(Path(command[1]).name, 'plot_cascade_compare.py')
            self.assertEqual(command[2:], [
                '--output', str(directory/'cycle_seeds123_training.png'),
                '--evaluator-output', str(directory/'cycle_seeds123_evaluator_return.png'),
                '--violations-output', str(directory/'cycle_seeds123_violation_rates.png')])
            self.assertEqual(list(directory.iterdir()), [])

    def test_plot_output_collision_is_rejected_before_data_or_training(self):
        for entry_point in ('happo', 'comparison'):
            with self.subTest(entry_point=entry_point), tempfile.TemporaryDirectory() as directory:
                stem = Path(directory)/'cycle'
                collision = stem.with_name('cycle_violation_rates.png')
                collision.write_text('existing figure', encoding='utf-8')
                with mock.patch.object(cli, 'CascadeData') as data_factory, \
                        mock.patch.object(compare, 'run_parallel_experiment') as helper, \
                        redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        if entry_point == 'happo':
                            cli.main(['--plot-output', str(stem)], experiment_runner=helper)
                        else:
                            compare.main(['--plot-output', str(stem)])
                    self.assertEqual(raised.exception.code, 2)
                    data_factory.assert_not_called()
                    helper.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [collision])
                self.assertEqual(collision.read_text(encoding='utf-8'), 'existing figure')

    def test_plot_output_requires_plotting(self):
        with mock.patch.object(cli, 'CascadeData') as data_factory, \
                redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as raised:
                cli.main(['--plot-output', 'cycle', '--no-plot'])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn('--plot-output requires plotting', errors.getvalue())
            data_factory.assert_not_called()

    def test_cli_defaults_to_both_algorithms_and_supports_single_selection(self):
        selections = [([], ['happo', 'mappo']),
                      (['--algorithms', 'mappo'], ['mappo']),
                      (['--algorithms', 'happo'], ['happo'])]
        for arguments, expected in selections:
            with self.subTest(algorithms=expected), ExitStack() as stack:
                data_factory = stack.enter_context(mock.patch.object(
                    cli, 'CascadeData', return_value=mock.sentinel.data))
                helper = stack.enter_context(mock.patch.object(compare, 'run_parallel_experiment',
                    side_effect=lambda *args, **kwargs: {'algorithm': kwargs['algorithm_name']}))
                # This test isolates CLI dispatch; complete histories and real
                # comparison validation are exercised by the integration test.
                stack.enter_context(mock.patch.object(compare, 'validate_comparison',
                                                       side_effect=lambda value: value))
                plots = stack.enter_context(mock.patch.object(compare.subprocess, 'run'))
                stack.enter_context(redirect_stdout(io.StringIO()))
                result = compare.main(arguments+['--no-plot', '--skip-final-evaluation'])
                self.assertEqual([row['algorithm'] for row in result['runs']], expected)
                self.assertEqual(helper.call_count, len(expected))
                data_factory.assert_called_once()
                plots.assert_not_called()
                for call, algorithm in zip(helper.call_args_list, expected):
                    self.assertIs(call.args[0], mock.sentinel.data)
                    self.assertIs(call.kwargs['runner_cls'], compare.RUNNERS[algorithm])
                    self.assertEqual(call.kwargs['algorithm_name'], algorithm)
                    self.assertEqual(call.args[2]['algo']['gamma'], .995)
                    self.assertIsNone(call.args[3].episode_log)
                    self.assertTrue(call.args[3].no_plot)

    def test_cli_gamma_override_reaches_both_algorithm_hooks(self):
        with mock.patch.object(cli, 'CascadeData', return_value=mock.sentinel.data), \
                mock.patch.object(compare, 'run_parallel_experiment',
                    side_effect=lambda *args, **kwargs: {'algorithm': kwargs['algorithm_name']}) as helper, \
                mock.patch.object(compare, 'validate_comparison', side_effect=lambda value: value), \
                redirect_stdout(io.StringIO()):
            compare.main(['--gamma', '.75', '--no-plot', '--skip-final-evaluation'])
        self.assertEqual([call.kwargs['algorithm_name'] for call in helper.call_args_list],
                         ['happo', 'mappo'])
        for call in helper.call_args_list:
            self.assertEqual(call.args[2]['algo']['gamma'], .75)

    def test_cli_enables_xld_diagnostics_for_both_algorithms_without_run_files(self):
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            directory = Path(directory)
            stack.enter_context(mock.patch.object(
                cli, 'CascadeData', return_value=mock.sentinel.data))
            helper = stack.enter_context(mock.patch.object(compare, 'run_parallel_experiment',
                side_effect=lambda *args, **kwargs: {'algorithm': kwargs['algorithm_name']}))
            stack.enter_context(mock.patch.object(compare, 'validate_comparison',
                                                   side_effect=lambda value: value))
            plots = stack.enter_context(mock.patch.object(compare.subprocess, 'run'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            os.chdir(directory)
            try:
                result = compare.main(['--xld-diagnostics', '--no-plot',
                                       '--skip-final-evaluation'])
            finally:
                os.chdir(original_directory)
            self.assertEqual([row['algorithm'] for row in result['runs']], ['happo', 'mappo'])
            self.assertEqual(helper.call_count, 2)
            for call in helper.call_args_list:
                self.assertTrue(call.args[2]['experiment']['xld_diagnostics'])
                self.assertTrue(call.args[3].xld_diagnostics)
                self.assertIsNone(call.args[3].episode_log)
                self.assertFalse(call.args[3].save_checkpoints)
                self.assertFalse(call.args[3].trajectory_diagnostics)
            self.assertEqual(list(directory.iterdir()), [])
            plots.assert_not_called()

    def test_each_algorithm_receives_fresh_settings_and_separate_output_paths(self):
        config, physical = cli.load_configuration()
        expected_config, expected_physical = copy.deepcopy(config), copy.deepcopy(physical)
        expected_mappo_config = copy.deepcopy(expected_config)
        expected_mappo_config['algo']['share_param'] = True
        entries = []

        def helper(data, received_physical, received_config, options, **kwargs):
            entries.append(dict(data=data, physical=received_physical, config=received_config,
                options=options, before_physical=copy.deepcopy(received_physical),
                before_config=copy.deepcopy(received_config), **kwargs))
            # A mutation by one algorithm must not contaminate its competitor
            # or the caller's configuration, including nested lists/mappings.
            received_config['model']['hidden_sizes'][0] = -1
            received_physical['reward_penalties']['guaranteed_output_violation_weight'] = 999.
            return dict(algorithm=kwargs['algorithm_name'])

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)/'paired.jsonl'
            options = comparison_options(episode_log=base)
            with mock.patch.object(compare, 'run_parallel_experiment', side_effect=helper), \
                    mock.patch.object(compare, 'validate_comparison'), \
                    redirect_stdout(io.StringIO()):
                compare.run_comparison(mock.sentinel.data, physical, config, options)
            self.assertEqual([entry['before_config'] for entry in entries],
                             [expected_config, expected_mappo_config])
            self.assertEqual([entry['before_physical'] for entry in entries],
                             [expected_physical, expected_physical])
            self.assertIsNot(entries[0]['config'], entries[1]['config'])
            self.assertIsNot(entries[0]['physical'], entries[1]['physical'])
            self.assertIsNot(entries[0]['options'], entries[1]['options'])
            for entry, algorithm, runner_class in zip(entries, ('happo', 'mappo'),
                    (CascadeReservoirRunner, CascadeMAPPORunner)):
                self.assertIs(entry['runner_cls'], runner_class)
                self.assertEqual(entry['options'].episode_log,
                                 base.with_name('paired_'+algorithm+'.jsonl'))
                self.assertIsNot(entry['options'], options)
            self.assertEqual(options.episode_log, base)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(config, expected_config)
        self.assertEqual(physical, expected_physical)

    def test_second_algorithm_collision_is_rejected_before_first_training(self):
        config, physical = cli.load_configuration()
        config['experiment']['seeds'] = [1, 2, 3]
        for kind in ('summary', 'steps', 'checkpoints'):
            with self.subTest(output=kind), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                base = directory/'paired.jsonl'
                second = directory/'paired_mappo.jsonl'
                if kind == 'summary':
                    collision = second
                elif kind == 'steps':
                    collision = directory/'paired_mappo_seed2_steps.csv'
                else:
                    collision = directory/'paired_mappo_seed2_checkpoints'
                if kind == 'checkpoints':
                    collision.mkdir()
                else:
                    collision.write_text('pre-existing output', encoding='utf-8')
                options = comparison_options(episode_log=base, save_checkpoints=True)
                with mock.patch.object(compare, 'run_parallel_experiment') as helper:
                    with self.assertRaisesRegex(ValueError, 'already exists'):
                        compare.run_comparison(mock.sentinel.data, physical, config, options)
                    helper.assert_not_called()
                self.assertEqual(list(directory.iterdir()), [collision])
                if kind != 'checkpoints':
                    self.assertEqual(collision.read_text(encoding='utf-8'), 'pre-existing output')

    def test_removed_kl_flags_are_rejected_before_configuration_data_or_training(self):
        removed_arguments = (['--target-kl', '.01'], ['--no-kl-limit'],
                             ['--kl-rollback'], ['--no-kl-rollback'],
                             ['--joint-target-kl', '.03'], ['--no-joint-kl-limit'])
        for entry_point in ('happo', 'comparison'):
            for arguments in removed_arguments:
                with self.subTest(entry_point=entry_point, flag=arguments[0]), \
                        mock.patch.object(cli, 'load_configuration') as configuration, \
                        mock.patch.object(cli, 'CascadeData') as data_factory, \
                        mock.patch.object(compare, 'run_parallel_experiment') as helper, \
                        redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
                    with self.assertRaises(SystemExit) as raised:
                        if entry_point == 'happo':
                            cli.main(['--no-plot']+arguments, experiment_runner=helper)
                        else:
                            compare.main(['--no-plot']+arguments)
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn('unrecognized arguments:', errors.getvalue())
                    self.assertIn(arguments[0], errors.getvalue())
                    configuration.assert_not_called()
                    data_factory.assert_not_called()
                    helper.assert_not_called()

    def test_original_entry_point_and_happo_training_dispatch_remain_defaults(self):
        self.assertIs(CascadeReservoirRunner.actor_class, HAPPO)
        self.assertEqual(CascadeReservoirRunner.algorithm_name, 'happo')
        self.assertIs(CascadeReservoirRunner.train, OnPolicyHARunner.train)
        self.assertIs(inspect.signature(cli.main).parameters['experiment_runner'].default,
                      parallel.run_parallel_experiment)
        defaults = inspect.signature(parallel.run_parallel_experiment).parameters
        self.assertIs(defaults['runner_cls'].default, CascadeReservoirRunner)
        self.assertIsNone(defaults['algorithm_name'].default)
        with mock.patch.object(cli, 'CascadeData', return_value=mock.sentinel.data), \
                redirect_stdout(io.StringIO()):
            helper = mock.Mock(return_value=mock.sentinel.result)
            result = cli.main(['--no-plot'], experiment_runner=helper)
        self.assertIs(result, mock.sentinel.result)
        helper.assert_called_once()
        self.assertFalse(hasattr(helper.call_args.args[3], 'algorithms'))
        self.assertEqual(helper.call_args.args[2]['algo']['gamma'], .995)


class ShortComparisonIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.physical = cli.load_configuration()
        cls.config['train'].update(episode_length=4, n_rollout_threads=2)
        cls.config['model'].update(hidden_sizes=[8, 8], action_bins=7)
        cls.config['algo'].update(ppo_epoch=1, critic_epoch=1)
        cls.config['experiment'].update(episodes=2, seeds=[1, 2, 3], diagnostics=False,
            trajectory_diagnostics=False, save_checkpoints=False, checkpoint_interval=0,
            xld_diagnostics=False)
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical,
                               cls.config['experiment'])
        cls.windows = select_periodic_windows(cls.data, episode_days=4)

    def test_happo_runs_one_shared_team_value_channel_without_changing_mappo_layout(self):
        config = copy.deepcopy(self.config)
        config['algo']['normalize_shared_advantages'] = False
        records = io.StringIO()
        runner = CascadeReservoirRunner(self.data, self.physical, config, 1,
                                        periodic_windows=self.windows, episode_log=records)
        self.addCleanup(runner.close)
        self.assertEqual(runner.critic_reward_layout, 'shared_team')
        self.assertEqual(runner.actor_value_channels, (0, 0, 0, 0, 0))
        self.assertEqual(runner.critic_output_dim, 1)
        self.assertEqual(runner.value_normalizer.input_shape, 1)
        initial = copy.deepcopy(runner.critic.critic.state_dict())
        with redirect_stdout(io.StringIO()):
            runner.run(2)
        self.assertEqual(runner.update_count, 1)
        self.assertEqual(runner.critic_buffer.rewards.shape, (4, 2, 1))
        self.assertEqual(runner.critic_buffer.returns.shape, (5, 2, 1))
        weights = config['algo']['gamma']**np.arange(4, dtype=np.float64)
        for lane, episode in enumerate(runner.episode_history):
            observed = float(weights @ runner.critic_buffer.rewards[:, lane, 0])
            # Collection stores float32 targets; episode accounting retains
            # the physical transition's float64 reward components.
            np.testing.assert_allclose(observed, episode['learning_rewards']['episode_system_reward'],
                                       rtol=5e-7, atol=2e-6)
            assert_learning_provenance(self, episode, 'happo')
            assert_learning_provenance(self, episode['learning_rewards'], 'happo')
        for evaluation in runner.periodic_evaluation_history:
            assert_learning_provenance(self, evaluation, 'happo')
            for window in evaluation['windows']:
                assert_learning_provenance(self, window, 'happo')
        self.assertTrue(any(not torch.equal(value, runner.critic.critic.state_dict()[name])
                            for name, value in initial.items()))
        final = dict(source_years=list(self.data.evaluation_years), episode_reward=1.25,
                     actor_reward_definition=ACTOR_REWARD_DEFINITION)
        with mock.patch('harl.runners.cascade_reservoir_runner.evaluate_policy', return_value=final):
            result = runner.final_evaluation()[0]
        assert_learning_provenance(self, result, 'happo')
        self.assertEqual(result['episode_reward'], 1.25)
        failure = InfeasibleCascade('Synthetic metadata-only final-evaluation failure')
        failure.evaluation_progress = dict(completed_steps=2, partial_metrics=dict(evaluator_return=1.25))
        with mock.patch('harl.runners.cascade_reservoir_runner.evaluate_policy', side_effect=failure):
            with self.assertRaises(InfeasibleCascade):
                runner.final_evaluation()
        failed = [json.loads(line) for line in records.getvalue().splitlines()
                  if json.loads(line).get('status') == 'failed'][-1]
        assert_learning_provenance(self, failed, 'happo')
        self.assertEqual(failed['partial_metrics']['evaluator_return'], 1.25)

    def test_shared_policy_preserves_raw_environment_and_buffer_layouts(self):
        config = copy.deepcopy(self.config)
        config['algo'].update(critic_epoch=5, critic_num_mini_batch=2)
        config_before = copy.deepcopy(config)
        runner = CascadeMAPPORunner(self.data, self.physical, config, 1,
                                    periodic_windows=self.windows)
        self.addCleanup(runner.close)
        core = runner.actor[0].core
        self.assertEqual(len(runner.actor), 5)
        self.assertTrue(all(view.core is core for view in runner.actor))
        self.assertTrue(all(view.actor is core.actor for view in runner.actor))
        self.assertTrue(all(view.actor_optimizer is core.actor_optimizer for view in runner.actor))
        self.assertEqual(core.obs_space.shape, (21,))
        self.assertTrue(all(buffer.obs.shape[-1] == 16 for buffer in runner.actor_buffer))
        self.assertEqual(runner.envs[0].observation_space[0].shape, (16,))
        self.assertEqual(runner.critic_buffer.share_obs.shape[-1], 38)
        self.assertEqual(runner.critic_output_dim, 6)
        self.assertEqual(runner.critic_reward_layout, 'mixed_by_reservoir_then_legacy_team')
        self.assertEqual(runner.actor_value_channels, (0, 1, 2, 3, 4))
        self.assertTrue(runner.algo_args['algo']['share_param'])
        self.assertEqual(runner.requested_args, config_before)
        old_statistics = copy.deepcopy(runner.value_normalizer.state_dict())
        normalization_states, actor_completed = [], []
        train_actor = core.share_param_train
        normalize = runner.value_normalizer.normalize
        update = runner.value_normalizer.update

        def actor_update(buffers, advantages, *args, **kwargs):
            for name, value in old_statistics.items():
                self.assertTrue(torch.equal(value, runner.value_normalizer.state_dict()[name]), name)
            raw = runner.critic_buffer.returns[:-1]-runner.value_normalizer.denormalize(
                runner.critic_buffer.value_preds[:-1])
            expected = (raw-raw.mean(axis=(0, 1), keepdims=True))/(
                raw.std(axis=(0, 1), keepdims=True)+1e-5)
            for i, own in enumerate(advantages):
                np.testing.assert_array_equal(own, expected[..., i:i+1])
            result = train_actor(buffers, advantages, *args, **kwargs)
            actor_completed.append(True)
            return result

        def update_statistics(values):
            self.assertEqual(actor_completed, [True])
            update(values)

        def normalize_targets(values):
            normalization_states.append(copy.deepcopy(runner.value_normalizer.state_dict()))
            return normalize(values)

        with mock.patch.object(core, 'share_param_train', side_effect=actor_update), \
                mock.patch.object(runner.value_normalizer, 'update', side_effect=update_statistics) as update_norm, \
                mock.patch.object(runner.value_normalizer, 'normalize', side_effect=normalize_targets), \
                mock.patch.object(runner.critic, 'update', wraps=runner.critic.update) as optimize, \
                mock.patch.object(runner.critic, 'train', wraps=runner.critic.train) as train_critic, \
                redirect_stdout(io.StringIO()):
            runner.run(2)
        train_critic.assert_called_once_with(runner.critic_buffer, runner.value_normalizer,
                                            update_normalizer_once=True)
        update_norm.assert_called_once()
        self.assertEqual(update_norm.call_args.args[0].shape, (8, 6))
        np.testing.assert_array_equal(update_norm.call_args.args[0],
                                      runner.critic_buffer.returns[:-1].reshape(8, 6))
        self.assertEqual(optimize.call_count, 10)
        self.assertTrue(all(call.kwargs['update_normalizer'] is False for call in optimize.call_args_list))
        self.assertTrue(normalization_states)
        for state in normalization_states:
            for name, value in runner.value_normalizer.state_dict().items():
                self.assertTrue(torch.equal(state[name], value), name)
        self.assertEqual(config, config_before)

    def test_three_paired_seeds_return_complete_histories_without_run_files(self):
        config = copy.deepcopy(self.config)
        config['experiment']['xld_diagnostics'] = True
        config_before, physical_before = copy.deepcopy(config), copy.deepcopy(self.physical)
        options = comparison_options(xld_diagnostics=True)
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            directory = Path(directory)
            stack.enter_context(mock.patch.object(parallel, 'select_periodic_windows',
                                                  return_value=self.windows))
            stack.enter_context(mock.patch.object(parallel, 'PERIODIC_EVALUATION_DAYS', 4))
            plots = stack.enter_context(mock.patch.object(compare.subprocess, 'run',
                side_effect=AssertionError('No-plot workflow launched a plotting process')))
            stack.enter_context(redirect_stdout(io.StringIO()))
            os.chdir(directory)
            try:
                result = compare.run_comparison(self.data, self.physical, config, options)
            finally:
                os.chdir(original_directory)
            self.assertEqual(list(directory.iterdir()), [])
            self.assertFalse((directory/'results').exists())
            plots.assert_not_called()
        self.assertEqual(config, config_before)
        self.assertEqual(self.physical, physical_before)
        self.assertEqual([row['algorithm'] for row in result['runs']], ['happo', 'mappo'])
        happo, mappo = [row['payload'] for row in result['runs']]
        self.assertEqual(happo['comparison_context'], mappo['comparison_context'])
        context = happo['comparison_context']
        self.assertTrue(context['data_hashes'])
        settings = context['shared_settings']
        self.assertEqual(settings['seeds'], [1, 2, 3])
        self.assertEqual(settings['episode_horizon_days'], 4)
        self.assertEqual(settings['parallel_environments'], 2)
        self.assertEqual(settings['episodes_per_seed'], 2)
        self.assertEqual(settings['gamma'], .995)
        self.assertEqual(settings['actor_temporal_weighting'], 'uniform_rollout_rows')
        self.assertEqual(settings['periodic_windows'], self.windows)
        for key in ('device', 'train', 'model', 'algo'):
            expected = copy.deepcopy(config[key])
            if key == 'algo':
                expected.pop('share_param', None)
            self.assertEqual(settings[key], expected)
        for algorithm, payload in (('happo', happo), ('mappo', mappo)):
            with self.subTest(algorithm=algorithm):
                self.assertEqual(payload['algorithm'], algorithm)
                self.assertEqual(payload['seeds'], [1, 2, 3])
                self.assertEqual(len(payload['histories']), 3)
                self.assertEqual(len(payload['evaluation_histories']), 3)
                for seed, history, evaluations in zip(payload['seeds'], payload['histories'],
                                                     payload['evaluation_histories']):
                    self.assertEqual(len(history), 1)
                    self.assertEqual(history[0]['seed'], seed)
                    self.assertEqual(history[0]['training_steps'], 8)
                    self.assertEqual(history[0]['completed_episodes'], 2)
                    self.assertNotIn('xld_dispatch_diagnostics', history[0])
                    self.assertEqual([row['evaluation_round'] for row in evaluations], [0, 1])
                    self.assertTrue(all(row['gamma'] == .995 for row in evaluations))
                    self.assertTrue(all(row['status'] == 'ok' for row in evaluations))
                    self.assertTrue(all(row['seed'] == seed for row in evaluations))
                    for evaluation in evaluations:
                        assert_learning_provenance(self, evaluation, algorithm)
                        for window in evaluation['windows']:
                            assert_learning_provenance(self, window, algorithm)
                    self.assertTrue(all('xld_dispatch_diagnostics' not in window
                                        for window in evaluations[0]['windows']))
                    for window in evaluations[-1]['windows']:
                        diagnostic = window['xld_dispatch_diagnostics']
                        guaranteed = window['service_shortfalls']['guaranteed_output']
                        self.assertEqual(diagnostic['seed'], seed)
                        self.assertEqual(diagnostic['evaluation_round'], 1)
                        self.assertEqual(diagnostic['completed_steps'], 4)
                        self.assertEqual(diagnostic['violation_count'],
                                         guaranteed['by_reservoir_days'][2])
                        self.assertEqual(diagnostic['active_days'],
                                         guaranteed['by_reservoir_active_days'][2])
                        self.assertEqual(sum(diagnostic['classification_counts'].values()),
                                         diagnostic['completed_steps'])
                        self.assertEqual(sum(value for name, value in
                            diagnostic['classification_counts'].items()
                            if name not in ('no_violation', 'requirement_inactive')),
                            diagnostic['violation_count'])
                final_windows = [window for evaluations in payload['evaluation_histories']
                                 for window in evaluations[-1]['windows']]
                final_diagnostics = [window['xld_dispatch_diagnostics']
                                     for window in final_windows]
                summary = payload['xld_dispatch_summary']
                self.assertIsNone(summary['held_out'])
                fixed = summary['fixed_training_windows']
                self.assertEqual(fixed['status'], 'ok')
                self.assertEqual(fixed['phase'], 'training_evaluation')
                self.assertEqual(fixed['component_count'], 9)
                self.assertEqual(fixed['completed_steps'], 36)
                self.assertEqual(fixed['seeds'], [1, 2, 3])
                self.assertEqual(fixed['evaluation_rounds'], [1])
                for name in ('active_days', 'violation_count'):
                    self.assertEqual(fixed[name], sum(row[name] for row in final_diagnostics))
                self.assertEqual(fixed['violation_count'], sum(
                    window['service_shortfalls']['guaranteed_output']['by_reservoir_days'][2]
                    for window in final_windows))
                expected_classes = {name: sum(row['classification_counts'][name]
                                             for row in final_diagnostics)
                                    for name in final_diagnostics[0]['classification_counts']}
                self.assertEqual(fixed['classification_counts'], expected_classes)
                expected_groups = dict(
                    actor_missed_feasible_discrete=expected_classes['same_day_policy_alternative'],
                    discrete_grid_gap=expected_classes['discrete_grid_gap'],
                    same_state_continuous_unattainable=sum(expected_classes[name] for name in (
                        'same_state_capacity_limit', 'same_state_water_limit',
                        'same_state_dispatch_limit', 'same_state_head_limit')))
                self.assertEqual(fixed['group_counts'], expected_groups)
                self.assertEqual(sum(fixed['group_counts'].values()), fixed['violation_count'])
                self.assertEqual(fixed['violation_rate'],
                                 fixed['violation_count']/fixed['active_days'])
                for name, count in expected_groups.items():
                    self.assertEqual(fixed['group_fractions_among_violations'][name],
                                     count/fixed['violation_count']
                                     if fixed['violation_count'] else None)
        for left, right in zip(happo['histories'], mappo['histories']):
            self.assertEqual(left[0]['window_start_year'], right[0]['window_start_year'])
        # Returned results must be ordinary detached data, not live workers,
        # tensors, optimizer objects, or mutable views into runner histories.
        json.dumps(result, allow_nan=False)


class HAPPOSharedTeamAdvantageTest(unittest.TestCase):
    def test_runtime_provenance_qualifies_failed_episodes_without_changing_accounting(self):
        for runner_class, algorithm in ((CascadeReservoirRunner, 'happo'),
                                        (CascadeMAPPORunner, 'mappo')):
            with self.subTest(algorithm=algorithm):
                runner = object.__new__(runner_class)
                runner.algo_args = dict(algo=dict(normalize_shared_advantages=False, use_gae=True))
                record = dict(record_type='training_episode', completed=False, total_reward=1.25,
                    learning_rewards=dict(actor_reward_definition=ACTOR_REWARD_DEFINITION,
                        episode_system_reward=2.5, episode_mixed_rewards=[1., 2., 3., 4., 5.]))
                runner._annotate_learning_rewards(record)
                assert_learning_provenance(self, record, algorithm, common_rows=False)
                assert_learning_provenance(self, record['learning_rewards'], algorithm, common_rows=False)
                self.assertFalse(record['completed'])
                self.assertEqual(record['total_reward'], 1.25)
                self.assertEqual(record['learning_rewards']['episode_system_reward'], 2.5)
                self.assertEqual(record['learning_rewards']['episode_mixed_rewards'], [1., 2., 3., 4., 5.])
                failed_round = dict(status='failed', windows=[dict(status='failed',
                    partial_metrics=dict(evaluator_return=1.25))])
                runner._annotate_learning_rewards(failed_round)
                assert_learning_provenance(self, failed_round['windows'][0], algorithm, common_rows=False)
                self.assertEqual(failed_round['windows'][0]['partial_metrics']['evaluator_return'], 1.25)

    def test_happo_rejects_the_old_individual_value_layout(self):
        runner = OnPolicyHARunner()
        runner.num_agents = 5
        runner.algo_args = dict(train=dict(episode_length=2, n_rollout_threads=1))
        runner.value_normalizer = None
        runner.critic_buffer = SimpleNamespace(returns=np.zeros((3, 1, 6)),
                                               value_preds=np.zeros((3, 1, 6)))
        with self.assertRaisesRegex(ValueError, 'one shared-team'):
            runner.train()

    def test_team_reward_uses_global_generation_once_and_mappo_keeps_own_rewards(self):
        data = SimpleNamespace(power_cap=np.array([100., 200., 300., 400., 500.]),
            service_requirements=SimpleNamespace(guaranteed_output_mw=np.full(5, 10.),
                                                 ecological_flow_m3s=np.full(5, 5.)))
        settings = reward_settings(dict(reward_penalties=dict(enabled=True,
            guaranteed_output_weight=1., ecological_flow_weight=1.,
            guaranteed_output_violation_weight=1.)))
        power = np.array([5., 80., 90., 100., 150.])
        transition = dict(power=power.copy(), release=np.array([2., 6., 8., 9., 10.]),
                          reward=float(power.sum()/data.power_cap.sum()))
        OperationReward(data, settings).apply(transition)
        expected_team = .5*transition['local_rewards'].sum()+.5*transition['global_reward']
        self.assertEqual(transition['system_reward'], expected_team)
        self.assertNotEqual(expected_team, float(transition['mixed_rewards'].sum()))
        self.assertNotEqual(expected_team, transition['reward'])
        saved = copy.deepcopy(transition)
        happo = object.__new__(CascadeReservoirRunner)
        mappo = object.__new__(CascadeMAPPORunner)
        team = happo._critic_rewards([transition])
        own = mappo._critic_rewards([transition])
        np.testing.assert_array_equal(team, np.asarray([[expected_team]], dtype=np.float32))
        np.testing.assert_array_equal(own,
            np.asarray([np.r_[transition['mixed_rewards'], transition['reward']]], dtype=np.float32))
        for key in ('power', 'release', 'local_rewards', 'mixed_rewards'):
            np.testing.assert_array_equal(transition[key], saved[key])
        self.assertEqual(transition['reward'], saved['reward'])

    def test_five_actors_receive_one_normalized_team_advantage_with_different_masks(self):
        for normalization_flag in (False, True):
            with self.subTest(normalize_shared_advantages=normalization_flag):
                runner = OnPolicyHARunner()
                runner.num_agents, runner.fixed_order, runner.action_aggregation = 5, True, 'prod'
                runner.algo_args = dict(train=dict(episode_length=4, n_rollout_threads=2),
                    algo=dict(normalize_shared_advantages=normalization_flag,
                              mask_inactive_factors=True))
                values = np.arange(10, dtype=np.float32).reshape(5, 2, 1)/7.
                returns = np.square(np.arange(10, dtype=np.float32).reshape(5, 2, 1))
                runner.critic_buffer = SimpleNamespace(returns=returns, value_preds=values)
                runner.value_normalizer = ValueNorm(1, beta=.5)
                runner.value_normalizer.update(np.array([[-3.], [1.], [4.]], dtype=np.float32))
                old_statistics = copy.deepcopy(runner.value_normalizer.state_dict())
                raw = returns[:-1]-runner.value_normalizer.denormalize(values[:-1])
                expected = (raw-raw.mean(axis=(0, 1), keepdims=True))/(
                    raw.std(axis=(0, 1), keepdims=True)+1e-5)
                calls, captured, factors = [], [], []

                class Buffer:
                    def __init__(self, index):
                        self.obs = np.zeros((5, 2, 2), dtype=np.float32)
                        self.actions = np.zeros((4, 2, 1), dtype=np.float32)
                        self.available_actions = np.ones((5, 2, 3), dtype=np.float32)
                        self.active_masks = np.ones((5, 2, 1), dtype=np.float32)
                        self.active_masks[index % 4, index % 2] = 0.
                    def update_factor(self, factor):
                        self.factor = factor.copy()

                class Actor:
                    def __init__(self, index):
                        self.index, self.updated = index, False
                    def evaluate_actions(self, obs, action, available_actions, active_masks):
                        logp = torch.full((len(obs), 1), .05*(self.index+1) if self.updated else 0.,
                                          dtype=torch.float32)
                        return logp, torch.tensor(0.), None
                    def train(self, buffer, advantages, state_type, diagnostics=False,
                              advantages_normalized=False):
                        if state_type != 'EP' or not advantages_normalized:
                            raise AssertionError('Shared team advantage must bypass per-actor normalization')
                        if any(not torch.equal(value, runner.value_normalizer.state_dict()[name])
                               for name, value in old_statistics.items()):
                            raise AssertionError('ValueNorm changed before all actor advantages were consumed')
                        calls.append(self.index)
                        captured.append(advantages.copy())
                        factors.append(buffer.factor.copy())
                        # A caller must supply an independent copy to every actor.
                        advantages.fill(-1234.)
                        self.updated = True
                        return dict(policy_loss=0.)

                runner.actor_buffer = [Buffer(i) for i in range(5)]
                runner.actor = [Actor(i) for i in range(5)]

                def critic_train(buffer, normalizer, *, update_normalizer_once=False):
                    self.assertTrue(update_normalizer_once)
                    self.assertEqual(calls, [0, 1, 2, 3, 4])
                    calls.append('critic')
                    normalizer.update(buffer.returns[:-1].reshape(-1, 1))
                    return dict(value_loss=0.)

                runner.critic = SimpleNamespace(train=mock.Mock(side_effect=critic_train))
                infos, _ = runner.train()
                self.assertEqual(calls, [0, 1, 2, 3, 4, 'critic'])
                for advantage in captured:
                    np.testing.assert_array_equal(advantage, expected)
                product = np.ones((4, 2, 1), dtype=np.float32)
                for i, buffer in enumerate(runner.actor_buffer):
                    np.testing.assert_array_equal(factors[i], product)
                    ratio = float(torch.exp(torch.tensor(.05*(i+1), dtype=torch.float32)))
                    product *= np.where(buffer.active_masks[:-1] > 0., ratio, 1.).astype(np.float32)
                    self.assertAlmostEqual(infos[i]['factor_max'], float(product.max()))
                runner.critic.train.assert_called_once_with(runner.critic_buffer, runner.value_normalizer,
                                                          update_normalizer_once=True)
                self.assertFalse(torch.equal(old_statistics['debiasing_term'],
                                             runner.value_normalizer.debiasing_term))

    def test_actor_explicitly_bypasses_per_mask_normalization(self):
        from gym.spaces import Box, Discrete
        config, _ = cli.load_configuration()
        args = dict(config['model'], **config['algo'])
        args.update(hidden_sizes=[8, 8], normalize_shared_advantages=False,
                    ppo_epoch=1, actor_num_mini_batch=1)
        space = Box(-np.inf, np.inf, (2,), dtype=np.float32)
        actor = HAPPO(args, space, Discrete(3))
        buffer = OnPolicyActorBuffer(dict(episode_length=4, n_rollout_threads=1), space, Discrete(3))
        buffer.obs[:-1, 0, 0] = np.arange(4)
        buffer.active_masks[0] = 0.
        buffer.update_factor(np.ones((4, 1, 1), dtype=np.float32))
        raw = np.array([-2., -.5, .3, 3.], dtype=np.float32).reshape(4, 1, 1)
        normalized = (raw-raw.mean())/(raw.std()+1e-5)
        captured = {}
        def update(sample):
            for obs, advantage in zip(sample[0], sample[5]):
                captured[int(obs[0])] = advantage.copy()
            zero = torch.tensor(0.)
            return zero, zero, zero, torch.ones_like(torch.from_numpy(sample[5]))
        with mock.patch.object(actor, 'update', side_effect=update):
            actor.train(buffer, normalized.copy(), 'EP', advantages_normalized=True)
        np.testing.assert_array_equal(np.stack([captured[i] for i in range(4)]),
                                      normalized[:, 0])



class SharedMAPPOTrainingDispatchTest(unittest.TestCase):
    def test_five_own_channels_use_one_joint_actor_update_and_one_critic_update(self):
        runner = OnPolicyMARunner()
        runner.num_agents = 5
        runner.algo_args = dict(algo=dict(share_param=True, normalize_shared_advantages=True))
        runner.diagnostics_enabled = True
        raw = np.arange(48, dtype=np.float32).reshape(4, 2, 6)
        runner.critic_buffer = SimpleNamespace(returns=raw**2, value_preds=raw/7.)
        runner.value_normalizer = ValueNorm(6, beta=.5)
        runner.value_normalizer.update(np.array([[-3.], [1.], [4.]], dtype=np.float32)*
            np.arange(1, 7, dtype=np.float32)+np.array([0., 8., -4., 20., -12., 32.], dtype=np.float32))
        old_statistics = copy.deepcopy(runner.value_normalizer.state_dict())

        class Buffer:
            def __init__(self):
                self.active_masks = np.ones((4, 2, 1), dtype=np.float32)
                self.factor = None

            def update_factor(self, factor):
                self.factor = factor.copy()

        runner.actor_buffer = [Buffer() for _ in range(5)]
        expected = raw[:-1]**2-runner.value_normalizer.denormalize(raw[:-1]/7.)
        expected = (expected-expected.mean(axis=(0, 1), keepdims=True))/(
            expected.std(axis=(0, 1), keepdims=True)+1e-5)
        calls = []

        def joint_train(buffers, advantages, num_agents, state_type, diagnostics=False):
            calls.append('actor')
            self.assertIs(buffers, runner.actor_buffer)
            self.assertEqual(num_agents, 5)
            self.assertEqual(state_type, 'EP')
            self.assertTrue(diagnostics)
            self.assertEqual(len(advantages), 5)
            for name, value in old_statistics.items():
                self.assertTrue(torch.equal(value, runner.value_normalizer.state_dict()[name]), name)
            for agent_id, (buffer, own_advantage) in enumerate(zip(buffers, advantages)):
                np.testing.assert_array_equal(buffer.factor, np.ones((3, 2, 1)))
                np.testing.assert_array_equal(own_advantage, expected[..., agent_id:agent_id+1])
            return dict(policy_loss=2., optimizer_steps=3.)

        core = SimpleNamespace(share_param_train=mock.Mock(side_effect=joint_train))
        runner.actor = [SimpleNamespace(core=core,
            train=mock.Mock(side_effect=AssertionError('Shared actor was updated per reservoir')))
            for _ in range(5)]
        def critic_train(buffer, normalizer, *, update_normalizer_once=False):
            self.assertTrue(update_normalizer_once)
            self.assertEqual(calls, ['actor'])
            calls.append('critic')
            normalizer.update(buffer.returns[:-1].reshape(-1, 6))
            return {'value_loss': 0.}

        runner.critic = SimpleNamespace(train=mock.Mock(side_effect=critic_train))
        infos, critic_info = runner.train()
        core.share_param_train.assert_called_once()
        runner.critic.train.assert_called_once_with(runner.critic_buffer, runner.value_normalizer,
                                                  update_normalizer_once=True)
        self.assertEqual(calls, ['actor', 'critic'])
        self.assertEqual(critic_info['value_loss'], 0.)
        self.assertEqual(len({id(info) for info in infos}), 5)
        for actor, info in zip(runner.actor, infos):
            actor.train.assert_not_called()
            self.assertEqual(info['optimizer_steps'], 3.)
            self.assertEqual(info['shared_actor'], 1.)
            self.assertEqual(info['shared_actor_group_size'], 5.)
            self.assertEqual(info['shared_actor_info_repeated'], 1.)
            self.assertEqual((info['factor_min'], info['factor_max']), (1., 1.))


if __name__ == '__main__':
    unittest.main(verbosity=2)

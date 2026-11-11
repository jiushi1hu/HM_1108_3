"""XLD dispatch attribution against actual physics, without training or artifacts."""
import copy
from contextlib import ExitStack, redirect_stdout
import csv
import io
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.data_loader import CascadeData, IDS
from harl.envs.cascade_reservoir.episodes import episode_environment
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics, InfeasibleCascade
from harl.envs.cascade_reservoir.reward import OperationReward, reward_settings
from harl.envs.cascade_reservoir.xld_diagnostics import (
    XLDDispatchDiagnostics, XLDDispatchDiagnosticError,
    XLDDispatchWriter, XLD_DIAGNOSTIC_FIELDS)


def synthetic_dispatch(*, head_model='fixed', guarantee=10., bins=200,
                       available_water=30., external=0., outlet=10000.,
                       turbine=10000., action=0, downstream_empty=False):
    """A five-reservoir actual-physics case with an independently known XLD curve.

    In the variable-head case, XLD storage spans 0..100 flow-days and levels
    0..100 metres. With initial storage 100 flow-days and inflow 100 m3/s,
    q lies in [100,200], H=160-q/2, and power has its interior peak at q=160.
    """
    seconds = 86400.
    maximum = np.full(5, seconds*10000.)
    storage = np.full(5, seconds*5000.)
    if head_model == 'storage_sensitivity':
        maximum[2] = seconds*100.
        storage[2] = maximum[2]
    else:
        storage[2] = seconds*available_water
    if downstream_empty:
        storage[3] = 0.
    anchors = {name: np.asarray([[0., 0.], [100., maximum[i]]])
               for i, name in enumerate(IDS)}
    data = SimpleNamespace(vmin=np.zeros(5), vmax=maximum,
        release_cap=np.full(5, 10000.), turbine_cap=np.full(5, 10000.),
        power_cap=np.full(5, 1e6), head=np.full(5, 100.), anchors=anchors,
        head_sensitivity=None)
    data.release_cap[2], data.turbine_cap[2] = outlet, turbine
    def levels(values):
        return np.asarray([np.interp(values[i], anchors[name][:, 1], anchors[name][:, 0])
                           for i, name in enumerate(IDS)])
    data.levels = levels
    if head_model == 'storage_sensitivity':
        slopes, references = np.zeros(5), np.full(5, 100.)
        slopes[2], references[2] = 1., 10.
        data.head_sensitivity = dict(efficiency=np.full(5, .9),
            reference_net_head_m=references,
            head_slope_m_per_m=slopes,
            assumed_reference_forebay_level_m=np.zeros(5))
    requirements = SimpleNamespace(guaranteed_output_mw=np.zeros(5),
                                   ecological_flow_m3s=np.zeros(5))
    requirements.guaranteed_output_mw[2] = guarantee
    data.service_requirements = requirements
    config = dict(head_model=head_model, release_mapping='quadratic',
        timestep_seconds=seconds, efficiency=.9, water_density_kg_m3=1000.,
        gravity_m_s2=9.81, reward_penalties=dict(enabled=True,
        guaranteed_output_weight=1., ecological_flow_weight=1.))
    physics = ReservoirPhysics(data, config)
    operation = OperationReward(data, reward_settings(config))
    inflow = np.zeros(5)
    inflow[2] = external
    indices = np.zeros(5, dtype=int)
    indices[2] = action
    if downstream_empty:
        indices[3] = bins-1
    transition = physics.transition(storage, inflow, indices/(bins-1),
                                    data.vmin, data.vmax)
    transition.update(date='2000-01-01', sampled_actions=indices.copy(),
        normalized_actions=transition['actions'].copy(), action_bins=bins,
        timestep_seconds=seconds, levels_start=data.levels(storage),
        levels_end=data.levels(transition['storage_end']))
    transition['actions'] = np.where(transition['controls'] > 0, indices, 0).astype(int)
    operation.apply(transition)
    env = SimpleNamespace(data=data, physics=physics, operation_reward=operation,
        action_bins=bins, storage=transition['storage_end'].copy(), index=1,
        last_transition=transition)
    return env, transition


def collector(env, writer=None):
    return XLDDispatchDiagnostics(env, writer, seed=1, evaluation_round=50,
        window_id='test_window', phase='training_evaluation')


def brute_grid(env, transition):
    """Independent exhaustive execution; no local diagnostic power kernel."""
    actions = transition['sampled_actions'].astype(float)/(env.action_bins-1)
    results = []
    for action in range(env.action_bins):
        candidate = actions.copy()
        candidate[2] = action/(env.action_bins-1)
        results.append(env.physics.transition(transition['storage_start'],
            transition['external'], candidate, transition['storage_lower_m3'],
            transition['storage_upper_m3'], storage_constraints=transition.get('storage_constraints')))
    return results


class XLDDispatchDiagnosticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.physical = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        cls.experiment = yaml.safe_load((ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())['experiment']
        cls.historical = {}
        for head_model in ('fixed', 'storage_sensitivity'):
            config = dict(cls.physical, head_model=head_model,
                          guaranteed_output_scale=.4, ecological_flow_scale=.4,
                          terminal_storage_equal_initial=False)
            cls.historical[head_model] = (CascadeData(ROOT/'data/cascade_reservoir_new',
                                                     config, cls.experiment), config)

    def assert_nested_equal(self, first, second):
        if isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_historical_best_discrete_action_matches_full_cascade_enumeration(self):
        for mode, (data, physical) in self.historical.items():
            with self.subTest(head_model=mode):
                env = episode_environment(data, physical, [1965], days=9, action_bins=200)
                audit = collector(env)
                # First move away from the high initial storage that forces
                # substantial January releases even for action zero.
                env.step(np.full(5, 199))
                audit.update(env.last_transition, 1)
                for day in range(8):
                    chosen = np.zeros(5, dtype=int)
                    env.step(chosen)
                    transition = env.last_transition
                    expected = brute_grid(env, transition)
                    best = max(float(row['power'][2]) for row in expected)
                    record = audit.update(transition, day+2)
                    self.assertTrue(record['violating'])
                    self.assertAlmostEqual(record['best_grid_power_mw'], best,
                                           delta=max(1e-7, abs(best)*1e-10))
                    index = record['best_grid_action_index']
                    self.assertGreaterEqual(index, 0)
                    self.assertLess(index, 200)
                    self.assertAlmostEqual(expected[index]['power'][2], best,
                                           delta=max(1e-7, abs(best)*1e-10))
                summary = audit.summary()
                self.assertEqual(summary['completed_steps'], 9)
                self.assertEqual(summary['violation_count'], 8)
                self.assertEqual(sum(summary['classification_counts'].values()), 9)
                json.dumps(summary, allow_nan=False)

    def test_water_dispatch_and_hardware_limits_are_distinguished(self):
        cases = [
            (dict(guarantee=10.), 'same_day_policy_alternative'),
            (dict(guarantee=100.), 'same_state_water_limit'),
            (dict(guarantee=100., available_water=1000., outlet=30.), 'same_state_dispatch_limit'),
            (dict(guarantee=10., turbine=1.), 'same_state_capacity_limit'),
        ]
        for settings, expected in cases:
            with self.subTest(settings=settings):
                env, row = synthetic_dispatch(**settings)
                record = collector(env).update(row, 1)
                self.assertTrue(record['violating'])
                self.assertEqual(record['classification'], expected)
                self.assertEqual(record['water_limit_proven'], expected == 'same_state_water_limit')

    def test_continuous_interior_peak_and_discrete_grid_gap(self):
        env, row = synthetic_dispatch(head_model='storage_sensitivity', external=100.,
                                      guarantee=110., bins=2)
        record = collector(env).update(row, 1)
        self.assertEqual(record['classification'], 'discrete_grid_gap')
        expected = .008829*160.*80.
        self.assertAlmostEqual(record['continuous_max_power_mw'], expected, places=8)
        # The maximum release, q=200, produces less than the interior optimum.
        brute = brute_grid(env, row)
        self.assertLess(max(float(r['power'][2]) for r in brute), 110.)
        self.assertGreater(record['continuous_max_power_mw'], 110.)
        env_fine, row_fine = synthetic_dispatch(head_model='storage_sensitivity',
            external=100., guarantee=110., bins=200)
        self.assertEqual(collector(env_fine).update(row_fine, 1)['classification'],
                         'same_day_policy_alternative')

    def test_head_limit_requires_actual_continuous_maximum_to_fail(self):
        env, row = synthetic_dispatch(head_model='storage_sensitivity', external=100.,
                                      guarantee=114., bins=200)
        record = collector(env).update(row, 1)
        self.assertEqual(record['classification'], 'same_state_head_limit')
        self.assertFalse(record['water_limit_proven'])
        self.assertFalse(record['dispatch_limit_proven'])
        self.assertLess(record['continuous_max_power_mw'], 114.)

    def test_head_contribution_can_coexist_with_a_policy_alternative(self):
        env, row = synthetic_dispatch(head_model='storage_sensitivity', external=100.,
                                      guarantee=110., bins=200, action=199)
        # Re-express the exact same H=10+forebay curve around a 100 m
        # reference head. At executed q=200, H=60; q around 160 still meets G.
        env.data.head_sensitivity['reference_net_head_m'][2] = 100.
        env.data.head_sensitivity['assumed_reference_forebay_level_m'][2] = 90.
        record = collector(env).update(row, 1)
        self.assertTrue(record['head_contribution_at_executed_release'])
        self.assertEqual(record['classification'], 'same_day_policy_alternative')

    def test_forced_xld_has_no_invented_discrete_choice(self):
        env, row = synthetic_dispatch(available_water=0.)
        self.assertEqual(row['controls'][2], 0.)
        record = collector(env).update(row, 1)
        self.assertTrue(record['forced_action'])
        self.assertEqual(record['classification'], 'same_state_water_limit')
        self.assertEqual(record['best_grid_power_mw'], 0.)
        self.assertEqual(record['continuous_max_power_mw'], 0.)

    def test_original_downstream_policy_action_is_preserved_when_unforced(self):
        env, row = synthetic_dispatch(downstream_empty=True)
        self.assertEqual(row['controls'][3], 0.)
        self.assertEqual(row['normalized_actions'][3], .5)
        self.assertEqual(row['sampled_actions'][3], 199)
        with mock.patch.object(env.physics, 'transition', wraps=env.physics.transition) as replay:
            record = collector(env).update(row, 1)
        self.assertEqual(record['classification'], 'same_day_policy_alternative')
        self.assertGreater(replay.call_count, 0)
        expected = row['sampled_actions']/(env.action_bins-1)
        for call in replay.call_args_list:
            actual = call.args[2]
            for i in (0, 1, 3, 4):
                self.assertEqual(actual[i], expected[i])

    def test_diagnostics_do_not_mutate_transition_environment_or_rng(self):
        env, row = synthetic_dispatch()
        original = copy.deepcopy(row)
        storage, index = env.storage.copy(), env.index
        numpy_before, python_before = np.random.get_state(), random.getstate()
        audit = collector(env)
        audit.update(row, 1)
        audit.summary()
        self.assert_nested_equal(row, original)
        np.testing.assert_array_equal(env.storage, storage)
        self.assertEqual(env.index, index)
        self.assertIs(env.last_transition, row)
        self.assert_nested_equal(np.random.get_state(), numpy_before)
        self.assertEqual(random.getstate(), python_before)

    def test_nonviolations_and_inactive_requirements_do_not_invent_causes(self):
        for guarantee, action, expected in ((0., 0, 'requirement_inactive'),
                                             (10., 199, 'no_violation')):
            with self.subTest(guarantee=guarantee):
                env, row = synthetic_dispatch(guarantee=guarantee, action=action)
                audit = collector(env)
                record = audit.update(row, 1)
                self.assertFalse(record['violating'])
                self.assertEqual(record['classification'], expected)
                self.assertIsNone(record['best_grid_action_index'])
                self.assertEqual(audit.summary()['violation_count'], 0)
                if guarantee == 0.:
                    self.assertEqual(audit.summary()['active_days'], 0)
                    self.assertIsNone(audit.summary()['violation_rate'])

    def test_csv_has_one_compact_row_per_day_and_rejects_nonfinite_values(self):
        stream = io.StringIO()
        writer = XLDDispatchWriter(stream)
        env, row = synthetic_dispatch()
        record = collector(env, writer).update(row, 1)
        parsed = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(len(parsed), 1)
        self.assertEqual(tuple(parsed[0]), tuple(XLD_DIAGNOSTIC_FIELDS))
        self.assertEqual(parsed[0]['classification'], record['classification'])
        self.assertNotIn('[', stream.getvalue())
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises((ValueError, FloatingPointError)):
                writer.write(dict(record, best_grid_power_mw=value))
        with self.assertRaises(ValueError):
            writer.write(dict(record, unexpected='field'))
        self.assertEqual(len(list(csv.DictReader(io.StringIO(stream.getvalue())))), 1)

    def test_counterfactual_failure_is_not_misreported_as_actual_infeasibility(self):
        env, row = synthetic_dispatch()
        with mock.patch.object(env.physics, 'transition',
                side_effect=InfeasibleCascade('Synthetic candidate replay failure')):
            with self.assertRaises(XLDDispatchDiagnosticError) as failure:
                collector(env).update(row, 1)
        self.assertNotIsInstance(failure.exception, InfeasibleCascade)
        self.assertIn('Synthetic candidate replay failure', str(failure.exception))


class XLDEvaluationIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.physical = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        cls.physical.update(guaranteed_output_scale=.4, ecological_flow_scale=.4)
        cls.args = yaml.safe_load((ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical,
                               cls.args['experiment'])

    @staticmethod
    def actors():
        import torch
        class ConstantActor:
            def __init__(self):
                self.args = dict(action_bins=200)
                self.actor = torch.nn.Linear(1, 1)
            def prep_rollout(self):
                self.actor.eval()
            def act(self, observations, deterministic=True):
                if not deterministic:
                    raise AssertionError('Diagnostic evaluation must be deterministic')
                return torch.tensor([[0]])
        return [ConstantActor() for _ in range(5)]

    def test_optional_diagnostics_leave_evaluation_results_and_parameters_unchanged(self):
        import torch
        from harl.envs.cascade_reservoir.evaluation import evaluate_policy
        actors = self.actors()
        options = dict(seed=1, episode=50, training_steps=547500, gamma=1.,
            source_years=[1965], episode_days=8, deterministic=True)
        before = [copy.deepcopy(a.actor.state_dict()) for a in actors]
        baseline = evaluate_policy(actors, self.data, self.physical, **options)
        numpy_before, python_before = np.random.get_state(), random.getstate()
        torch_before = torch.get_rng_state().clone()
        output = io.StringIO()
        measured = evaluate_policy(actors, self.data, self.physical, **dict(options,
            xld_log=XLDDispatchWriter(output), evaluation_round=50, window_id='check_1965'))
        summary = measured.pop('xld_dispatch_diagnostics')
        self.assertEqual(measured, baseline)
        console = evaluate_policy(actors, self.data, self.physical, **dict(options,
            xld_diagnostics=True, evaluation_round=50, window_id='check_1965'))
        self.assertEqual(console.pop('xld_dispatch_diagnostics'), summary)
        self.assertEqual(console, baseline)
        self.assertEqual(summary['completed_steps'], 8)
        self.assertEqual(summary['violation_count'],
            baseline['service_shortfalls']['guaranteed_output']['by_reservoir_days'][2])
        self.assertEqual(len(list(csv.DictReader(io.StringIO(output.getvalue())))), 8)
        for actor, expected in zip(actors, before):
            for key, value in actor.actor.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        self.assertEqual(random.getstate(), python_before)
        self.assertEqual(numpy_before[0], np.random.get_state()[0])
        np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
        self.assertEqual(numpy_before[2:], np.random.get_state()[2:])

    def test_periodic_three_windows_reconcile_without_changing_rates_or_returns(self):
        from harl.envs.cascade_reservoir.periodic_evaluation import (
            evaluate_periodic_policy, select_periodic_windows)
        actors = self.actors()
        windows = select_periodic_windows(self.data, episode_days=8)
        options = dict(seed=1, evaluation_round=2, training_steps=96,
                       windows=windows, gamma=1.)
        baseline = evaluate_periodic_policy(actors, self.data, self.physical, **options)
        output = io.StringIO()
        measured = evaluate_periodic_policy(actors, self.data, self.physical, **dict(
            options, xld_log=XLDDispatchWriter(output)))
        console = evaluate_periodic_policy(actors, self.data, self.physical, **dict(
            options, xld_diagnostics=True))
        self.assertEqual(console, measured)
        self.assertEqual(measured['status'], 'ok')
        aggregate = measured.pop('xld_dispatch_diagnostics')
        self.assertEqual(aggregate['completed_steps'], 24)
        self.assertEqual(aggregate['violation_count'], sum(
            window['service_shortfalls']['guaranteed_output']['by_reservoir_days'][2]
            for window in measured['windows']))
        self.assertEqual(sum(aggregate['classification_counts'].values()), 24)
        self.assertEqual(sum(aggregate['group_counts'].values()), aggregate['violation_count'])
        for window in measured['windows']:
            summary = window.pop('xld_dispatch_diagnostics')
            self.assertEqual(summary['completed_steps'], 8)
            self.assertEqual(summary['violation_count'],
                window['service_shortfalls']['guaranteed_output']['by_reservoir_days'][2])
        self.assertEqual(measured, baseline)
        records = list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual(len(records), 24)
        self.assertEqual({row['window_id'] for row in records},
                         {w['window_id'] for w in windows})
        self.assertEqual({row['evaluation_round'] for row in records}, {'2'})

    def test_console_diagnostic_switch_requires_boolean_before_environment_creation(self):
        from harl.envs.cascade_reservoir.evaluation import evaluate_policy
        from harl.envs.cascade_reservoir.periodic_evaluation import evaluate_periodic_policy
        for invalid in (None, 0, 1, 'true', np.bool_(True)):
            with self.subTest(value=invalid), mock.patch(
                    'harl.envs.cascade_reservoir.evaluation.episode_environment') as create_env:
                with self.assertRaisesRegex(ValueError, 'boolean'):
                    evaluate_policy(self.actors(), self.data, self.physical, seed=1,
                        episode=1, training_steps=0, gamma=1., source_years=[1965],
                        episode_days=8, xld_diagnostics=invalid)
                create_env.assert_not_called()
            with self.subTest(periodic_value=invalid), mock.patch(
                    'harl.envs.cascade_reservoir.periodic_evaluation._validated_windows') as validate:
                with self.assertRaisesRegex(ValueError, 'boolean'):
                    evaluate_periodic_policy(self.actors(), self.data, self.physical, seed=1,
                        evaluation_round=1, training_steps=0, windows=[], gamma=1.,
                        xld_diagnostics=invalid)
                validate.assert_not_called()

    def test_missing_window_diagnostic_does_not_produce_a_survivor_aggregate(self):
        from harl.envs.cascade_reservoir.evaluation import evaluate_policy
        from harl.envs.cascade_reservoir.periodic_evaluation import (
            evaluate_periodic_policy, select_periodic_windows)
        windows = select_periodic_windows(self.data, episode_days=8)

        def missing_first(*args, **kwargs):
            measured = evaluate_policy(*args, **kwargs)
            if kwargs['window_id'] == windows[0]['window_id']:
                measured.pop('xld_dispatch_diagnostics')
            return measured

        with mock.patch('harl.envs.cascade_reservoir.periodic_evaluation.evaluate_policy',
                        side_effect=missing_first), mock.patch(
                'harl.envs.cascade_reservoir.xld_diagnostics.aggregate_xld_summaries',
                side_effect=AssertionError('Incomplete windows were aggregated')):
            result = evaluate_periodic_policy(self.actors(), self.data, self.physical,
                seed=1, evaluation_round=2, training_steps=96, windows=windows,
                gamma=1., xld_diagnostics=True)
        self.assertEqual(result['status'], 'ok')  # The physical evaluations succeeded.
        self.assertIsNone(result['xld_dispatch_diagnostics'])
        partial = result['partial_xld_dispatch_diagnostics']
        self.assertEqual(partial['status'], 'incomplete')
        self.assertEqual(partial['complete_window_count'], 2)
        self.assertEqual(partial['expected_window_count'], 3)
        self.assertIsNone(partial['windows'][0]['summary'])

    def test_failed_window_preserves_console_prefix_without_pooled_diagnostic(self):
        from harl.envs.cascade_reservoir.evaluation import evaluate_policy
        from harl.envs.cascade_reservoir.periodic_evaluation import (
            evaluate_periodic_policy, select_periodic_windows)
        windows = select_periodic_windows(self.data, episode_days=8)
        env = episode_environment(self.data, self.physical, windows[0]['source_years'],
                                  days=8, action_bins=200)
        actual_step = env.step

        def fail_second_day(actions):
            if env.index-env.start == 1:
                raise InfeasibleCascade('Synthetic second-day physical failure')
            return actual_step(actions)

        def evaluate_with_one_failure(*args, **kwargs):
            if kwargs['window_id'] != windows[0]['window_id']:
                return evaluate_policy(*args, **kwargs)
            with mock.patch('harl.envs.cascade_reservoir.evaluation.episode_environment',
                            return_value=env), mock.patch.object(env, 'step', side_effect=fail_second_day):
                return evaluate_policy(*args, **kwargs)

        with mock.patch('harl.envs.cascade_reservoir.periodic_evaluation.evaluate_policy',
                        side_effect=evaluate_with_one_failure), mock.patch(
                'harl.envs.cascade_reservoir.xld_diagnostics.aggregate_xld_summaries',
                side_effect=AssertionError('Failed windows were aggregated')):
            result = evaluate_periodic_policy(self.actors(), self.data, self.physical,
                seed=1, evaluation_round=2, training_steps=96, windows=windows,
                gamma=1., xld_diagnostics=True)
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['evaluator_return'])
        self.assertIsNone(result['xld_dispatch_diagnostics'])
        self.assertEqual(result['partial_xld_dispatch_diagnostics']['complete_window_count'], 2)
        first = result['windows'][0]
        self.assertEqual(first['failure']['completed_steps'], 1)
        self.assertEqual(first['partial_xld_dispatch_diagnostics']['completed_steps'], 1)

    def test_console_diagnostics_accept_shared_mappo_views_without_policy_updates(self):
        import torch
        from harl.algorithms.actors.shared_mappo import make_shared_mappo_agents
        from harl.envs.cascade_reservoir.evaluation import evaluate_policy
        env = episode_environment(self.data, self.physical, [1965], days=8, action_bins=7)
        network_args = {**self.args['model'], **self.args['algo'],
                        'hidden_sizes': [8, 8], 'action_bins': 7, 'share_param': True}
        actors = make_shared_mappo_agents(network_args, env.observation_space,
                                         env.action_space, torch.device('cpu'))
        core = actors[0].core
        self.assertTrue(all(actor.core is core for actor in actors))
        parameters = copy.deepcopy(core.actor.state_dict())
        optimizer = copy.deepcopy(core.actor_optimizer.state_dict())
        torch_before = torch.get_rng_state().clone()
        numpy_before, python_before = np.random.get_state(), random.getstate()
        options = dict(seed=1, episode=1, training_steps=0, gamma=1.,
                       source_years=[1965], episode_days=8, deterministic=True)
        with mock.patch.object(core.actor_optimizer, 'step',
                               side_effect=AssertionError('Evaluation updated the shared actor')):
            baseline = evaluate_policy(actors, self.data, self.physical, **options)
            measured = evaluate_policy(actors, self.data, self.physical,
                                       **dict(options, xld_diagnostics=True))
        summary = measured.pop('xld_dispatch_diagnostics')
        self.assertEqual(measured, baseline)
        self.assertEqual(summary['violation_count'],
            baseline['service_shortfalls']['guaranteed_output']['by_reservoir_days'][2])
        for key, expected in parameters.items():
            self.assertTrue(torch.equal(core.actor.state_dict()[key], expected))
        self.assertEqual(core.actor_optimizer.state_dict(), optimizer)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])

    def test_actual_failure_preserves_only_completed_diagnostic_days(self):
        from harl.envs.cascade_reservoir.evaluation import evaluate_policy
        env = episode_environment(self.data, self.physical, [1965], days=8, action_bins=200)
        actual_step = env.step
        def fail_second_day(actions):
            if env.index-env.start == 1:
                raise InfeasibleCascade('Synthetic second-day physical failure')
            return actual_step(actions)
        output = io.StringIO()
        with mock.patch('harl.envs.cascade_reservoir.evaluation.episode_environment',
                        return_value=env), mock.patch.object(env, 'step', side_effect=fail_second_day):
            with self.assertRaises(InfeasibleCascade) as failure:
                evaluate_policy(self.actors(), self.data, self.physical,
                    seed=1, episode=2, training_steps=96, gamma=1., source_years=[1965],
                    episode_days=8, xld_log=XLDDispatchWriter(output),
                    evaluation_round=2, window_id='failure_check')
        progress = failure.exception.evaluation_progress
        self.assertEqual(progress['completed_steps'], 1)
        self.assertEqual(progress['xld_dispatch_diagnostics']['completed_steps'], 1)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(output.getvalue())))), 1)
        self.assertEqual(progress['date'], '1965-01-02')

    def test_runner_routes_diagnostics_only_to_final_round_and_held_out_evaluation(self):
        from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
        from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
        from harl.runners.on_policy_ha_runner import OnPolicyHARunner
        args = copy.deepcopy(self.args)
        args['train'].update(episode_length=8, n_rollout_threads=6)
        args['experiment'].update(episodes=12, diagnostics=False)
        output = io.StringIO()
        writer = XLDDispatchWriter(io.StringIO())
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(output))
            training = stack.enter_context(mock.patch.object(OnPolicyHARunner, 'train',
                side_effect=AssertionError('This test must not train HAPPO')))
            runner = CascadeReservoirRunner(self.data, self.physical, args, 1,
                xld_log=writer, episode_log=io.StringIO(),
                periodic_windows=select_periodic_windows(self.data, episode_days=8))
            stack.callback(runner.close)
            for optimizer in [a.actor_optimizer for a in runner.actor]+[runner.critic.critic_optimizer]:
                stack.enter_context(mock.patch.object(optimizer, 'step',
                    side_effect=AssertionError('Optimizer step is forbidden')))
            def observation(*unused, **keywords):
                return dict(record_type='periodic_evaluation', status='ok', seed=1,
                            evaluation_round=keywords['evaluation_round'])
            periodic = stack.enter_context(mock.patch(
                'harl.runners.cascade_reservoir_runner.evaluate_periodic_policy',
                side_effect=observation))
            for update in range(3):
                runner.update_count, runner.training_steps = update, update*48
                runner.periodic_evaluation()
            self.assertEqual([call.kwargs['xld_log'] for call in periodic.call_args_list],
                             [None, None, writer])
            final = stack.enter_context(mock.patch(
                'harl.runners.cascade_reservoir_runner.evaluate_policy',
                return_value=dict(source_years=list(self.data.evaluation_years))))
            runner.completed_episodes = 12
            runner.final_evaluation()
            self.assertIs(final.call_args.kwargs['xld_log'], writer)
            self.assertEqual(final.call_args.kwargs['evaluation_round'], 2)
            self.assertEqual(final.call_args.kwargs['window_id'], 'held_out_sequence')
            self.assertEqual(runner.training_sampler.batches_consumed, 0)
            training.assert_not_called()

    def test_runner_routes_console_diagnostics_without_a_csv_writer(self):
        from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
        from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
        args = copy.deepcopy(self.args)
        args['train'].update(episode_length=4, n_rollout_threads=2)
        args['model'].update(hidden_sizes=[8, 8], action_bins=7)
        args['experiment'].update(episodes=4, diagnostics=False, xld_diagnostics=True)

        def observation(*unused, **keywords):
            return dict(record_type='periodic_evaluation', status='ok', seed=1,
                        evaluation_round=keywords['evaluation_round'])

        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            runner = CascadeReservoirRunner(self.data, self.physical, args, 1,
                periodic_windows=select_periodic_windows(self.data, episode_days=4))
            stack.callback(runner.close)
            periodic = stack.enter_context(mock.patch(
                'harl.runners.cascade_reservoir_runner.evaluate_periodic_policy',
                side_effect=observation))
            for update in range(3):
                runner.update_count, runner.training_steps = update, update*8
                runner.periodic_evaluation()
            self.assertEqual([call.kwargs['xld_diagnostics'] for call in periodic.call_args_list],
                             [False, False, True])
            self.assertTrue(all(call.kwargs['xld_log'] is None for call in periodic.call_args_list))
            final = stack.enter_context(mock.patch(
                'harl.runners.cascade_reservoir_runner.evaluate_policy',
                return_value=dict(source_years=list(self.data.evaluation_years))))
            runner.completed_episodes = 4
            runner.final_evaluation()
            self.assertTrue(final.call_args.kwargs['xld_diagnostics'])
            self.assertIsNone(final.call_args.kwargs['xld_log'])
            self.assertEqual(runner.training_sampler.batches_consumed, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

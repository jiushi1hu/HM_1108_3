"""Independent full-physics checks of investigation-only dispatch attribution."""
import copy
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
for path in (ROOT, ROOT/'tests'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.dispatch_diagnostics.feasibility import analyze_transition, DispatchDiagnosticError
from harl.envs.cascade_reservoir.data_loader import CascadeData, IDS
from harl.envs.cascade_reservoir.episodes import episode_environment
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics
from harl.envs.cascade_reservoir.xld_diagnostics import XLDDispatchDiagnostics
from test_xld_dispatch_diagnostics import synthetic_dispatch


def brute_grid(env, row, reservoir):
    actions = row['sampled_actions'].astype(float)/(env.action_bins-1)
    results = []
    for index in range(env.action_bins):
        candidate = actions.copy()
        candidate[reservoir] = index/(env.action_bins-1)
        results.append(env.physics.transition(row['storage_start'], row['external'],
            candidate, row['storage_lower_m3'], row['storage_upper_m3'],
            storage_constraints=row.get('storage_constraints')))
    return results


def synthetic(i=2, **kwargs):
    """Move the independently specified XLD fixture to any cascade station."""
    ecology = kwargs.pop('ecology', 0.)
    env, row = synthetic_dispatch(**kwargs)
    if i != 2:
        order = np.arange(5)
        order[i], order[2] = order[2], order[i]
        data = env.data
        for name in ('vmin', 'vmax', 'release_cap', 'turbine_cap', 'power_cap', 'head'):
            setattr(data, name, getattr(data, name)[order].copy())
        original_anchors = data.anchors.copy()
        data.anchors = {name:original_anchors[IDS[order[j]]].copy() for j,name in enumerate(IDS)}
        data.levels = lambda values: np.asarray([
            np.interp(values[j], data.anchors[name][:, 1], data.anchors[name][:, 0]) for j,name in enumerate(IDS)])
        if data.head_sensitivity is not None:
            data.head_sensitivity = {key:value[order].copy() for key,value in data.head_sensitivity.items()}
        for name in ('guaranteed_output_mw', 'ecological_flow_m3s'):
            setattr(data.service_requirements, name, getattr(data.service_requirements, name)[order].copy())
        physics = ReservoirPhysics(data, env.physics.config)
        storage, external, actions = (row[name][order].copy() for name in ('storage_start', 'external', 'sampled_actions'))
        row = physics.transition(storage, external, actions/(env.action_bins-1), data.vmin, data.vmax)
        row.update(date='2000-01-01', sampled_actions=actions, action_bins=env.action_bins,
                   normalized_actions=row['actions'].copy())
        row['actions'] = np.where(row['controls'] > 0., actions, 0)
        env.physics = physics
        env.storage = row['storage_end'].copy()
        env.last_transition = row
    env.data.service_requirements.ecological_flow_m3s[i] = ecology
    return env, row


class DispatchFeasibilityTest(unittest.TestCase):
    def test_historical_all_five_exact_grid_against_full_cascade(self):
        config = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        experiment = yaml.safe_load((ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())['experiment']
        for head in ('fixed', 'storage_sensitivity'):
            physical = dict(config, head_model=head, terminal_storage_equal_initial=False)
            data = CascadeData(ROOT/'data/cascade_reservoir_new', physical, experiment)
            env = episode_environment(data, physical, [1965], days=4, action_bins=21)
            # Diagnostic-only thresholds above actual capacities make every
            # grid search observable; no source arrays or executed rewards change.
            env.operation_reward.requirements = SimpleNamespace(
                guaranteed_output_mw=np.full(5, 1e6), ecological_flow_m3s=np.full(5, 1e6))
            for day in range(4):
                env.step(np.full(5, 20 if day == 0 else 0))
                row = env.last_transition
                records = analyze_transition(env, row)
                self.assertEqual([r['reservoir_id'] for r in records], list(IDS))
                for i, record in enumerate(records):
                    expected = brute_grid(env, row, i)
                    for term, source in (('guaranteed_output', 'power'), ('ecological_flow', 'release')):
                        values = [float(r[source][i]) for r in expected]
                        self.assertEqual(record[term+'_best_grid_action_index'], int(np.argmax(values)))
                        np.testing.assert_allclose(record[term+'_best_grid_actual'], max(values), rtol=1e-10, atol=1e-7)
                        self.assertEqual(record[term+'_meeting_grid_action_count'], 0)
                    self.assertEqual(record['ecological_flow_classification'], 'same_state_capacity_limit')
                json.dumps(records, allow_nan=False)

    def test_terminal_reachability_is_preserved_by_diagnostic_witnesses(self):
        physical = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        physical.update(head_model='fixed', terminal_storage_equal_initial=True)
        data = CascadeData(ROOT/'data/cascade_reservoir_new', physical)
        env = episode_environment(data, physical, [1965], days=4, action_bins=11)
        initial = env.storage.copy()
        # Large diagnostic requirements exercise every alternative-search path;
        # they do not change the terminal or physical constraints.
        env.operation_reward.requirements = SimpleNamespace(
            guaranteed_output_mw=np.full(5, 1e6), ecological_flow_m3s=np.full(5, 1e6))
        xld = XLDDispatchDiagnostics(env, seed=1, evaluation_round=0,
            window_id='terminal_reachability', phase='training_evaluation')
        for day in range(4):
            env.step((np.arange(5)*3+day*7) % env.action_bins)
            row = env.last_transition
            self.assertIn('storage_constraints', row)
            records = analyze_transition(env, row)
            xld_record = xld.update(row, day+1)
            for i, record in enumerate(records):
                expected = brute_grid(env, row, i)
                np.testing.assert_allclose(record['guaranteed_output_best_grid_actual'],
                    max(float(candidate['power'][i]) for candidate in expected), rtol=1e-10, atol=1e-7)
                for candidate in expected:
                    prefix = np.r_[0., np.cumsum(candidate['storage_end'])]
                    self.assertTrue(np.all(prefix[:, None]-prefix[None, :] <=
                                           row['storage_constraints']+1e-3))
                    if day == 3:
                        np.testing.assert_allclose(candidate['storage_end'], initial, rtol=0., atol=1e-3)
            self.assertEqual(xld_record['classification'], records[2]['guaranteed_output_classification'])
            self.assertAlmostEqual(xld_record['best_grid_power_mw'],
                records[2]['guaranteed_output_best_grid_actual'], places=7)
        np.testing.assert_allclose(env.storage, initial, rtol=0., atol=1e-3)

    def test_xld_equivalence_for_each_classification(self):
        cases = [dict(guarantee=10.), dict(guarantee=100.),
                 dict(guarantee=100., available_water=1000., outlet=30.),
                 dict(guarantee=10., turbine=1.), dict(available_water=0.),
                 dict(head_model='storage_sensitivity', external=100., guarantee=110., bins=2),
                 dict(head_model='storage_sensitivity', external=100., guarantee=114., bins=200),
                 dict(guarantee=0.)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                env, row = synthetic_dispatch(**kwargs)
                old = XLDDispatchDiagnostics(env, seed=1, evaluation_round=1,
                    window_id='equivalence', phase='training_evaluation').update(row, 1)
                new = analyze_transition(env, row)[2]
                self.assertEqual(new['guaranteed_output_classification'], old['classification'])
                if old['violating']:
                    self.assertEqual(new['guaranteed_output_best_grid_action_index'], old['best_grid_action_index'])
                    self.assertAlmostEqual(new['guaranteed_output_best_grid_actual'], old['best_grid_power_mw'], places=8)
                    self.assertAlmostEqual(new['guaranteed_output_continuous_max_actual'], old['continuous_max_power_mw'], places=8)
                    self.assertEqual(new['guaranteed_output_minimum_meeting_action_index'], old['minimum_meeting_action_index'])

    def test_all_station_nonmonotone_grid_gap_and_forced_cases(self):
        for i in range(5):
            with self.subTest(reservoir=IDS[i]):
                env, row = synthetic(i, head_model='storage_sensitivity', external=100.,
                    guarantee=110., bins=2, ecology=150.)
                record = analyze_transition(env, row)[i]
                self.assertEqual(record['guaranteed_output_classification'], 'discrete_grid_gap')
                self.assertEqual(record['ecological_flow_classification'], 'same_day_policy_alternative')
                self.assertAlmostEqual(record['guaranteed_output_continuous_best_release_m3s'], 160., places=8)
                self.assertAlmostEqual(record['guaranteed_output_continuous_max_actual'], .008829*160*80, places=8)
                self.assertLess(record['guaranteed_output_continuous_minimum_meeting_release_m3s'], 160.)
                self.assertAlmostEqual(record['ecological_flow_continuous_minimum_meeting_release_m3s'], 150., places=6)
                self.assertEqual(record['ecological_flow_minimum_meeting_action_index'], 1)
                self.assertAlmostEqual(record['ecological_flow_meeting_end_storage_delta_m3'], -100.*86400.)
                env, row = synthetic(i, available_water=0., ecology=1.)
                record = analyze_transition(env, row)[i]
                self.assertTrue(record['forced_action'])
                self.assertEqual(record['guaranteed_output_classification'], 'same_state_water_limit')
                self.assertEqual(record['ecological_flow_classification'], 'same_state_water_limit')
                self.assertEqual(record['ecological_flow_continuous_max_actual'], 0.)

    def test_original_downstream_sampled_action_and_observational_state(self):
        env, row = synthetic_dispatch(downstream_empty=True)
        env.data.service_requirements.ecological_flow_m3s[2] = 10.
        original, storage = copy.deepcopy(row), env.storage.copy()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        with mock.patch.object(env.physics, 'transition', wraps=env.physics.transition) as replay:
            records = analyze_transition(env, row)
        self.assertGreater(replay.call_count, 0)
        for call in replay.call_args_list:
            # Only XLD has an active/violated service in this fixture.
            np.testing.assert_array_equal(call.args[2][[0,1,3,4]], row['sampled_actions'][[0,1,3,4]]/(env.action_bins-1))
        for key in row:
            if isinstance(row[key], np.ndarray):
                np.testing.assert_array_equal(row[key], original[key])
        np.testing.assert_array_equal(env.storage, storage)
        self.assertIs(env.last_transition, row)
        self.assertEqual(env.index, 1)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(records[2]['guaranteed_output_classification'], 'same_day_policy_alternative')
        with mock.patch.object(env.physics, 'transition', side_effect=RuntimeError('broken counterfactual')):
            with self.assertRaises(DispatchDiagnosticError):
                analyze_transition(env, row)
        invalid = copy.deepcopy(row)
        del invalid['sampled_actions']
        with self.assertRaises(DispatchDiagnosticError):
            analyze_transition(env, invalid)

    def test_ecological_dispatch_ceiling_and_monotone_storage_breaks(self):
        env, original = synthetic(0, available_water=1000., guarantee=0., ecology=100.)
        data = env.data
        data.release_cap[1] = 30.
        storage = original['storage_start'].copy()
        storage[1] = data.vmax[1]
        env.physics = ReservoirPhysics(data, env.physics.config)
        row = env.physics.transition(storage, original['external'],
            original['sampled_actions']/(env.action_bins-1), data.vmin, data.vmax)
        row.update(date='2000-01-01', sampled_actions=original['sampled_actions'])
        record = analyze_transition(env, row)[0]
        self.assertEqual(record['ecological_flow_classification'], 'same_state_dispatch_limit')
        self.assertEqual(record['ecological_flow_continuous_max_actual'], 30.)
        self.assertFalse(record['ecological_flow_water_limit_proven'])

        env, original = synthetic(2, head_model='storage_sensitivity', external=100.,
                                  guarantee=1e5, bins=31)
        # A genuine interpolation kink creates separate quadratic power pieces.
        env.data.anchors['XLD'] = np.array([[0., 0.], [35., 86400.*50.], [100., 86400.*100.]])
        env.physics = ReservoirPhysics(env.data, env.physics.config)
        row = env.physics.transition(original['storage_start'], original['external'],
            original['sampled_actions']/(env.action_bins-1), env.data.vmin, env.data.vmax)
        row.update(date='2000-01-01', sampled_actions=original['sampled_actions'])
        record = analyze_transition(env, row)[2]
        normalized = original['sampled_actions']/(env.action_bins-1)
        dense = []
        for action in np.linspace(0., 1., 1001):
            commands = normalized.copy()
            commands[2] = action
            dense.append(env.physics.transition(row['storage_start'], row['external'],
                commands, row['storage_lower_m3'], row['storage_upper_m3'],
                storage_constraints=row.get('storage_constraints'))['power'][2])
        self.assertGreaterEqual(record['guaranteed_output_continuous_max_actual']+1e-7, max(dense))
        self.assertLess(record['guaranteed_output_continuous_max_actual']-max(dense), .01)


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Compare every batched counterfactual against authoritative scalar physics."""
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'tests'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from harl.envs.cascade_reservoir.batched_physics import batched_transition
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.episodes import episode_environment
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from test_xld_dispatch_diagnostics import synthetic_dispatch


class BatchedPhysicsTest(unittest.TestCase):
    def assert_scalar_equivalent(self, physics, storage, external, actions, lower=None, upper=None,
                                 constraints=None):
        storage, external, actions = [np.asarray(value, dtype=np.float64)
                                      for value in (storage, external, actions)]
        copies = [value.copy() for value in (storage, external, actions)]
        for value in (storage, external, actions):
            value.setflags(write=False)
        if lower is not None:
            lower, upper = [np.asarray(value, dtype=np.float64) for value in (lower, upper)]
            lower.setflags(write=False)
            upper.setflags(write=False)
        rng = np.random.get_state()
        options = {} if constraints is None else dict(storage_constraints=constraints)
        actual = batched_transition(physics, storage, external, actions, lower, upper, **options)
        expected = []
        for i in range(len(storage)):
            bounds = {} if lower is None else dict(
                storage_lower=lower if lower.ndim == 1 else lower[i],
                storage_upper=upper if upper.ndim == 1 else upper[i])
            if constraints is not None:
                bounds['storage_constraints'] = constraints if constraints.ndim == 2 else constraints[i]
            expected.append(physics.transition(storage[i], external[i], actions[i], **bounds))
        self.assertEqual(set(actual), set(expected[0]))
        for name in actual:
            # Scalar and vector kernels preserve the same operations; require
            # equality rather than an engineering tolerance that could hide a
            # capacity, endpoint or routing discrepancy.
            with self.subTest(field=name):
                np.testing.assert_array_equal(actual[name], np.asarray([row[name] for row in expected]))
        for original, unchanged in zip((storage, external, actions), copies):
            np.testing.assert_array_equal(original, unchanged)
        after = np.random.get_state()
        self.assertEqual(rng[0], after[0])
        np.testing.assert_array_equal(rng[1], after[1])
        self.assertEqual(rng[2:], after[2:])
        return actual

    def test_all_hundred_commands_at_every_reservoir_on_historical_states(self):
        physical = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        experiment = yaml.safe_load((ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())['experiment']
        grid = np.arange(100, dtype=np.float64)/99.
        for head_model in ('fixed', 'storage_sensitivity'):
            config = dict(physical, head_model=head_model)
            data = CascadeData(ROOT/'data/cascade_reservoir_new', config, experiment)
            env = episode_environment(data, config, [1965], days=4, action_bins=100)
            for commands in (np.full(5, 99), np.zeros(5), np.array([5,20,45,70,99]),
                             np.array([90,65,40,15,0])):
                env.step(commands)
                row = env.last_transition
                candidates = np.repeat((commands/99.)[None], 500, axis=0)
                for i in range(5):
                    candidates[i*100:(i+1)*100, i] = grid
                arrays = [np.repeat(row[name][None], 500, axis=0)
                          for name in ('storage_start', 'external', 'storage_lower_m3', 'storage_upper_m3')]
                with self.subTest(head=head_model, date=row['date']):
                    self.assert_scalar_equivalent(env.physics, arrays[0], arrays[1], candidates,
                                                  arrays[2], arrays[3], row.get('storage_constraints'))

    def test_forced_intervals_and_downstream_control_changes(self):
        for head_model in ('fixed', 'storage_sensitivity'):
            env, row = synthetic_dispatch(head_model=head_model, external=100., bins=100)
            storage = row['storage_start']
            # Requiring the unchanged storage makes each release forced to its
            # actual inflow, independently of the sampled categorical command.
            actual = self.assert_scalar_equivalent(env.physics,
                np.repeat(storage[None], 100, axis=0),
                np.repeat(row['external'][None], 100, axis=0),
                np.repeat((np.arange(100)/99.)[:, None], 5, axis=1), storage.copy(), storage.copy())
            np.testing.assert_array_equal(actual['controls'], np.zeros((100,5)))
            np.testing.assert_array_equal(actual['actions'], np.full((100,5), .5))
        env, row = synthetic_dispatch(available_water=30., downstream_empty=True, bins=100)
        candidates = np.repeat((row['sampled_actions']/99.)[None], 100, axis=0)
        candidates[:, 2] = np.arange(100)/99.
        actual = self.assert_scalar_equivalent(env.physics,
            np.repeat(row['storage_start'][None], 100, axis=0),
            np.repeat(row['external'][None], 100, axis=0), candidates)
        self.assertEqual(actual['controls'][0,3], 0.)
        self.assertTrue(np.all(actual['controls'][1:,3] == 1.))

    def test_tolerance_sized_interval_inversion_uses_bounded_midpoint(self):
        for head_model in ('fixed', 'storage_sensitivity'):
            env, row = synthetic_dispatch(head_model=head_model)
            storage = row['storage_start'].copy()
            storage[0] = -env.physics.dt*5e-9
            actions = np.zeros((1,5))
            actions[0,0] = .9
            actual = self.assert_scalar_equivalent(env.physics, storage[None],
                np.zeros((1,5)), actions, env.data.vmin.copy(), env.data.vmax.copy())
            self.assertGreater(actual['low'][0,0], actual['high'][0,0])
            self.assertEqual(actual['release'][0,0], 0.)
            self.assertEqual(actual['controls'][0,0], 0.)

    def test_invalid_rows_and_bounds_fail_without_mutation(self):
        env, row = synthetic_dispatch()
        storage = np.repeat(row['storage_start'][None], 2, axis=0)
        external, actions = np.zeros((2,5)), np.zeros((2,5))
        bad_inputs = [(storage[0], external, actions),
                      (storage, external[:1], actions),
                      (storage, external, np.full((2,5), np.nan)),
                      (storage, -np.ones((2,5)), actions),
                      (storage, external, np.full((2,5), 1.01))]
        for inputs in bad_inputs:
            with self.subTest(shapes=[np.shape(x) for x in inputs]), self.assertRaises(ValueError):
                batched_transition(env.physics, *inputs)
        for lower, upper in ((env.data.vmin, None), (env.data.vmin-1., env.data.vmax),
                             (env.data.vmax, env.data.vmin),
                             (np.zeros((1,5)), env.data.vmax)):
            with self.subTest(lower=np.shape(lower), upper=np.shape(upper)), self.assertRaises(ValueError):
                batched_transition(env.physics, storage, external, actions, lower, upper)
        external[1,0] = 1e9
        with self.assertRaises(InfeasibleCascade):
            batched_transition(env.physics, storage, external, actions)
        self.assertEqual(external[1,0], 1e9)

    def test_empty_batch_preserves_vector_shapes(self):
        for head_model in ('fixed', 'storage_sensitivity'):
            env, _ = synthetic_dispatch(head_model=head_model)
            result = batched_transition(env.physics, np.empty((0,5)), np.empty((0,5)), np.empty((0,5)))
            for name, value in result.items():
                self.assertEqual(value.shape, (0,) if name in ('reward','system_residual') else (0,5))


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Independent scalar-physics and environment-state checks of actor credit."""
import copy
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.common.valuenorm import ValueNorm
from harl.envs.cascade_reservoir.counterfactual_credit import (
    compute_counterfactual_credit, normalize_counterfactual_advantages,
    system_rewards_from_dispatch, validate_credit_assignment)
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.episodes import episode_environment
from harl.envs.cascade_reservoir.reward import OperationReward, reward_settings


def value_function(states):
    """A known state-dependent frozen critic; output is normalized units."""
    states = np.asarray(states, dtype=np.float32)
    # Explicit row reductions keep this analytical reference independent of
    # BLAS's batch-size-dependent float32 matrix-multiply accumulation order.
    return ((states[:, :5]*np.arange(1., 6., dtype=np.float32)).sum(axis=1)+
            3.*states[:, 10:15].sum(axis=1)+states[:, -1]).reshape(-1, 1)


class CounterfactualCreditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        physical = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        algorithm = yaml.safe_load((ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())
        cls.fixtures = {}
        for head in ('fixed', 'storage_sensitivity'):
            config = copy.deepcopy(physical)
            config['head_model'] = head
            config['reward_penalties'].update(guaranteed_output_weight=.7,
                ecological_flow_weight=.3, guaranteed_output_violation_weight=.8)
            data = CascadeData(ROOT/'data/cascade_reservoir_new', config, algorithm['experiment'])
            envs = [episode_environment(data, config, [year], 3, 5) for year in (1965, 1973)]
            records, snapshots, next_states = [], [], []
            for step in range(3):
                day, copies, states = [], [], []
                for lane, env in enumerate(envs):
                    env.step((np.arange(5)*2+step+lane) % 5)
                    day.append(copy.deepcopy(env.last_transition))
                    copies.append(copy.copy(env))
                    states.append(env.state().copy())
                records.append(day); snapshots.append(copies); next_states.append(states)
            context = {key:np.asarray([[row[source] for row in day] for day in records])
                for key, source in dict(storage_start='storage_start', external='external',
                    actions='sampled_actions', storage_lower='storage_lower_m3',
                    storage_upper='storage_upper_m3', controls='controls').items()}
            if 'storage_constraints' in records[0][0]:
                context['storage_constraints'] = np.asarray([
                    [row['storage_constraints'] for row in day] for day in records])
            probabilities = np.stack([np.roll(np.arange(1.,6.), i) for i in range(5)])
            probabilities = np.broadcast_to(probabilities/15., (3,2,5,5)).copy()
            masks = np.ones((3,2,1)); masks[-1] = 0.
            cls.fixtures[head] = dict(env=envs[0], records=records, snapshots=snapshots,
                context=context, next_states=np.asarray(next_states), probabilities=probabilities, masks=masks)

    def compute(self, fixture, *, batch=7, critic=value_function, normalizer=None, gamma=.995, **overrides):
        values = dict(context=fixture['context'], next_states=fixture['next_states'],
            policy_probabilities=fixture['probabilities'], continuation_masks=fixture['masks'])
        values.update(overrides)
        return compute_counterfactual_credit(fixture['env'].physics, fixture['env'].operation_reward,
            critic, normalizer, gamma=gamma, batch_size=batch, **values)

    def test_all_q_values_match_full_scalar_cascade_and_actual_environment_next_state(self):
        normalizer = ValueNorm(1)
        normalizer.update(np.array([[-10.],[30.]], dtype=np.float32))
        for head, fixture in self.fixtures.items():
            with self.subTest(head=head):
                result = self.compute(fixture, normalizer=normalizer)
                expected = np.empty((3,2,5,5))
                for step in range(3):
                    for lane in range(2):
                        row = fixture['records'][step][lane]
                        for actor in range(5):
                            for command in range(5):
                                actions = row['sampled_actions'].astype(float)/4.
                                actions[actor] = command/4.
                                transition = fixture['env'].physics.transition(row['storage_start'],
                                    row['external'], actions, row['storage_lower_m3'], row['storage_upper_m3'],
                                    storage_constraints=row.get('storage_constraints'))
                                fixture['env'].operation_reward.apply(transition)
                                value = 0.
                                if step != 2:
                                    next_env = copy.copy(fixture['snapshots'][step][lane])
                                    next_env.storage = transition['storage_end'].copy()
                                    next_env.previous_release = transition['release'].copy()
                                    value = float(normalizer.denormalize(value_function(next_env.state()[None]))[0,0])
                                expected[step,lane,actor,command] = transition['system_reward']+.995*value
                np.testing.assert_allclose(result['q_values'], expected, rtol=1e-7, atol=2e-6)
                baseline = np.sum(fixture['probabilities']*expected, axis=-1)
                selected = np.take_along_axis(expected, fixture['context']['actions'][...,None], axis=-1)[...,0]
                np.testing.assert_allclose(result['baseline'], baseline, rtol=1e-7, atol=2e-6)
                np.testing.assert_allclose(result['q_actual'], selected, rtol=1e-7, atol=2e-6)
                np.testing.assert_allclose(result['raw_advantages'], selected-baseline, rtol=1e-6, atol=2e-6)
                # Every actor-specific selected command reconstructs the same actual joint action.
                np.testing.assert_allclose(result['q_actual'],
                    np.broadcast_to(result['q_actual'][...,:1], result['q_actual'].shape), rtol=0., atol=1e-9)
                residual = np.sum(fixture['probabilities']*(result['q_values']-result['baseline'][...,None]), axis=-1)
                np.testing.assert_allclose(residual, 0., atol=1e-12)

    def test_chunk_limits_rng_inputs_and_old_normalizer_are_preserved(self):
        fixture = self.fixtures['storage_sensitivity']
        before = copy.deepcopy((fixture['context'], fixture['next_states'], fixture['probabilities']))
        normalizer = ValueNorm(1); normalizer.update(np.array([[-4.],[8.]], dtype=np.float32))
        moments = copy.deepcopy(normalizer.state_dict())
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        batches = []
        def predict(states):
            batches.append(len(states))
            return torch.as_tensor(value_function(states))
        result = self.compute(fixture, batch=3, critic=predict, normalizer=normalizer)
        self.assertTrue(batches); self.assertLessEqual(max(batches), 3)
        second = self.compute(fixture, batch=10000, normalizer=normalizer)
        np.testing.assert_array_equal(result['q_values'], second['q_values'])
        np.testing.assert_array_equal(result['normalized_advantages'], second['normalized_advantages'])
        for key in fixture['context']:
            np.testing.assert_array_equal(fixture['context'][key], before[0][key])
        np.testing.assert_array_equal(fixture['next_states'], before[1])
        np.testing.assert_array_equal(fixture['probabilities'], before[2])
        for key, value in moments.items():
            self.assertTrue(torch.equal(value, normalizer.state_dict()[key]))
        self.assertEqual(python_rng, random.getstate())
        for a,b in zip(numpy_rng, np.random.get_state()):
            np.testing.assert_array_equal(a,b)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))

    def test_terminal_and_gamma_zero_skip_all_critic_queries(self):
        fixture = self.fixtures['fixed']
        def forbidden(_):
            self.fail('Terminal or gamma-zero counterfactual queried a continuation value')
        first = self.compute(fixture, critic=forbidden, gamma=0.)
        terminal = self.compute(fixture, critic=forbidden, continuation_masks=np.zeros((3,2,1)))
        np.testing.assert_array_equal(first['q_values'], terminal['q_values'])

    def test_forced_choices_are_zero_credit_and_keep_original_sampled_commands(self):
        fixture = self.fixtures['fixed']; env = fixture['env']
        storage = env.data.initial_storage(1965)
        external = env.data.external[0]
        context = dict(storage_start=storage[None,None], external=external[None,None],
            actions=np.array([[[4,3,2,1,0]]]), storage_lower=storage[None,None],
            storage_upper=storage[None,None], controls=np.zeros((1,1,5)))
        result = self.compute(fixture, context=context, next_states=np.zeros((1,1,38)),
            policy_probabilities=np.full((1,1,5,5), .2), continuation_masks=np.zeros((1,1,1)))
        np.testing.assert_array_equal(result['raw_advantages'], np.zeros((1,1,5)))
        np.testing.assert_array_equal(result['normalized_advantages'], np.zeros((1,1,5)))
        np.testing.assert_allclose(np.ptp(result['q_values'], axis=-1), 0., atol=1e-12)
        self.assertEqual(result['summary']['controllable_rows'], 0)

    def test_common_rms_preserves_sign_and_excludes_forced_rows(self):
        raw = np.array([[[-2., 1., 4., 1000., 0.]]]); controls = np.array([[[1.,1.,1.,0.,1.]]])
        normalized, scale = normalize_counterfactual_advantages(raw, controls)
        self.assertAlmostEqual(scale, np.sqrt(21./4.)+1e-5)
        np.testing.assert_allclose(normalized[controls>0], raw[controls>0]/scale)
        self.assertEqual(normalized[0,0,3], 0.)
        positive, _ = normalize_counterfactual_advantages(np.ones((1,1,5))*2., np.ones((1,1,5)))
        self.assertTrue(np.all(positive > 0.))

    def test_vector_reward_handles_roundoff_inactive_services_and_disabled_penalties(self):
        fixture = self.fixtures['fixed']; data = fixture['env'].data
        requirements = copy.copy(data.service_requirements)
        requirements.guaranteed_output_mw = np.array([100.,0.,100.,100.,100.])
        requirements.ecological_flow_m3s = np.array([10.,10.,0.,10.,10.])
        fake_data = copy.copy(data); fake_data.service_requirements = requirements
        power = np.array([[100.-1e-9, 0., 100.-1e-7, 80., 200.], [0.,0.,0.,0.,0.]])
        release = np.array([[10.,10.-1e-10,0.,0.,20.], [0.,0.,0.,0.,0.]])
        for enabled in (True, False):
            settings = reward_settings(dict(reward_penalties=dict(enabled=enabled,
                guaranteed_output_weight=.7, ecological_flow_weight=.3, guaranteed_output_violation_weight=.8)))
            reward = OperationReward(fake_data, settings)
            result = system_rewards_from_dispatch(reward, dict(power=power, release=release))
            for i in range(2):
                actual = dict(power=power[i].copy(), release=release[i].copy(), reward=0.)
                reward.apply(actual)
                self.assertAlmostEqual(result[i], actual['system_reward'], places=14)
        fake_data.service_requirements = None
        reward = OperationReward(fake_data, reward_settings({}))
        np.testing.assert_allclose(system_rewards_from_dispatch(reward, dict(power=power, release=release)),
            .5*(power/data.power_cap).sum(axis=-1)+.5*power.sum(axis=-1)/data.power_cap.sum())

    def test_invalid_probabilities_commands_masks_and_value_width_are_rejected(self):
        fixture = self.fixtures['fixed']
        for p in (np.full((3,2,5,5), .3), np.full((3,2,5,5), np.nan)):
            with self.assertRaisesRegex(ValueError, 'probabilities'):
                self.compute(fixture, policy_probabilities=p)
        invalid = copy.deepcopy(fixture['context']); invalid['actions'][0,0,0] = -1
        with self.assertRaisesRegex(ValueError, 'original integer'):
            self.compute(fixture, context=invalid)
        with self.assertRaisesRegex(ValueError, 'binary'):
            self.compute(fixture, continuation_masks=np.full((3,2,1), .5))
        with self.assertRaisesRegex(ValueError, 'scalar shared-team'):
            self.compute(fixture, critic=lambda states:np.zeros((len(states),6)))
        for mode in ('team_gae', 'counterfactual'):
            self.assertEqual(validate_credit_assignment(mode), mode)
        with self.assertRaises(ValueError):
            validate_credit_assignment('individual')


if __name__ == '__main__':
    unittest.main(verbosity=2)

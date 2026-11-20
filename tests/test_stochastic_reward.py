"""Unit tests for the optional mean / CVaR training reward.

Run from the repository root:

    python -m unittest tests.test_stochastic_reward -v
"""

import os
import random
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.environment import Env
from env.message_passing_evl import MassagePassingEval
from env.stochastic_jsp import (
    StochasticRewardConfig,
    resolve_cvar_weight,
    scenario_objective,
)


def random_instance(n_j, n_m, seed, low=1, high=20):
    rng = np.random.default_rng(seed)
    times = rng.integers(low, high, size=(n_j, n_m))
    machines = np.stack([rng.permutation(np.arange(1, n_m + 1)) for _ in range(n_j)])
    return np.stack([times, machines], axis=0)


def make_env(instances, cfg=None, seed=0):
    np.random.seed(seed)
    random.seed(seed)
    env = Env()
    if cfg is not None:
        env.configure_stochastic_reward(cfg)
    env.reset(
        instances=instances,
        init_sol_type="spt",
        tabu_size=1,
        device="cpu",
        mask_previous_action=False,
        longest_path_finder="pytorch",
    )
    return env


def deterministic_makespan(env):
    evaluator = MassagePassingEval()
    _est, _lst, make_span, _count, _fwd, _bwd = evaluator.eval(
        env.G_batch, num_nodes_per_example=env.num_nodes_per_example
    )
    return make_span.detach().cpu().numpy().reshape(-1)


def first_actions(env):
    moves, _mark, _paths = env.get_candidate_moves()
    rows = []
    for mv in moves:
        if mv:
            rows.append(mv[0][:1, :2])
    if not rows:
        raise AssertionError("expected at least one N5 move")
    return torch.cat(rows, dim=0)


class FlagTests(unittest.TestCase):
    def test_defaults_train_on_the_mean_and_can_stay_off(self):
        cfg = StochasticRewardConfig.from_flags()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.cvar_weight, 0.0)
        self.assertEqual(cfg.reward_objective, "mean")
        # A CVaR flag with n_scenarios=0 must not turn scoring on.
        cvar_off = StochasticRewardConfig.from_flags(
            n_scenarios=0,
            cv=0.3,
            distribution="lognormal",
            alpha=0.9,
            reward_objective="cvar",
            cvar_weight=None,
            seed=1,
        )
        self.assertFalse(cvar_off.enabled)
        self.assertEqual(cvar_off.cvar_weight, 1.0)
        self.assertEqual(resolve_cvar_weight("mean", None), 0.0)
        self.assertEqual(resolve_cvar_weight("cvar", None), 1.0)
        self.assertEqual(resolve_cvar_weight("mix", 0.25), 0.25)
        with self.assertRaises(ValueError):
            resolve_cvar_weight("mean", 0.25)
        with self.assertRaises(ValueError):
            resolve_cvar_weight("mix", None)
        with self.assertRaises(ValueError):
            resolve_cvar_weight("mix", 1.5)

    def test_scenario_objective_is_the_convex_combination(self):
        mean = np.array([10.0, 12.0])
        cvar = np.array([14.0, 18.0])
        np.testing.assert_allclose(scenario_objective((mean, cvar), 0.0), mean)
        np.testing.assert_allclose(scenario_objective((mean, cvar), 1.0), cvar)
        np.testing.assert_allclose(scenario_objective((mean, cvar), 0.25), 0.75 * mean + 0.25 * cvar)
        with self.assertRaises(ValueError):
            scenario_objective((mean, cvar), 1.2)


class RewardHookTests(unittest.TestCase):
    def test_default_step_does_not_evaluate_scenarios(self):
        import env.environment as environment

        inst = random_instance(3, 3, seed=9)
        calls = []
        real = environment.evaluate_scenarios

        def wrapped(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        environment.evaluate_scenarios = wrapped
        try:
            env = make_env([inst], seed=0)
            self.assertIsNone(env.stochastic_objs)
            env.step(action=first_actions(env))
            cfg = StochasticRewardConfig.from_flags(n_scenarios=0, reward_objective="cvar", cv=0.4)
            flagged = make_env([inst], cfg=cfg, seed=0)
            self.assertIsNone(flagged.stochastic_objs)
            flagged.step(action=first_actions(flagged))
        finally:
            environment.evaluate_scenarios = real
        self.assertEqual(calls, [])

    def test_cv0_and_nominal_match_the_deterministic_evaluator(self):
        insts = [random_instance(3, 3, seed=5), random_instance(4, 3, seed=6)]
        for distribution, cv, weight in (
            ("lognormal", 0.0, 0.0),
            ("uniform", 0.0, 0.0),
            ("nominal", 0.4, 0.0),
            ("nominal", 0.4, 1.0),
        ):
            cfg = StochasticRewardConfig(
                n_scenarios=4,
                cv=cv,
                distribution=distribution,
                alpha=0.9,
                cvar_weight=weight,
                seed=2,
            )
            det = make_env(insts, seed=0)
            st = make_env(insts, cfg=cfg, seed=0)
            fresh = deterministic_makespan(det)
            np.testing.assert_allclose(st.stochastic_objs.numpy(), fresh, atol=1e-5)
            np.testing.assert_allclose(st.stochastic_mean.numpy(), det.current_objs.numpy(), atol=1e-5)
            np.testing.assert_allclose(st.stochastic_cvar.numpy(), det.current_objs.numpy(), atol=1e-5)
            self.assertTrue(torch.equal(st.G_batch.dur, det.G_batch.dur))
            torch.testing.assert_close(st.G_batch.x, det.G_batch.x)

            action = first_actions(det)
            old_incumbent = st.stochastic_incumbent.clone()
            _g, reward_det, (moves_d, mark_d, _p) = det.step(action=action.clone())
            _g, reward_st, (moves_s, mark_s, _p) = st.step(action=action.clone())
            expected = torch.clamp(old_incumbent - st.stochastic_objs, min=0).unsqueeze(-1)
            torch.testing.assert_close(reward_st, expected)
            np.testing.assert_allclose(reward_st.numpy(), reward_det.numpy(), atol=1e-4)
            np.testing.assert_allclose(st.stochastic_objs.numpy(), st.current_objs.numpy(), atol=1e-5)
            np.testing.assert_allclose(st.current_objs.numpy(), deterministic_makespan(st), atol=1e-5)
            self.assertTrue(torch.equal(st.G_batch.dur, det.G_batch.dur))
            torch.testing.assert_close(st.G_batch.x, det.G_batch.x)
            torch.testing.assert_close(st.G_batch.est, det.G_batch.est)
            self.assertEqual(len(moves_s), len(moves_d))
            for left, right in zip(moves_s, moves_d):
                if not left and not right:
                    continue
                self.assertTrue(torch.equal(left[0], right[0]))
            np.testing.assert_array_equal(mark_s, mark_d)

    def test_mean_versus_cvar_changes_the_scalar(self):
        inst = random_instance(4, 4, seed=3)

        def opened(weight, inflate):
            cfg = StochasticRewardConfig(
                n_scenarios=10,
                cv=0.0,
                distribution="nominal",
                alpha=0.9,
                cvar_weight=weight,
                seed=0,
            )
            env = make_env([inst], cfg=cfg, seed=1)
            if inflate:
                block = env._scenario_batch[0]
                scaled = np.array(block.scenarios, dtype=np.float64, copy=True)
                scaled[-1] *= 4.0
                block.scenarios = scaled
                env.score_current_order()
            return env

        nominal = opened(0.0, inflate=False)
        det = make_env([inst], seed=1)
        np.testing.assert_allclose(nominal.stochastic_objs.numpy(), det.current_objs.numpy(), atol=1e-5)

        mean_env = opened(0.0, inflate=True)
        cvar_env = opened(1.0, inflate=True)
        mix_env = opened(0.5, inflate=True)
        mean_v = float(mean_env.stochastic_objs)
        cvar_v = float(cvar_env.stochastic_objs)
        self.assertGreater(cvar_v, mean_v)
        self.assertAlmostEqual(float(mix_env.stochastic_objs), 0.5 * (mean_v + cvar_v), places=4)
        # alpha=0.9, n=10: CVaR is the worst scenario, strictly above the mean.
        self.assertAlmostEqual(cvar_v, float(cvar_env.stochastic_cvar), places=4)
        self.assertGreater(float(cvar_env.stochastic_cvar), float(cvar_env.stochastic_mean))
        # The nominal CPM features are still the unscaled instance.
        self.assertTrue(torch.equal(mean_env.G_batch.dur, det.G_batch.dur))
        torch.testing.assert_close(mean_env.G_batch.x, det.G_batch.x)
        torch.testing.assert_close(mean_env.current_objs, det.current_objs)

    def test_noisy_step_keeps_nominal_features_and_fixed_scenarios(self):
        insts = [random_instance(3, 4, seed=11), random_instance(3, 3, seed=12)]

        def opened(weight):
            cfg = StochasticRewardConfig(
                n_scenarios=10,
                cv=0.3,
                distribution="lognormal",
                alpha=0.9,
                cvar_weight=weight,
                seed=19,
            )
            return make_env(insts, cfg=cfg, seed=2)

        det = make_env(insts, seed=2)
        mean_env = opened(0.0)
        cvar_env = opened(1.0)
        self.assertTrue(
            np.all(cvar_env.stochastic_cvar.numpy() + 1e-6 >= cvar_env.stochastic_mean.numpy())
        )
        self.assertTrue(
            np.any(cvar_env.stochastic_cvar.numpy() > cvar_env.stochastic_mean.numpy() + 1e-3)
        )
        self.assertFalse(np.allclose(mean_env.stochastic_objs.numpy(), cvar_env.stochastic_objs.numpy()))
        torch.testing.assert_close(mean_env.G_batch.x, det.G_batch.x)
        torch.testing.assert_close(mean_env.current_objs, det.current_objs)

        before = [np.array(block.scenarios, copy=True) for block in mean_env._scenario_batch]
        dur_before = mean_env.G_batch.dur.clone()
        old = mean_env.stochastic_incumbent.clone()
        action = first_actions(det)
        _g, reward, (moves_s, _mark_s, _p) = mean_env.step(action=action.clone())
        _g, _reward_d, (moves_d, _mark_d, _p) = det.step(action=action.clone())
        expected = torch.clamp(old - mean_env.stochastic_objs, min=0).unsqueeze(-1)
        torch.testing.assert_close(reward, expected)
        for stored, block in zip(before, mean_env._scenario_batch):
            np.testing.assert_array_equal(block.scenarios, stored)
        self.assertTrue(torch.equal(mean_env.G_batch.dur, dur_before))
        torch.testing.assert_close(mean_env.G_batch.x, det.G_batch.x)
        torch.testing.assert_close(mean_env.G_batch.est, det.G_batch.est)
        torch.testing.assert_close(mean_env.current_objs, det.current_objs)
        for left, right in zip(moves_s, moves_d):
            if not left and not right:
                continue
            self.assertTrue(torch.equal(left[0], right[0]))

    def test_seed_repeats_and_does_not_touch_the_global_rng(self):
        inst = random_instance(3, 3, seed=4)

        def objective(seed):
            cfg = StochasticRewardConfig(
                n_scenarios=6,
                cv=0.25,
                distribution="lognormal",
                alpha=0.9,
                cvar_weight=0.0,
                seed=seed,
            )
            env = make_env([inst], cfg=cfg, seed=0)
            return env.stochastic_objs.numpy().copy()

        np.testing.assert_array_equal(objective(7), objective(7))
        self.assertFalse(np.array_equal(objective(7), objective(8)))

        np.random.seed(0)
        random.seed(0)
        cfg = StochasticRewardConfig(
            n_scenarios=5, cv=0.3, distribution="lognormal", alpha=0.9, cvar_weight=0.0, seed=1
        )
        make_env([inst], cfg=cfg, seed=0)
        after_stoch = np.random.rand(4)
        np.random.seed(0)
        random.seed(0)
        make_env([inst], seed=0)
        after_det = np.random.rand(4)
        np.testing.assert_array_equal(after_stoch, after_det)


if __name__ == "__main__":
    unittest.main()

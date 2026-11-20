"""Unit tests for stochastic processing-time scenarios on a fixed JSSP sequence.

Run from the repository root:

    python -m unittest tests.test_stochastic_jsp -v
"""

import io
import os
import random
import sys
import unittest
from contextlib import redirect_stdout

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.environment import Env
from env.message_passing_evl import MassagePassingEval
from env.stochastic_jsp import (
    empirical_cvar,
    evaluate_scenarios,
    generate_stochastic_scenarios,
    main,
    sample_processing_times,
)


def random_instance(n_j, n_m, seed, low=1, high=20, dtype=np.int64):
    rng = np.random.default_rng(seed)
    times = rng.integers(low, high, size=(n_j, n_m)).astype(dtype, copy=False)
    machines = np.stack([rng.permutation(np.arange(1, n_m + 1)) for _ in range(n_j)]).astype(dtype, copy=False)
    return np.stack([times, machines], axis=0)


def spt_graph(instances, seed=0):
    """Deterministic SPT disjunctive graph. Tie breaks follow the seeded global RNG."""
    np.random.seed(seed)
    random.seed(seed)
    env = Env()
    env.reset(
        instances=instances,
        init_sol_type="spt",
        tabu_size=1,
        device="cpu",
        mask_previous_action=False,
        longest_path_finder="pytorch",
    )
    return env


def deterministic_makespan(graph, num_nodes_per_example=None):
    evaluator = MassagePassingEval()
    _est, _lst, make_span, _count, _fwd, _bwd = evaluator.eval(
        graph, num_nodes_per_example=num_nodes_per_example
    )
    return make_span.detach().cpu().numpy()


class GeneratorTests(unittest.TestCase):
    def test_reproducible_under_fixed_seed(self):
        inst = random_instance(4, 3, seed=1)
        for distribution in ("lognormal", "uniform"):
            first = generate_stochastic_scenarios(
                inst, n_scenarios=8, cv=0.25, distribution=distribution, seed=7
            )
            second = generate_stochastic_scenarios(
                inst, n_scenarios=8, cv=0.25, distribution=distribution, seed=7
            )
            other = generate_stochastic_scenarios(
                inst, n_scenarios=8, cv=0.25, distribution=distribution, seed=8
            )
            np.testing.assert_array_equal(first.scenarios, second.scenarios)
            self.assertFalse(np.array_equal(first.scenarios, other.scenarios))
            np.testing.assert_array_equal(first.machines, inst[1])
            np.testing.assert_array_equal(first.nominal_times, inst[0])
            self.assertTrue(np.all(first.scenarios > 0))

        lognormal = sample_processing_times(inst[0], 8, 0.25, "lognormal", seed=7)
        uniform = sample_processing_times(inst[0], 8, 0.25, "uniform", seed=7)
        self.assertFalse(np.array_equal(lognormal, uniform))

    def test_cv0_and_nominal_are_the_original_instance(self):
        inst = random_instance(5, 4, seed=2, dtype=np.int32)
        n_scenarios = 6
        for distribution in ("lognormal", "uniform", "nominal"):
            got = generate_stochastic_scenarios(
                inst, n_scenarios=n_scenarios, cv=0.0, distribution=distribution, seed=99
            )
            expected = np.broadcast_to(inst[0], (n_scenarios,) + inst[0].shape)
            np.testing.assert_array_equal(got.scenarios, expected)
            self.assertEqual(got.scenarios.dtype, inst[0].dtype)
            np.testing.assert_array_equal(got.machines, inst[1])

        nominal = generate_stochastic_scenarios(
            inst, n_scenarios=1, cv=0.4, distribution="nominal", seed=3
        )
        rebuilt = np.stack([nominal.scenarios[0], nominal.machines], axis=0)
        np.testing.assert_array_equal(rebuilt, inst)
        self.assertEqual(nominal.scenarios.dtype, inst[0].dtype)

        # Caller-side edits must not alias the stored nominal data.
        inst[0][0, 0] = 12345
        self.assertNotEqual(int(nominal.nominal_times[0, 0]), 12345)

    def test_batch_and_list_are_reproducible(self):
        a = random_instance(3, 3, seed=10)
        b = random_instance(3, 3, seed=11)
        stack = np.stack([a, b], axis=0)
        first = generate_stochastic_scenarios(stack, n_scenarios=5, cv=0.2, distribution="lognormal", seed=4)
        second = generate_stochastic_scenarios(stack, n_scenarios=5, cv=0.2, distribution="lognormal", seed=4)
        np.testing.assert_array_equal(first.scenarios, second.scenarios)
        self.assertEqual(first.scenarios.shape, (2, 5, 3, 3))
        np.testing.assert_array_equal(first.machines, stack[:, 1])
        zero = generate_stochastic_scenarios(
            stack, n_scenarios=3, cv=0.0, distribution="uniform", seed=1
        )
        nominal_times = stack[:, 0][:, None, ...]
        np.testing.assert_array_equal(zero.scenarios, np.broadcast_to(nominal_times, (2, 3, 3, 3)))
        self.assertEqual(zero.scenarios.dtype, stack.dtype)

        small = random_instance(3, 3, seed=12)
        wide = random_instance(4, 5, seed=13)
        listed = generate_stochastic_scenarios(
            [small, wide], n_scenarios=4, cv=0.15, distribution="uniform", seed=20
        )
        again = generate_stochastic_scenarios(
            [small, wide], n_scenarios=4, cv=0.15, distribution="uniform", seed=20
        )
        self.assertEqual(len(listed), 2)
        np.testing.assert_array_equal(listed[0].scenarios, again[0].scenarios)
        np.testing.assert_array_equal(listed[1].scenarios, again[1].scenarios)
        alone = generate_stochastic_scenarios(
            wide, n_scenarios=4, cv=0.15, distribution="uniform", seed=21
        )
        np.testing.assert_array_equal(listed[1].scenarios, alone.scenarios)
        np.testing.assert_array_equal(listed[0].machines, small[1])
        np.testing.assert_array_equal(listed[1].machines, wide[1])

    def test_lognormal_and_uniform_match_the_cv_formulas(self):
        nominal = np.array([[10, 20], [30, 40]], dtype=np.int64)
        cv = 0.25
        n_scenarios = 5
        seed = 11
        got = sample_processing_times(nominal, n_scenarios, cv, "lognormal", seed=seed)
        rng = np.random.default_rng(seed)
        sigma = np.sqrt(np.log1p(cv ** 2))
        mu = np.log(nominal.astype(np.float64)) - 0.5 * sigma ** 2
        expected = np.exp(mu + sigma * rng.standard_normal(size=(n_scenarios, 2, 2)))
        np.testing.assert_allclose(got, expected)

        got_u = sample_processing_times(nominal, n_scenarios, cv, "uniform", seed=seed)
        rng = np.random.default_rng(seed)
        half_width = cv * np.sqrt(3.0)
        noise = rng.uniform(-half_width, half_width, size=(n_scenarios, 2, 2))
        np.testing.assert_allclose(got_u, nominal * (1.0 + noise))
        self.assertTrue(np.all(got_u >= nominal * (1.0 - half_width) - 1e-8))
        self.assertTrue(np.all(got_u <= nominal * (1.0 + half_width) + 1e-8))

    def test_generator_does_not_touch_global_rng(self):
        inst = random_instance(3, 3, seed=0)
        np.random.seed(123)
        before = np.random.rand(6)
        np.random.seed(123)
        generate_stochastic_scenarios(inst, n_scenarios=9, cv=0.3, distribution="lognormal", seed=5)
        sample_processing_times(inst[0], 9, 0.0, "uniform", seed=5)
        after = np.random.rand(6)
        np.testing.assert_array_equal(before, after)

    def test_rejects_bad_requests(self):
        inst = random_instance(2, 2, seed=0)
        with self.assertRaises(ValueError):
            generate_stochastic_scenarios(inst, n_scenarios=2, cv=-0.1, distribution="lognormal", seed=0)
        with self.assertRaises(ValueError):
            sample_processing_times(inst[0], n_scenarios=2, cv=0.9, distribution="uniform", seed=0)
        zero = inst.copy()
        zero[0, 0, 0] = 0
        with self.assertRaises(ValueError):
            sample_processing_times(zero[0], n_scenarios=2, cv=0.2, distribution="lognormal", seed=0)
        # cv = 0 still accepts a zero nominal time and keeps it.
        kept = sample_processing_times(zero[0], n_scenarios=3, cv=0.0, distribution="lognormal", seed=0)
        np.testing.assert_array_equal(kept, np.broadcast_to(zero[0], (3,) + zero[0].shape))


def _rockafellar_uryasev(samples, alpha):
    """Minimum of the empirical Rockafellar-Uryasev function over the sample points."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    scale = 1.0 / (x.size * (1.0 - float(alpha)))
    best = None
    for zeta in x:
        value = float(zeta + scale * np.maximum(x - zeta, 0.0).sum())
        if best is None or value < best:
            best = value
    return best


class CvarTests(unittest.TestCase):
    def test_alpha_0_9_n_10_is_the_worst_10_percent(self):
        samples = np.arange(1, 11, dtype=np.float64)  # 1..10
        # (1-0.9)*10 is 0.9999999999999998 in float64. The snap still leaves
        # a tail of one scenario: the worst makespan, not the mean of two.
        self.assertLess((1.0 - 0.9) * 10.0, 1.0)
        cvar = float(empirical_cvar(samples, 0.9))
        self.assertEqual(cvar, 10.0)
        self.assertNotAlmostEqual(cvar, 9.5)
        shuffled = np.array([4, 10, 1, 7, 3, 9, 2, 8, 6, 5], dtype=np.float64)
        self.assertEqual(float(empirical_cvar(shuffled, 0.9)), 10.0)
        batch = np.stack([samples, samples[::-1], shuffled], axis=0)
        np.testing.assert_allclose(empirical_cvar(batch, 0.9), [10.0, 10.0, 10.0])

    def test_integer_tail_is_the_mean_of_that_many_worst_scenarios(self):
        samples = np.arange(1, 11, dtype=np.float64)
        # alpha=0.8 -> w=2. The float product is 1.9999999999999996 and must snap.
        self.assertEqual(float(empirical_cvar(samples, 0.8)), 9.5)
        # alpha=0.5 -> w=5, mean of 6..10.
        self.assertEqual(float(empirical_cvar(samples, 0.5)), 8.0)
        self.assertEqual(float(empirical_cvar(samples, 0.0)), float(samples.mean()))

    def test_weighted_remainder_keeps_tail_mass(self):
        samples = np.arange(1, 11, dtype=np.float64)
        # alpha=0.75 -> w=2.5. (10 + 9 + 0.5*8) / 2.5 = 9.2
        cvar = float(empirical_cvar(samples, 0.75))
        self.assertAlmostEqual(cvar, 9.2)
        self.assertNotAlmostEqual(cvar, 9.5)
        self.assertNotAlmostEqual(cvar, 9.0)
        # Thinner than one scenario: still the worst, and the mass is 1-alpha.
        self.assertEqual(float(empirical_cvar(samples, 0.95)), 10.0)

    def test_alpha_0_is_the_mean_and_alpha_1_is_the_worst(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=20.0, scale=5.0, size=(6, 13))
        np.testing.assert_allclose(empirical_cvar(samples, 0.0), samples.mean(axis=-1))
        np.testing.assert_allclose(empirical_cvar(samples, 1.0), samples.max(axis=-1))
        self.assertEqual(float(empirical_cvar(np.array(4.0), 0.3)), 4.0)

    def test_mean_is_at_most_cvar_on_the_same_samples(self):
        rng = np.random.default_rng(1)
        samples = rng.normal(loc=20.0, scale=5.0, size=(6, 13))
        for alpha in (0.0, 0.1, 0.5, 0.9, 0.95, 1.0):
            cvar = empirical_cvar(samples, alpha)
            self.assertTrue(np.all(samples.mean(axis=-1) <= np.asarray(cvar) + 1e-8))

    def test_matches_rockafellar_uryasev_minimization(self):
        samples = np.array([1, 4, 2, 9, 8, 3, 7, 5, 6, 10], dtype=np.float64)
        for alpha in (0.0, 0.5, 0.75, 0.9, 0.95):
            self.assertAlmostEqual(
                float(empirical_cvar(samples, alpha)),
                _rockafellar_uryasev(samples, alpha),
                places=7,
            )


class EvaluatorTests(unittest.TestCase):
    def test_cv0_and_nominal_match_deterministic_evaluator(self):
        sizes = [(3, 3), (3, 4), (4, 3), (5, 4), (4, 5)]
        instances = [random_instance(n_j, n_m, seed=100 + i) for i, (n_j, n_m) in enumerate(sizes)]
        ft_path = os.path.join(ROOT, "test_data_jssp", "ft6x6.npy")
        ft06 = np.load(ft_path)[0]
        instances.append(ft06)

        # Several instances in one Env batch, including mixed sizes and ft06.
        env = spt_graph(instances, seed=0)
        det = deterministic_makespan(env.G_batch, num_nodes_per_example=env.num_nodes_per_example)
        self.assertEqual(det.shape[0], len(instances))

        cv0 = [
            generate_stochastic_scenarios(inst, n_scenarios=5, cv=0.0, distribution="lognormal", seed=i)
            for i, inst in enumerate(instances)
        ]
        # Also cover uniform at CV 0 inside the same identity check.
        cv0_uniform = [
            generate_stochastic_scenarios(inst, n_scenarios=5, cv=0.0, distribution="uniform", seed=i)
            for i, inst in enumerate(instances)
        ]
        for left, right in zip(cv0, cv0_uniform):
            np.testing.assert_array_equal(left.scenarios, right.scenarios)

        nominal = [
            generate_stochastic_scenarios(inst, n_scenarios=1, cv=0.35, distribution="nominal", seed=0)
            for inst in instances
        ]
        for got, inst in zip(cv0 + nominal, list(instances) + list(instances)):
            np.testing.assert_array_equal(got.scenarios, np.broadcast_to(inst[0], got.scenarios.shape))
            np.testing.assert_array_equal(got.machines, inst[1])

        dur_before = env.G_batch.dur.clone()
        edge_before = env.G_batch.edge_index.clone()
        scored_cv0 = evaluate_scenarios(env.G_batch, cv0, alpha=0.9, device="cpu")
        scored_nominal = evaluate_scenarios(env.G_batch, nominal, alpha=0.9, device="cpu")
        torch.testing.assert_close(env.G_batch.dur, dur_before)
        torch.testing.assert_close(env.G_batch.edge_index, edge_before)

        np.testing.assert_allclose(scored_cv0.mean, det, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(scored_nominal.mean, det, rtol=0.0, atol=1e-5)
        # Every scenario is the nominal instance, so each makespan equals the deterministic one.
        repeated = np.repeat(det[:, None], scored_cv0.makespans.shape[1], axis=1)
        np.testing.assert_allclose(scored_cv0.makespans, repeated, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(scored_nominal.makespans[:, 0], det, rtol=0.0, atol=1e-5)
        self.assertTrue(np.all(scored_cv0.mean <= scored_cv0.cvar + 1e-8))

        # One graph at a time, both distributions, against a fresh MassagePassingEval.
        for i, inst in enumerate(instances):
            one = spt_graph([inst], seed=1)
            det_one = deterministic_makespan(one.G_batch, num_nodes_per_example=one.num_nodes_per_example)
            for distribution in ("lognormal", "uniform"):
                scenarios = generate_stochastic_scenarios(
                    inst, n_scenarios=4, cv=0.0, distribution=distribution, seed=0
                )
                scored = evaluate_scenarios(one.G_batch, scenarios, alpha=0.9)
                np.testing.assert_allclose(scored.mean, det_one, rtol=0.0, atol=1e-5)
                self.assertEqual(scored.makespans.shape, (4,))

    def test_mean_does_not_exceed_cvar(self):
        inst = random_instance(4, 4, seed=7)
        env = spt_graph([inst], seed=0)
        base = generate_stochastic_scenarios(inst, n_scenarios=10, cv=0.0, distribution="nominal", seed=0)
        scenarios = base.scenarios.astype(np.float64).copy()
        scenarios[-1] *= 5.0  # one bad scenario; critical-path length scales
        scored = evaluate_scenarios(env.G_batch, scenarios, alpha=0.9)
        self.assertEqual(scored.makespans.shape, (10,))
        self.assertGreater(float(scored.cvar), float(scored.mean))
        self.assertLessEqual(float(scored.mean), float(scored.cvar))
        # alpha=0.9 and 10 scenarios: the worst 10 percent, one scenario.
        ordered = np.sort(scored.makespans)
        self.assertAlmostEqual(float(scored.cvar), float(ordered[-1]))
        self.assertNotAlmostEqual(float(scored.cvar), float(ordered[-2:].mean()))

        noisy = generate_stochastic_scenarios(
            inst, n_scenarios=12, cv=0.3, distribution="lognormal", seed=19
        )
        noisy_score = evaluate_scenarios(env.G_batch, noisy, alpha=0.9)
        self.assertLessEqual(float(noisy_score.mean), float(noisy_score.cvar) + 1e-8)
        self.assertFalse(np.allclose(noisy_score.makespans, noisy_score.makespans[0]))
        self.assertGreater(float(noisy_score.cvar), float(noisy_score.mean))

    def test_equal_scenarios_have_equal_mean_and_cvar(self):
        inst = random_instance(3, 3, seed=8)
        env = spt_graph([inst], seed=2)
        scenarios = generate_stochastic_scenarios(
            inst, n_scenarios=7, cv=0.0, distribution="uniform", seed=1
        )
        scored = evaluate_scenarios(env.G_batch, scenarios, alpha=0.9)
        self.assertAlmostEqual(float(scored.mean), float(scored.cvar))


class CliTests(unittest.TestCase):
    def test_cli_cv0_matches_deterministic_makespan(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            deterministic, result = main(
                ["--n-jobs", "3", "--n-machines", "3", "--n-scenarios", "4", "--cv", "0", "--seed", "2",
                 "--distribution", "lognormal"]
            )
        text = buf.getvalue()
        self.assertIn("deterministic makespan", text)
        self.assertIn("mean makespan", text)
        self.assertAlmostEqual(float(result.mean), float(deterministic), places=5)
        self.assertAlmostEqual(float(result.cvar), float(deterministic), places=5)


if __name__ == "__main__":
    unittest.main()

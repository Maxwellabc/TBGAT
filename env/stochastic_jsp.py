"""Scenario layer for stochastic processing times on a fixed JSSP sequence.

Machine routes stay the ones stored in a Taillard-style instance. Only operation
processing times change. A fixed disjunctive graph (the machine order already
chosen) is scored by the existing message-passing calculator
``MassagePassingEval`` — the class name is the original spelling and is not
renamed here. This module does not change N5, the policy, or training.
"""

import argparse
import math

import numpy as np
import torch
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data

from env.message_passing_evl import MassagePassingEval


DISTRIBUTIONS = ("lognormal", "uniform", "nominal")
# Multiplicative uniform noise on [1 - a, 1 + a] has CV a/sqrt(3).
# a <= 1 keeps every sample non-negative.
UNIFORM_CV_MAX = 1.0 / math.sqrt(3.0)


class StochasticScenarios:
    """Nominal Taillard-style instance plus a batch of processing-time scenarios.

    ``nominal_times`` and ``machines`` have shape ``(n_jobs, n_machines)`` or
    ``(n_instances, n_jobs, n_machines)``. ``scenarios`` inserts a scenario axis
    after the instance axis: ``(n_scenarios, n_jobs, n_machines)`` or
    ``(n_instances, n_scenarios, n_jobs, n_machines)``. Machine ids are copied
    from the nominal instance and are never resampled.
    """

    def __init__(self, nominal_times, machines, scenarios, cv, distribution, seed):
        self.nominal_times = np.asarray(nominal_times)
        self.machines = np.asarray(machines)
        self.scenarios = np.asarray(scenarios)
        self.cv = float(cv)
        self.distribution = distribution
        self.seed = int(seed)

    @property
    def n_scenarios(self):
        if self.scenarios.ndim == 3:
            return int(self.scenarios.shape[0])
        return int(self.scenarios.shape[1])

    def as_instances(self):
        """Stack each scenario as ``(times, machines)`` in the repo's layout.

        Single instance: ``(n_scenarios, 2, n_jobs, n_machines)``.
        Batch: ``(n_instances, n_scenarios, 2, n_jobs, n_machines)``.
        """
        if self.scenarios.ndim == 3:
            machines = np.broadcast_to(self.machines, self.scenarios.shape)
            return np.stack([self.scenarios, machines], axis=1)
        machines = np.broadcast_to(self.machines[:, None, ...], self.scenarios.shape)
        return np.stack([self.scenarios, machines], axis=2)

    def __repr__(self):
        return (
            "StochasticScenarios(distribution={dist!r}, cv={cv}, scenarios={shape}, seed={seed})"
            .format(dist=self.distribution, cv=self.cv, shape=tuple(self.scenarios.shape), seed=self.seed)
        )


class ScenarioEvalResult:
    """Makespan of one fixed sequence on every scenario, plus mean and CVaR.

    For one disjunctive graph, ``makespans`` has shape ``(n_scenarios,)`` and
    ``mean`` / ``cvar`` are NumPy scalars. For a batch of graphs they have a
    leading instance axis.
    """

    def __init__(self, makespans, mean, cvar, alpha):
        self.makespans = np.asarray(makespans, dtype=np.float64)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cvar = np.asarray(cvar, dtype=np.float64)
        self.alpha = float(alpha)

    def __repr__(self):
        return (
            "ScenarioEvalResult(alpha={alpha}, mean={mean}, cvar={cvar}, n_scenarios={n})"
            .format(
                alpha=self.alpha,
                mean=np.array2string(self.mean, precision=4),
                cvar=np.array2string(self.cvar, precision=4),
                n=self.makespans.shape[-1],
            )
        )


def split_taillard_instance(instance):
    """Split a Taillard-style array into processing times and machine routes.

    Accepted shapes:

    * ``(2, n_jobs, n_machines)`` — one instance, index 0 times, index 1 machines
    * ``(n_instances, 2, n_jobs, n_machines)`` — a stack of equal-sized instances

    Machine ids stay 1-based, matching ``env.generateJSP.uni_instance_gen`` and
    the ``.npy`` benchmarks under ``test_data_jssp/``.
    """
    arr = np.asarray(instance)
    if arr.ndim == 3 and arr.shape[0] == 2:
        times, machines = arr[0], arr[1]
    elif arr.ndim == 4 and arr.shape[1] == 2:
        times, machines = arr[:, 0], arr[:, 1]
    else:
        raise ValueError(
            "instance must have shape (2, n_jobs, n_machines) or "
            "(n_instances, 2, n_jobs, n_machines); got {shape}".format(shape=arr.shape)
        )
    if times.shape != machines.shape:
        raise ValueError(
            "processing times and machine routes must have the same shape, "
            "got {t} and {m}".format(t=times.shape, m=machines.shape)
        )
    if times.size == 0:
        raise ValueError("instance has no operations")
    return times, machines


def _normalize_distribution(distribution):
    name = str(distribution).lower().strip()
    if name not in DISTRIBUTIONS:
        raise ValueError(
            "distribution must be one of {opts}; got {name!r}".format(opts=DISTRIBUTIONS, name=distribution)
        )
    return name


def _check_sample_request(nominal, n_scenarios, cv):
    nominal = np.asarray(nominal)
    if nominal.ndim not in (2, 3):
        raise ValueError(
            "nominal_times must have shape (n_jobs, n_machines) or "
            "(n_instances, n_jobs, n_machines); got {shape}".format(shape=nominal.shape)
        )
    if nominal.size == 0:
        raise ValueError("nominal_times is empty")
    n_scenarios = int(n_scenarios)
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be >= 1")
    cv = float(cv)
    if cv < 0.0 or not math.isfinite(cv):
        raise ValueError("cv must be a finite number >= 0")
    return nominal, n_scenarios, cv


def _repeat_nominal(nominal, n_scenarios):
    """Exact copies of ``nominal``. Dtype is preserved so CV=0 stays identical."""
    if nominal.ndim == 2:
        return np.broadcast_to(nominal, (n_scenarios,) + tuple(nominal.shape)).copy()
    return np.broadcast_to(
        nominal[:, None, ...],
        (nominal.shape[0], n_scenarios) + tuple(nominal.shape[1:]),
    ).copy()


def _scenario_shape(nominal, n_scenarios):
    if nominal.ndim == 2:
        return (n_scenarios,) + tuple(nominal.shape)
    return (nominal.shape[0], n_scenarios) + tuple(nominal.shape[1:])


def _align_nominal(nominal):
    """Insert a scenario axis so a batch of nominal matrices broadcasts over samples."""
    if nominal.ndim == 2:
        return nominal
    return nominal[:, None, ...]


def sample_processing_times(nominal_times, n_scenarios, cv, distribution="lognormal", seed=0):
    """Sample processing times around ``nominal_times`` at a given CV.

    ``nominal_times`` is ``(n_jobs, n_machines)`` or
    ``(n_instances, n_jobs, n_machines)``. The sample inserts a scenario axis
    in front of the job axis.

    Distributions:

    * ``nominal`` — every scenario is exactly the nominal matrix. ``cv`` is ignored.
    * ``lognormal`` — independent factors with mean 1 and standard deviation ``cv``,
      so each operation has mean equal to its nominal time and coefficient of
      variation ``cv``. Drawn as ``exp(log(p) - sigma^2/2 + sigma Z)`` with
      ``sigma = sqrt(log(1+cv^2))`` and ``Z`` standard normal.
    * ``uniform`` — multiplicative noise ``p * (1+U)``, ``U ~ Unif[-a, a]``,
      ``a = cv * sqrt(3)``, which has mean ``p`` and coefficient of variation
      ``cv``. Requires ``cv <= 1/sqrt(3)`` so times stay non-negative.

    When ``cv == 0`` every distribution returns exact copies of the nominal
    times (same dtype, no floating-point round trip). ``seed`` is a NumPy
    Generator seed and fully determines the sample for a given nominal matrix,
    shape, distribution, and scenario count. The global ``np.random`` stream
    is not used.
    """
    nominal, n_scenarios, cv = _check_sample_request(nominal_times, n_scenarios, cv)
    distribution = _normalize_distribution(distribution)
    if cv == 0.0 or distribution == "nominal":
        return _repeat_nominal(nominal, n_scenarios)
    if np.any(np.asarray(nominal) <= 0):
        raise ValueError(
            "lognormal and uniform noise need strictly positive nominal processing times"
        )

    nominal_f = nominal.astype(np.float64, copy=False)
    rng = np.random.default_rng(int(seed))
    shape = _scenario_shape(nominal_f, n_scenarios)
    if distribution == "lognormal":
        sigma = math.sqrt(math.log1p(cv * cv))
        mu = np.log(nominal_f) - 0.5 * sigma * sigma
        z = rng.standard_normal(size=shape)
        return np.exp(_align_nominal(mu) + sigma * z)
    half_width = cv * math.sqrt(3.0)
    if half_width > 1.0 + 1e-12:
        raise ValueError(
            "uniform multiplicative noise stays non-negative only for "
            "cv <= 1/sqrt(3) ≈ {limit:.6f}; got cv={cv}. "
            "Use distribution='lognormal' for a larger coefficient of variation.".format(
                limit=UNIFORM_CV_MAX, cv=cv
            )
        )
    noise = rng.uniform(-half_width, half_width, size=shape)
    return _align_nominal(nominal_f) * (1.0 + noise)


def generate_stochastic_scenarios(instance, n_scenarios, cv, distribution="lognormal", seed=0):
    """Wrap one Taillard-style instance, a stack of them, or a list.

    * One ``(2, n_jobs, n_machines)`` array → one :class:`StochasticScenarios`.
    * A stack ``(n_instances, 2, n_jobs, n_machines)`` → one object whose
      ``scenarios`` has shape ``(n_instances, n_scenarios, n_jobs, n_machines)``.
      ``seed`` drives a single draw for the whole stack.
    * A list or tuple of instances (sizes may differ) → a list of objects.
      Instance ``i`` is sampled with ``seed + i``, which matches a direct call
      on that instance alone.

    Machine routes are copied unchanged. ``cv == 0`` and ``distribution='nominal'``
    (including a single nominal scenario) reproduce the deterministic instance.
    """
    if isinstance(instance, (list, tuple)):
        return [
            generate_stochastic_scenarios(
                inst,
                n_scenarios=n_scenarios,
                cv=cv,
                distribution=distribution,
                seed=int(seed) + i,
            )
            for i, inst in enumerate(instance)
        ]

    times, machines = split_taillard_instance(instance)
    distribution = _normalize_distribution(distribution)
    scenarios = sample_processing_times(
        times,
        n_scenarios=n_scenarios,
        cv=cv,
        distribution=distribution,
        seed=seed,
    )
    return StochasticScenarios(
        nominal_times=np.array(times, copy=True),
        machines=np.array(machines, copy=True),
        scenarios=scenarios,
        cv=float(cv),
        distribution=distribution,
        seed=int(seed),
    )


def _tail_start(alpha, n):
    """0-based index of the first ascending order statistic included in CVaR.

    ``CVaR_alpha`` averages ``C_(i*), ..., C_(n)`` with ``i* = ceil(alpha * n)``
    (1-based). ``alpha == 0`` uses every scenario. ``alpha == 1`` uses the worst.
    ``alpha * n`` is snapped to an integer when it is within ``1e-8``, so binary
    rounding of common levels (0.9, 0.95, ...) does not move the cut.
    """
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if n < 1:
        raise ValueError("CVaR needs at least one scenario")
    if alpha <= 0.0:
        return 0
    if alpha >= 1.0:
        return n - 1
    scaled = alpha * n
    nearest = round(scaled)
    if abs(scaled - nearest) <= 1e-8:
        scaled = float(nearest)
    start = int(math.ceil(scaled)) - 1
    if start < 0:
        return 0
    if start > n - 1:
        return n - 1
    return start


def empirical_cvar(samples, alpha):
    """Upper-tail empirical CVaR of makespan at level ``alpha``.

    ``samples`` is ``(n_scenarios,)`` or ``(..., n_scenarios)``. Scenarios are
    sorted ascending on the last axis and averaged from index
    ``ceil(alpha * n) - 1`` through the worst makespan. This is at least the
    mean of the same samples. ``alpha=0.9`` is the usual risk level.
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 0:
        values = values.reshape(1)
    n = int(values.shape[-1])
    start = _tail_start(alpha, n)
    ordered = np.sort(values, axis=-1)
    return ordered[..., start:].mean(axis=-1)


def _graph_list(graph):
    if isinstance(graph, Batch):
        return graph.to_data_list()
    if isinstance(graph, (list, tuple)):
        return list(graph)
    return [graph]


def _num_nodes(data):
    n = data.num_nodes
    if n is None:
        if data.dur is None:
            raise ValueError("disjunctive graph is missing dur and num_nodes")
        n = data.dur.shape[0]
    return int(n)


def _scenario_node_durations(data, scenario_times):
    """Map one instance's scenarios onto node durations, S and T kept at 0.

    ``scenario_times`` may be operation times ``(n_scenarios, n_jobs, n_machines)``
    in job-major order (the order of ``dur[1:-1]``), or a full node vector
    ``(n_scenarios, n_nodes)`` / ``(n_scenarios, n_nodes, 1)``.
    """
    n_nodes = _num_nodes(data)
    st = np.asarray(scenario_times)
    if st.ndim == 3 and st.shape[-1] == 1 and st.shape[1] == n_nodes:
        st = st[..., 0]
    if st.ndim == 2 and st.shape[1] == n_nodes:
        durs = np.array(st, dtype=np.float64, copy=True)
    elif st.ndim == 3 and st.shape[1] * st.shape[2] == n_nodes - 2:
        durs = np.zeros((st.shape[0], n_nodes), dtype=np.float64)
        durs[:, 1:-1] = np.reshape(st, (st.shape[0], -1))
    else:
        raise ValueError(
            "scenario times for a graph with {n} nodes must have shape "
            "(n_scenarios, n_jobs, n_machines) with n_jobs*n_machines={ops}, "
            "or (n_scenarios, {n}); got {shape}".format(
                n=n_nodes, ops=n_nodes - 2, shape=getattr(st, "shape", None)
            )
        )
    if durs.shape[0] < 1:
        raise ValueError("at least one scenario is required")
    if not np.isfinite(durs).all():
        raise ValueError("scenario processing times must be finite")
    return durs


def _unwrap_scenario_input(graphs, scenario_times):
    if isinstance(scenario_times, StochasticScenarios):
        scenario_times = scenario_times.scenarios
    n = len(graphs)
    if isinstance(scenario_times, (list, tuple)):
        if len(scenario_times) != n:
            raise ValueError(
                "expected {n} scenario collections for {n} graphs, got {m}".format(
                    n=n, m=len(scenario_times)
                )
            )
        arrays = []
        for item in scenario_times:
            if isinstance(item, StochasticScenarios):
                arrays.append(np.asarray(item.scenarios))
            else:
                arrays.append(item)
        return arrays
    arr = np.asarray(scenario_times)
    if n == 1 and arr.ndim == 4 and arr.shape[0] == 1:
        return [arr[0]]
    if n > 1 and arr.ndim >= 3 and arr.shape[0] == n:
        return [arr[i] for i in range(n)]
    if n == 1:
        return [arr]
    raise ValueError(
        "for {n} graphs pass a list of scenario arrays or an array whose first "
        "axis has length {n}; got shape {shape}".format(n=n, shape=arr.shape)
    )


def evaluate_scenarios(graph, scenario_times, alpha=0.9, device=None, evaluator=None):
    """Evaluate one fixed machine order on each processing-time scenario.

    ``graph`` is a ``Data`` or a ``Batch`` of disjunctive graphs. Edges are the
    solution (conjunctive job arcs plus the chosen machine arcs). Only ``dur``
    changes across scenarios. Each scenario is placed in its own graph and the
    whole collection is scored by one ``MassagePassingEval.eval`` call — the
    forward/backward message passing is not reimplemented.

    Returns a :class:`ScenarioEvalResult` with per-scenario makespan, the mean,
    and upper-tail CVaR at ``alpha`` (default 0.9). The input graph is not modified.
    """
    graphs = _graph_list(graph)
    if len(graphs) == 0:
        raise ValueError("no disjunctive graph to evaluate")
    arrays = _unwrap_scenario_input(graphs, scenario_times)
    durations = [_scenario_node_durations(g, arr) for g, arr in zip(graphs, arrays)]
    n_scenarios = int(durations[0].shape[0])
    if any(int(block.shape[0]) != n_scenarios for block in durations):
        raise ValueError("every instance in the batch must have the same number of scenarios")

    if device is None:
        device = graphs[0].edge_index.device
    device = torch.device(device)
    if evaluator is None:
        evaluator = MassagePassingEval()

    scenario_graphs = []
    for data, durs in zip(graphs, durations):
        edge_index = data.edge_index.detach().clone().to(dtype=torch.long)
        n_nodes = durs.shape[1]
        for s in range(n_scenarios):
            dur = torch.tensor(durs[s], dtype=torch.float32, device="cpu").view(-1, 1)
            scenario_graphs.append(Data(dur=dur, edge_index=edge_index.clone(), num_nodes=int(n_nodes)))

    batch = Batch.from_data_list(scenario_graphs).to(device)
    _est, _lst, make_span, _count, _fwd, _bwd = evaluator.eval(batch)
    makespans = make_span.detach().cpu().numpy().astype(np.float64, copy=False)
    makespans = np.reshape(makespans, (len(graphs), n_scenarios))
    if len(graphs) == 1:
        makespans = makespans[0]
    mean = makespans.mean(axis=-1)
    cvar = empirical_cvar(makespans, alpha)
    return ScenarioEvalResult(makespans=makespans, mean=mean, cvar=cvar, alpha=alpha)


def _random_instance(n_jobs, n_machines, low, high, seed):
    rng = np.random.default_rng(int(seed))
    if low >= high:
        raise ValueError("low must be < high")
    times = rng.integers(int(low), int(high), size=(int(n_jobs), int(n_machines)))
    machines = np.stack(
        [rng.permutation(np.arange(1, int(n_machines) + 1)) for _ in range(int(n_jobs))]
    )
    return np.stack([times, machines], axis=0)


def run_demo(argv=None):
    """Build an SPT disjunctive graph and score it under processing-time noise.

    Returns ``(deterministic_makespan, ScenarioEvalResult)``. Used by the CLI
    and by tests. Does not train and does not read ``saved_model/``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Score one fixed JSSP sequence under lognormal or uniform processing-time noise. "
            "The sequence is the deterministic SPT initial solution. "
            "Official training and test entry points are not involved."
        )
    )
    parser.add_argument("--n-jobs", type=int, default=5)
    parser.add_argument("--n-machines", type=int, default=5)
    parser.add_argument("--low", type=int, default=1)
    parser.add_argument("--high", type=int, default=99)
    parser.add_argument("--n-scenarios", type=int, default=20)
    parser.add_argument("--cv", type=float, default=0.2)
    parser.add_argument("--distribution", choices=DISTRIBUTIONS, default="lognormal")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", choices=("spt", "fdd-divide-wkr"), default="spt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    import random

    from env.environment import Env

    instance = _random_instance(args.n_jobs, args.n_machines, args.low, args.high, args.seed)
    # SPT / MWKR break ties with the global NumPy stream. Seed it so the
    # deterministic sequence printed by this demo does not depend on caller state.
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = Env()
    env.reset(
        instances=[instance],
        init_sol_type=args.init,
        tabu_size=1,
        device=args.device,
        mask_previous_action=False,
        longest_path_finder="pytorch",
    )
    scenarios = generate_stochastic_scenarios(
        instance,
        n_scenarios=args.n_scenarios,
        cv=args.cv,
        distribution=args.distribution,
        seed=args.seed,
    )
    result = evaluate_scenarios(env.G_batch, scenarios, alpha=args.alpha, device=args.device)
    deterministic = env.current_objs.detach().cpu().numpy().reshape(-1)[0]
    print(
        "size={j}x{m} init={init} distribution={dist} cv={cv} scenarios={s} alpha={a} seed={seed}".format(
            j=args.n_jobs,
            m=args.n_machines,
            init=args.init,
            dist=args.distribution,
            cv=args.cv,
            s=args.n_scenarios,
            a=args.alpha,
            seed=args.seed,
        )
    )
    print("deterministic makespan: {v:.6f}".format(v=float(deterministic)))
    print("mean makespan: {v:.6f}".format(v=float(result.mean)))
    print("CVaR_{a}: {v:.6f}".format(a=args.alpha, v=float(result.cvar)))
    return deterministic, result


def main(argv=None):
    deterministic, result = run_demo(argv)
    return deterministic, result


if __name__ == "__main__":
    main()

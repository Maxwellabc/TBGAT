"""Scenario layer for stochastic processing times on a fixed JSSP sequence.

Machine routes stay the ones stored in a Taillard-style instance. Only operation
processing times change. A fixed disjunctive graph (the machine order already
chosen) is scored by the existing message-passing calculator
``MassagePassingEval`` — the class name is the original spelling and is not
renamed here.

N5 and the policy's critical-path features stay on one nominal CPM pass.
``StochasticRewardConfig`` can opt the training reward into the mean or CVaR
of scenario makespans. ``n_scenarios=0`` (the default) samples nothing.
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


def _tail_mass(alpha, n):
    """Tail length ``w = (1 - alpha) * n`` in units of scenarios.

    ``w`` is snapped to the nearest integer when it is within ``1e-8``, so
    binary rounding of a common level (``(1 - 0.9) * 10`` is
    ``0.9999999999999998`` in float64) does not move an integer cut.
    ``alpha = 0`` returns ``n``. ``alpha = 1`` returns ``0``.
    """
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if n < 1:
        raise ValueError("CVaR needs at least one scenario")
    if alpha <= 0.0:
        return float(n)
    if alpha >= 1.0:
        return 0.0
    w = (1.0 - alpha) * float(n)
    nearest = round(w)
    if abs(w - nearest) <= 1e-8:
        w = float(nearest)
    if w < 0.0:
        return 0.0
    if w > float(n):
        return float(n)
    return float(w)


def empirical_cvar(samples, alpha):
    """Upper-tail empirical CVaR (Expected Shortfall) of makespan.

    ``samples`` is ``(n_scenarios,)`` or ``(..., n_scenarios)``. Larger is
    worse. ``alpha`` is in ``[0, 1]``. Let ``C_(1) <= ... <= C_(n)`` be the
    ascending order statistics and ``w = (1 - alpha) * n`` (see
    :func:`_tail_mass` for the integer snap).

    For ``alpha < 1`` this is the Rockafellar-Uryasev discrete CVaR

        min over zeta of
            zeta + (1 / (n * (1 - alpha))) * sum_i max(C_i - zeta, 0)

    which is also the integral of the empirical quantile function
    ``q(u) = C_(ceil(n * u))``:

        CVaR_alpha = (1 / (1 - alpha)) * integral_alpha^1 q(u) du.

    ``alpha = 1`` is the worst scenario ``C_(n)``. The minimization is not
    used there, because the tail probability is 0.

    Closed form, with ``k = floor(w)`` and ``r = w - k``:

    * If ``w`` is an integer, including ``alpha = 0`` where ``w = n``, CVaR is
      exactly the mean of the worst ``w`` scenarios
      ``C_(n - w + 1), ..., C_(n)``.
    * Otherwise CVaR is ``(sum of the k worst + r * C_(n - k)) / w``.
      Each of those ``k`` scenarios has weight ``1 / w`` and the next order
      statistic has weight ``r / w``. The weights sum to 1, so the probability
      mass is exactly ``w / n = 1 - alpha``.
    * If ``k = 0`` (the tail is thinner than one scenario), CVaR is ``C_(n)``.

    ``alpha = 0.9`` and ``n = 10`` give ``w = 1``, the single worst scenario
    (the worst 10 percent). On the same samples, ``mean <= CVaR_alpha``.
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 0:
        values = values.reshape(1)
    n = int(values.shape[-1])
    w = _tail_mass(alpha, n)
    ordered = np.sort(values, axis=-1)
    if w <= 0.0:
        return ordered[..., -1]
    if w >= n:
        return ordered.mean(axis=-1)
    k = int(math.floor(w))
    r = w - float(k)
    if k <= 0:
        return ordered[..., -1]
    tail = ordered[..., n - k:]
    if r == 0.0:
        return tail.mean(axis=-1)
    boundary = ordered[..., n - k - 1]
    return (tail.sum(axis=-1) + r * boundary) / w


def scenario_objective(result, cvar_weight=0.0):
    """Convex combination ``(1 - w) * mean + w * CVaR`` of a scenario score.

    ``w = 0`` is the expected makespan. ``w = 1`` is CVaR. ``result`` is a
    :class:`ScenarioEvalResult` or a ``(mean, cvar)`` pair. Lower is better.
    """
    w = float(cvar_weight)
    if not math.isfinite(w) or w < 0.0 or w > 1.0:
        raise ValueError("cvar_weight must be in [0, 1]; got {w}".format(w=cvar_weight))
    if isinstance(result, ScenarioEvalResult):
        mean, cvar = result.mean, result.cvar
    else:
        mean, cvar = result
    mean = np.asarray(mean, dtype=np.float64)
    cvar = np.asarray(cvar, dtype=np.float64)
    return (1.0 - w) * mean + w * cvar


def resolve_cvar_weight(reward_objective="mean", cvar_weight=None):
    """Map the reward flag onto the convex-combination weight.

    ``mean`` -> 0, ``cvar`` -> 1. ``mix`` uses ``cvar_weight`` in ``[0, 1]``.
    Passing a non-default weight together with ``mean`` or ``cvar`` is an
    error so the flag and the weight cannot silently disagree.
    """
    name = str(reward_objective).lower().strip()
    if name not in ("mean", "cvar", "mix"):
        raise ValueError(
            "reward_objective must be 'mean', 'cvar', or 'mix'; got {name!r}".format(name=reward_objective)
        )
    if name == "mean":
        if cvar_weight not in (None, 0, 0.0):
            raise ValueError(
                "reward_objective='mean' trains on expected makespan. "
                "Use reward_objective='mix' with cvar_weight in [0, 1] for a convex combination."
            )
        return 0.0
    if name == "cvar":
        if cvar_weight not in (None, 1, 1.0):
            raise ValueError(
                "reward_objective='cvar' trains on CVaR. "
                "Use reward_objective='mix' with cvar_weight in [0, 1] for a convex combination."
            )
        return 1.0
    if cvar_weight is None:
        raise ValueError("reward_objective='mix' requires cvar_weight in [0, 1]")
    w = float(cvar_weight)
    if not math.isfinite(w) or w < 0.0 or w > 1.0:
        raise ValueError("cvar_weight must be in [0, 1]; got {w}".format(w=cvar_weight))
    return w


class StochasticRewardConfig:
    """Optional scenario objective for the training reward.

    ``n_scenarios=0`` (the default) leaves the reward on the nominal makespan
    and does not draw scenarios. Otherwise one batch of processing times is
    drawn in ``Env.reset`` and every later machine order is rescored on that
    same batch. The scalar is

        (1 - cvar_weight) * mean(makespan) + cvar_weight * CVaR_alpha(makespan).

    ``cvar_weight=0`` trains on the mean. ``cvar_weight=1`` trains on CVaR.
    N5 and the actor's critical-path features are not read from this config.
    """

    def __init__(self, n_scenarios=0, cv=0.0, distribution="lognormal", alpha=0.9, cvar_weight=0.0, seed=0):
        self.n_scenarios = int(n_scenarios)
        if self.n_scenarios < 0:
            raise ValueError("n_scenarios must be >= 0 (0 disables scenario scoring)")
        self.cv = float(cv)
        self.distribution = str(distribution).lower().strip()
        self.alpha = float(alpha)
        if not math.isfinite(self.alpha) or self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.cvar_weight = float(cvar_weight)
        if not math.isfinite(self.cvar_weight) or self.cvar_weight < 0.0 or self.cvar_weight > 1.0:
            raise ValueError("cvar_weight must be in [0, 1]")
        self.seed = int(seed)
        if self.n_scenarios > 0:
            self.distribution = _normalize_distribution(self.distribution)
            if not math.isfinite(self.cv) or self.cv < 0.0:
                raise ValueError("cv must be a finite number >= 0")
        if self.cvar_weight <= 0.0:
            self.reward_objective = "mean"
        elif self.cvar_weight >= 1.0:
            self.reward_objective = "cvar"
        else:
            self.reward_objective = "mix"

    @property
    def enabled(self):
        return self.n_scenarios > 0

    @classmethod
    def from_flags(
        cls,
        n_scenarios=0,
        cv=0.0,
        distribution="lognormal",
        alpha=0.9,
        reward_objective="mean",
        cvar_weight=None,
        seed=0,
    ):
        """Build a config from the training CLI flags."""
        weight = resolve_cvar_weight(reward_objective, cvar_weight)
        return cls(
            n_scenarios=n_scenarios,
            cv=cv,
            distribution=distribution,
            alpha=alpha,
            cvar_weight=weight,
            seed=seed,
        )

    def __repr__(self):
        return (
            "StochasticRewardConfig(n_scenarios={n}, cv={cv}, distribution={dist!r}, "
            "alpha={alpha}, cvar_weight={w}, seed={seed}, enabled={enabled})"
            .format(
                n=self.n_scenarios,
                cv=self.cv,
                dist=self.distribution,
                alpha=self.alpha,
                w=self.cvar_weight,
                seed=self.seed,
                enabled=self.enabled,
            )
        )


def _graph_list(graph):
    """One ``Data`` per disjunctive graph.

    A ``Batch`` is split by node pointer. ``Batch.to_data_list`` reuses the
    ``edge_index`` slices saved when the batch was built. ``Env.step`` replaces
    ``edge_index`` with a new concatenation those slices do not describe, so
    edges are grouped by their endpoints instead. Makespan does not depend on
    edge order.
    """
    if isinstance(graph, (list, tuple)):
        return list(graph)
    if isinstance(graph, Batch) and getattr(graph, "ptr", None) is not None and graph.ptr.numel() > 1:
        edge_index = graph.edge_index
        graphs = []
        ptr = graph.ptr
        for i in range(int(ptr.numel()) - 1):
            left = int(ptr[i].item())
            right = int(ptr[i + 1].item())
            src = edge_index[0]
            dst = edge_index[1]
            mask = (src >= left) & (src < right) & (dst >= left) & (dst < right)
            graphs.append(
                Data(
                    dur=graph.dur[left:right].detach().clone(),
                    edge_index=(edge_index[:, mask] - left).detach().clone(),
                    num_nodes=right - left,
                )
            )
        return graphs
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
    and upper-tail CVaR at ``alpha`` (default 0.9). CVaR is the discrete
    Expected Shortfall in :func:`empirical_cvar`. The input graph is not modified.
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

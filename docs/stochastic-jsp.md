# Stochastic JSP: scenario evaluation (first increment)

This note describes the first step from the deterministic TBGAT / L2S improvement heuristic (UAI 2024) toward job-shop instances whose **processing times are random** and whose **machine routes are not**. It is an evaluation layer only. It does not try to close deterministic Taillard gaps, and it does not change training.

## What stays deterministic

`n-step_reinforce.py` and `test_jssp.py` are unchanged. Checkpoints under `saved_model/` are unchanged. `Env` still builds one disjunctive graph, runs one critical-path pass, and proposes N5 moves from that pass.

N5 today reads a single CPM: node durations `p`, earliest start `est`, and latest start `lst` from one call to `MassagePassingEval` (the class name is the original spelling and is left as-is). A stochastic neighbourhood would need a scenario-wise or robust critical path. **This increment does not add that.** It only rescores a **complete** machine order.

The new code lives in `env/stochastic_jsp.py` rather than `env/generateJSP.py`. The existing generator imports the training argument parser at import time; coupling noise sampling to that parser would risk changing the deterministic CLI.

## Processing-time scenarios

A Taillard-style instance in this repo is an array of shape `(2, n_jobs, n_machines)` (or a stack `(n_instances, 2, n_jobs, n_machines)`):

- index `0`: nominal processing times
- index `1`: 1-based machine route of each job

That is the layout of `uni_instance_gen` and of `test_data_jssp/*.npy`. Routes are copied and never resampled. Scenario `s` replaces only the time matrix.

`generate_stochastic_scenarios(instance, n_scenarios, cv, distribution, seed)` draws a batch of scenarios from one integer seed (`numpy.random.Generator`, not the global `np.random` stream).

| `distribution` | Sample |
| --- | --- |
| `nominal` | Every scenario is the nominal matrix. `cv` is ignored. One scenario is the deterministic instance. |
| `lognormal` | Independent factors with mean 1 and standard deviation `cv`. Operation time has mean equal to the nominal time and coefficient of variation `cv`: `exp(log(p) - sigma^2/2 + sigma Z)`, `sigma = sqrt(log(1+cv^2))`. |
| `uniform` | `p * (1+U)` with `U ~ Unif[-a, a]`, `a = cv * sqrt(3)`. Mean is `p` and the coefficient of variation is `cv`. Allowed only for `cv <= 1/sqrt(3)` so times stay non-negative. |

`cv == 0` short-circuits every distribution and returns **exact** copies of the nominal times (same dtype, no `exp`/`log` round trip). A single `nominal` scenario has the same processing times and the same machine routes as the deterministic instance.

A list of instances may mix sizes. Instance `i` uses seed `seed + i`. An equal-sized stack uses one seed for one draw of shape `(n_instances, n_scenarios, n_jobs, n_machines)`.

Noise is continuous. The message-passing makespan is therefore real-valued when `cv > 0`. Integer nominal data is returned unchanged when `cv == 0` or `distribution='nominal'`.

## Scenario evaluator

`evaluate_scenarios(graph, scenario_times, alpha=0.9)` keeps the disjunctive graph's edges (the chosen operation order on each machine, plus the job precedence arcs) and swaps durations.

Node `0` is the dummy source and the last node is the dummy sink, both with duration 0. Operation times are written in job-major order into `dur[1:-1]`, which is how `Env._init_solver` lays out `dur`. You can also pass a full node-duration vector of length `n_jobs * n_machines + 2`.

Each scenario becomes its own `Data` graph. One `Batch` is handed to `MassagePassingEval.eval`. The forward and backward passes in `env/message_passing_evl.py` are not modified. The input graph is not modified.

The result carries:

- per-scenario makespan
- mean makespan
- upper-tail empirical CVaR of makespan (default level `0.9`)

Lower makespan is better, so CVaR is the **upper** tail, not a lower tail of reward. Sort scenario makespans `C_(1) <= ... <= C_(n)`. Let `i* = ceil(alpha * n)` (1-based). `CVaR_alpha` is the average of `C_(i*), ..., C_(n)`. `alpha = 0` is the mean of every scenario. `alpha = 1` is the worst scenario. For the same samples, `mean <= CVaR_alpha`. On 10 scenarios at `alpha = 0.9` this is the mean of the two worst makespans.

`graph` may be one `Data` or a `Batch` from `Env.reset` (several instances of different sizes). Pass a list of per-instance scenario arrays, or one `StochasticScenarios` object.

## Out of scope

- Scenario-dependent N5, critical blocks, or est/lst features
- A policy or reward that optimizes mean or CVaR makespan
- Hooking this evaluator into `n-step_reinforce.py` or `test_jssp.py`
- Retraining, editing `saved_model/`, or storing generated scenario tensors

## How to run

From the repository root, with the same PyTorch / PyG stack the deterministic code already imports:

```bash
python -m unittest tests.test_stochastic_jsp -v
```

The tests check three things: `cv = 0` and a single nominal scenario match `MassagePassingEval` on small random instances (and on `ft06`); mean makespan is at most CVaR on the same scenarios; the same seed repeats the same processing times. They do not load checkpoints.

A small demo scores an SPT sequence and prints the deterministic makespan, the scenario mean, and CVaR. It does not train:

```bash
python -m env.stochastic_jsp --n-jobs 4 --n-machines 4 --n-scenarios 16 --cv 0.2 --distribution lognormal --alpha 0.9 --seed 0
```

`--cv 0` (or `--distribution nominal`) prints a mean that matches the deterministic makespan of that same SPT graph.

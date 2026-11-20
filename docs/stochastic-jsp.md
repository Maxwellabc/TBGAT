# Stochastic JSP: scenarios and an optional mean / CVaR reward

This note describes the step from the deterministic TBGAT / L2S improvement heuristic (UAI 2024) toward job-shop instances whose **processing times are random** and whose **machine routes are not**. Machine routes stay fixed. A complete machine order can be rescored on a batch of processing-time scenarios. N5 and the policy's critical-path features stay on one nominal critical-path pass.

It does not try to close deterministic Taillard gaps, and it does not retrain the checkpoints under `saved_model/`.

## What stays deterministic

With the default flags (`--n_scenarios 0`), `n-step_reinforce.py`, `test_jssp.py`, and `test_fssp.py` behave as before. Checkpoints under `saved_model/` are not modified. The `algo_config` string that names those checkpoints is unchanged unless `--n_scenarios` is positive, so a stochastic run cannot overwrite a deterministic checkpoint.

`Env` still builds one disjunctive graph, runs one critical-path pass on the nominal durations, and proposes N5 moves from that pass. The actor still reads that pass: duration, earliest start, latest start, and the two topological features. `move_selector_ts_inner` is not on the training forward path.

N5 today reads a single CPM: node durations `p`, earliest start `est`, and latest start `lst` from one call to `MassagePassingEval` (the class name is the original spelling and is left as-is). A stochastic neighbourhood would need a scenario-wise or robust critical path. **This increment does not add that.** It only rescores a **complete** machine order, and only for the training reward.

Validation gaps stay on the nominal makespan against the deterministic CP-SAT reference. The validation environment is not given a scenario batch. Its reward is not used for the policy update.

The scenario code lives in `env/stochastic_jsp.py` rather than `env/generateJSP.py`. The existing generator imports the training argument parser at import time; coupling noise sampling to that parser would risk changing the deterministic CLI. The environment imports the scenario helpers, but it draws scenarios only when a `StochasticRewardConfig` has `n_scenarios > 0`.

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

A `Batch` is split by node pointer, and edges are grouped by their endpoints. `Env.step` replaces `edge_index` with a concatenation that the slices saved at `Batch` creation do not describe, so `Batch.to_data_list` is not used for this split. Makespan does not depend on edge order.

The result carries:

- per-scenario makespan
- mean makespan
- upper-tail empirical CVaR of makespan (default level `0.9`)

Lower makespan is better, so CVaR is the **upper** tail, not a lower tail of reward.

### Empirical CVaR

`empirical_cvar` is the Rockafellar-Uryasev discrete CVaR, also called empirical Expected Shortfall. Let `C_(1) <= ... <= C_(n)` be the ascending scenario makespans and let `w = (1 - alpha) * n`. If `w` is within `1e-8` of an integer it is snapped to that integer, so binary rounding of a common level does not move the cut (`(1 - 0.9) * 10` is `0.9999999999999998` in float64).

For `alpha < 1` the value is

```
min over zeta of  zeta + (1 / (n * (1 - alpha))) * sum_i max(C_i - zeta, 0)
```

which is the same number as the integral of the empirical quantile `q(u) = C_(ceil(n * u))`:

```
CVaR_alpha = (1 / (1 - alpha)) * integral from alpha to 1 of q(u) du
```

`alpha = 1` is the worst scenario `C_(n)`. The minimization is not used there, because the tail probability is 0.

Closed form, with `k = floor(w)` and `r = w - k`:

- When `w` is an integer, including `alpha = 0` where `w = n`, CVaR is exactly the mean of the worst `w` scenarios `C_(n - w + 1), ..., C_(n)`.
- Otherwise CVaR is `(sum of the k worst + r * C_(n - k)) / w`. Each of those `k` scenarios has weight `1 / w` and the next order statistic has weight `r / w`. The weights sum to 1, so the probability mass is exactly `w / n = 1 - alpha`.
- When `k = 0` (the tail is thinner than one scenario), CVaR is the worst scenario.

`alpha = 0` is the mean of every scenario. For the same samples, `mean <= CVaR_alpha`. For `alpha = 0.9` and `n = 10`, `w = 1`, so CVaR is the single worst makespan (the worst 10 percent), not the mean of the two worst.

`graph` may be one `Data` or a `Batch` from `Env.reset` (several instances of different sizes). Pass a list of per-instance scenario arrays, or one `StochasticScenarios` object.

## Training reward

The primary objective is expected makespan. CVaR is a secondary objective, either alone or as a convex combination. The scalar is

```
J = (1 - w) * mean(scenario makespans) + w * CVaR_alpha(scenario makespans)
```

`w = 0` is the mean. `w = 1` is CVaR. `Env.step` still returns an improvement reward, `max(J_incumbent - J, 0)`, the same shape as the deterministic reward `max(C_incumbent - C, 0)`. REINFORCE maximizes that reward. `J_incumbent` is the best scenario objective seen on this episode and is stored separately from the nominal incumbent makespan. `current_objs` and the actor features stay nominal, including when `w > 0`.

Scenarios are drawn once in `reset` and reused after every N5 move. The machine order changes. The processing-time batch does not. The training environment passes the batch as a list, so instance `i` is drawn with seed `training_seed + i` through a private Generator. Dispatching-rule tie breaks keep the global NumPy stream.

Flags on `n-step_reinforce.py` (parsed in `parameters.py`):

| Flag | Default | Role |
| --- | --- | --- |
| `--n_scenarios` | `0` | Scenario count. `0` does not sample and does not call the scenario evaluator. |
| `--cv` | `0` | Coefficient of variation. Used only when `--n_scenarios > 0`. |
| `--distribution` | `lognormal` | `lognormal`, `uniform`, or `nominal`. |
| `--alpha` | `0.9` | CVaR level. |
| `--reward_objective` | `mean` | `mean`, `cvar`, or `mix`. |
| `--cvar_weight` | unset | Weight `w` for `--reward_objective mix`. Do not pass it with `mean` or `cvar`. |

`n_scenarios = 0` wins over the objective flag: `--reward_objective cvar` with the default scenario count is still a deterministic run. The first stochastic experiments should pass `--n_scenarios` and leave `--reward_objective mean`.

`n-step_reinforce.py` attaches the config only to the training environment. The logged batch performance and the validation gap remain the nominal makespan. When scoring is on, the progress bar also shows `StochObj`, the mean of `J` over the batch, and the checkpoint name grows a `-stoch-...` suffix.

## Out of scope

- Scenario-dependent N5, critical blocks, or est/lst features
- Changing the actor to read a scenario-wise or robust critical path
- Selecting validation checkpoints by the stochastic objective
- Retraining, or editing files under `saved_model/`

## How to run

From the repository root, with the same PyTorch / PyG stack the deterministic code already imports:

```bash
python -m unittest tests.test_stochastic_jsp tests.test_stochastic_reward -v
python test_jssp.py --help
python n-step_reinforce.py --help
python test_fssp.py --help
```

The CVaR tests check the locked tail: `alpha = 0.9` and `n = 10` is the worst 10 percent; the mean is at most CVaR on the same samples; `alpha = 0` is the mean; `alpha = 1` is the worst scenario; a seed repeats. The reward tests check that `cv = 0` and a nominal scenario match `MassagePassingEval`, that mean versus CVaR changes the scalar, and that a default step does not evaluate scenarios. They do not load checkpoints and they do not train.

A small demo scores an SPT sequence and prints the deterministic makespan, the scenario mean, and CVaR. It does not train:

```bash
python -m env.stochastic_jsp --n-jobs 4 --n-machines 4 --n-scenarios 16 --cv 0.2 --distribution lognormal --alpha 0.9 --seed 0
```

`--cv 0` (or `--distribution nominal`) prints a mean and a CVaR that match the deterministic makespan of that same SPT graph.

To train on expected makespan, add the scenario flags to the usual training command and leave the objective at its default. This repository's checked-in run is not a long training job:

```bash
python n-step_reinforce.py --n_scenarios 20 --cv 0.2 --distribution lognormal --alpha 0.9 --reward_objective mean
```

`--reward_objective cvar` trains on CVaR at `--alpha`. `--reward_objective mix --cvar_weight 0.25` trains on `0.75 * mean + 0.25 * CVaR`.

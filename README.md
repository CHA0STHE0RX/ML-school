# ML-school

A benchmark framework for grading reinforcement-learning agents from different families on identical, unbiased capability exams in shared Gymnasium environments.

For project vision, architecture, and design rationale, see [docs/DESIGN.md](docs/DESIGN.md).

---

## Install

```bash
python -m venv ML-school
ML-school/Scripts/activate          # Windows; or `source ML-school/bin/activate` on Unix
pip install -r requirements.txt
pip install nvidia-ml-py>=13.595    # optional, for GPU power-draw measurement
```

Requires Python 3.10+. For GPU runs, install the CUDA-matching torch wheel instead of the default CPU one:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## Run

```bash
# Smoke test — 1 agent, 1 env, 5k training steps (~2 min on CPU)
python run_experiments.py --agents PPO --envs Pendulum-v1 --timesteps 5000

# Full v1 run — uses the ROSTER constant at the top of run_experiments.py
python run_experiments.py

# Reuse saved weights, re-run exams only
python run_experiments.py --skip-training --force-eval

# Preview what would run, without executing
python run_experiments.py --dry-run

# Generate plots from results.jsonl — one script per research question
python plot_resilience.py --tag <tag>     # resilience signature: curves + leaderboard + breaking-point
python plot_progression.py                # progression across tagged budgets: triptych + bundle
```

For the canonical PPO/Pendulum progression dataset:

```powershell
./reproduce_ppo_pendulum_progression.ps1    # ~2.5 hrs CPU — 6 budgets x 4 seeds
```

Outputs:

| Path | Contents |
|---|---|
| `results/results.jsonl` | One row per (agent × env × seed × exam × mod-type). |
| `results/run_log.jsonl` | Completed cells — used for resumption after a crash. |
| `results/errors.log` | Tracebacks of failed cells. |
| `results/status.txt` | Single-line progress indicator for the current cell. |
| `results/plots/` | PNG figures: per-mod curves, leaderboard, breaking-point scatter. |
| `weights/<agent>/<env>/seed<N>/` | `model.zip` + `train_meta.json` — checkpoints, reused by `--skip-training`. |

`results/` and `weights/` are gitignored.

---

## Architecture

```
.
├── agents/
│   ├── base.py            # Policy, StatelessPolicy, TrainResult, AgentProtocol
│   ├── ppo_agent.py       # SB3 PPO wrapped behind the protocol (device-forced CPU)
│   └── esn_agent.py       # Echo State Network (reservoirpy) + CMA-ES on the readout
├── exams/
│   ├── base.py            # Exam protocol
│   └── resilience.py      # ResilienceExam — wraps EnvironmentCritic
├── tests/                 # 33 pytest tests (instrument calibration)
├── docs/DESIGN.md         # design rationale and bug-fix audit
├── critic.py              # EnvironmentCritic — bisection probe + logistic fit
├── wrappers.py            # FLICKER / GAUSSIAN_NOISE / ACTION_DELAY / PHYSICS_SHIFT
├── records.py             # ExperimentRecord schema (the row written to JSONL)
├── task_metrics.py        # Per-env creative metrics (Pendulum: smoothness/energy/upright)
├── plots.py               # Utility module: loaders + drawing primitives (no main)
├── plot_resilience.py     # One research question: agent's resilience signature
├── run_experiments.py     # Orchestrator (cartesian product, resumability)
├── requirements.txt
└── README.md
```

Every agent satisfies one protocol: `observation → action`, plus `reset()` for episode-stateful policies. Every exam emits `ExperimentRecord` rows into JSONL. The orchestrator runs the cartesian product of agents × envs × seeds × exams, saves checkpoints, and resumes after interruption via `results/run_log.jsonl`.

---

## Reading a record

Each row in `results/results.jsonl` is one JSON object. Key fields:

- `config.{agent_name, env_id, train_seed, total_timesteps, hyperparams}` — what we set out to do.
- `hardware.{cpu, gpu, ram_gb, precision, backend, energy_efficiency_w}` — where it ran (`backend ∈ {cuda, cpu, mps, rocm, xla}`).
- `train_{time_sec, env_steps, opt_steps}`, `param_count` — training cost.
- `inf_{lat_ms, macs, mem_mb, gpu_mem_mb}` — inference cost.
- `clean_return`, `clean_return_std` — unperturbed performance, in raw env reward units.
- `success ∈ [0, 1]`, `adapt_score ∈ [0, 1] | None` — universal grades. Formulas declared in `exam.formula`.
- `exam.{name, config, raw, formula}` — capability-test-specific block. For ResilienceExam, `exam.raw` carries `{aurc, s_half, s_max, cliff_slope, clean_return, points[]}`.
- `env_metrics.*` — per-env creative metrics.
- `diagnostics.*` — agent-specific extras.

The split between universal grades (`success`, `adapt_score`) and exam-specific raw (`exam.raw`) is what lets you produce cross-exam plots without losing per-exam fidelity. The `exam.formula` string is the receipt — every record carries its own derivation.

---

## Resilience methodology (one-paragraph version)

For each perturbation type, an `EnvironmentCritic` probes the agent across a strength axis: clean (s=0), worst (s=s_max), then bisects toward the 50% crossing for `max_iters` iterations. It fits a logistic `f(s) = 1/(1 + exp(k(s - s_half)))` to the (strength, normalized-return) points using `scipy.optimize.curve_fit`, integrates the fitted curve on a 200-point grid to get **AURC ∈ [0, 1]** (the area under the recovery curve), and reports `s_half` (where the agent loses half its performance) and `cliff_slope = k` (how sharp the failure is). Fitting-then-integrating, rather than trapezoidal integration of raw points, makes AURC insensitive to bisection sampling details. See [docs/DESIGN.md](docs/DESIGN.md) for the design rationale.

---

## Tests

```bash
pytest tests/
```

33 tests, ~20 seconds. They enforce instrument-calibration invariants: bit-identity of wrappers at strength=0, stateful-policy isolation between episodes, schema round-trip through JSONL, logistic fit recovers known parameters from synthetic data, etc. Test names describe the invariant being protected (e.g. `test_stateful_leak_isolated_by_reset`, `test_flicker_identity_at_zero`).

---

## Status

**v1: PPO + ESN + Pendulum-v1 + Resilience exam + progression-curve analysis, end-to-end working.**

End-to-end means: orchestrator trains the agent, persists weights + train_meta, measures inference latency/MACs/memory, runs the resilience critic across all 4 perturbation types, writes one record per (agent, env, seed, mod) to `results.jsonl`. `plot_resilience.py` produces the per-mod curves + leaderboard + breaking-point. `plot_progression.py` produces the 4-row triptych (clean | AURC | absolute_aurc | hardware) and the 2×2 bundle (adds the resilience↔clean_return coupling panel).

PPO uses SB3 with a forced `device="cpu"` (tiny-MLP on Pendulum is faster on CPU than CUDA due to kernel-launch overhead, even with the CUDA torch wheel installed). ESN uses reservoirpy for the fixed reservoir and CMA-ES on the linear readout.

SAC, LNN, SNN, NEAT agents and additional capability exams (sample efficiency, generalization, adaptation, stability) are scoped for follow-on plans.

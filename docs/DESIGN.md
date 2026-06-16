# ML-school — Design Spec

**Project:** `folder1/` (to be renamed `ML-school` once the v1 pipeline runs cleanly)
**Status:** Approved design, pre-implementation

---

## 1. Vision

ML-school is a benchmark framework for grading reinforcement-learning agents from very different families (gradient-based RL, reservoir computing, spiking networks, neuroevolution, liquid networks) on **identical, unbiased exams** in shared environments. The framing is a "school of AI" — slightly dystopian — where the environment is the impartial jury, the agents are the students, and the only legitimate way to compare them is by the verdicts of capability-specific exams.

### Core principles

1. **The agent is a black box.** It satisfies one protocol: `observation -> action`. The framework never introspects internals. The agent self-reports its own intrinsic properties (param count, inference MACs).
2. **The environment is the jury.** Success is what the environment-bound exam declares it to be, not what the agent claims.
3. **Exams are decoupled from the record format.** A new capability test (sample efficiency, generalization, adaptation) can be added without changing `ExperimentRecord`.
4. **Raw measurements are frozen; semantic interpretations are mutable.** Store every underlying number so success formulas can be revised later without re-running.
5. **Local-machine independence.** Every path is project-relative. Every experiment is reproducible from a fresh clone via `python run_experiments.py`.

---

## 2. Capabilities the school grades

| # | Capability | Question | Test mechanism | v1? |
|---|---|---|---|---|
| 1 | **Resilience** | Does the agent hold up when the world degrades? | `EnvironmentCritic` — bisection probe per `ModType`, logistic fit yields `aurc`, `s_half`, `cliff_slope`. | ✅ |
| 2 | **Sample efficiency** | How fast does it learn? | Evaluate at training checkpoints; steps-to-reach-X-return. | later |
| 3 | **Generalization** | Does training on seed=0 transfer to seeds 100..120? | Train one seed, eval many; report the gap. | later |
| 4 | **Adaptation** | Can it cope with mid-episode distribution shift? | Custom envs (The Drifting World) with online physics changes. | later |
| 5 | **Stability** | Are results reproducible across seeds? | Run N seeds, report variance of `clean_return`. | possible v1 |
| 6 | **Inference efficiency** | How expensive is one decision? | `inf_latency_ms` (extrinsic, orchestrator-measured) and `inference_macs()` (intrinsic, agent-reported). Universal per record, not a separate exam. | ✅ baked in |
| 7 | **Compositionality / transfer** | Train on env A, deploy on env B? | Cross-env evaluation; requires shared obs/action shapes. | far future |

Only **Resilience** is implemented as a dedicated exam in v1. The architecture supports the others as drop-in additions.

---

## 3. File structure

```
folder1/  (later: ML-school/)
├── agents/
│   ├── __init__.py
│   ├── base.py            # Policy protocol, AgentProtocol, TrainResult dataclass
│   ├── ppo_agent.py       # SB3 PPO wrapper
│   ├── sac_agent.py       # SB3 SAC wrapper
│   ├── esn_agent.py       # ReservoirPy ESN wrapper
│   ├── lnn_agent.py       # ncps / nn.LSTMCell wrapper
│   ├── neat_agent.py      # neat-python wrapper
│   └── snn_agent.py       # snnTorch wrapper (rate coding)
├── envs/
│   └── __init__.py        # placeholder for future custom envs
├── exams/
│   ├── __init__.py
│   ├── base.py            # Exam protocol
│   └── resilience.py      # ResilienceExam wrapping EnvironmentCritic
├── weights/               # checkpoints, gitignored
│   └── <agent>/<env>/seed<N>/{model.zip, train_meta.json}
├── results/               # outputs, gitignored
│   ├── results.jsonl
│   ├── run_log.jsonl
│   ├── errors.log
│   ├── status.txt
│   └── plots/
├── docs/
│   └── DESIGN.md          # this file
├── critic.py              # EnvironmentCritic (used by ResilienceExam)
├── records.py             # ExperimentRecord schema (exam-agnostic)
├── wrappers.py            # perturbation wrappers
├── plots.py               # drawing primitives + record loaders (no __main__)
├── plot_resilience.py     # one research question per file: composes plots.py
├── run_experiments.py     # orchestrator
├── task_metrics.py        # per-env creative metrics (action_smoothness, energy, ...)
├── requirements.txt
├── .gitignore             # ignores results/, weights/, __pycache__/, *.pyc
├── LICENSE                # Apache-2.0
└── README.md              # concise: how to use, sources, architecture
```

---

## 4. Agent interface

### 4.1 The Policy protocol — episode-state safety

Every agent's policy is an object, not a bare callable, so stateful agents (ESN, LNN, SNN) can reset their internal state at episode boundaries.

```python
class Policy(Protocol):
    def reset(self) -> None: ...                          # clears stateful internals
    def __call__(self, obs: np.ndarray) -> np.ndarray: ... # one decision
```

- **PPO/SAC**: `reset()` is a no-op (MLPs are stateless).
- **ESN**: `reset()` zeros the reservoir activation.
- **LNN**: `reset()` zeros LSTM hidden + cell states.
- **SNN**: `reset()` zeros membrane potentials and spike traces.
- **NEAT**: no-op for feedforward genomes; clears recurrent state otherwise.

`critic.py`'s eval loop pairs every `env.reset()` with `policy.reset()`. This pairing is the discipline that prevents the stateful memory leak. A regression test confirms episode-N returns are independent of episode-(N-1) for stateful agents on identical seeds.

### 4.2 TrainResult — frozen training metrics

```python
@dataclass
class TrainResult:
    train_time_sec : float
    train_env_steps: int   # actual env interactions used
    train_opt_steps: int   # gradient steps / generations / ridge solves
    diagnostics    : dict  # agent-specific (loss curves, NEAT gen count, ESN spectral radius)
```

### 4.3 AgentProtocol

```python
class AgentProtocol(Protocol):
    def train(self, env_fn, total_timesteps: int, seed: int) -> TrainResult: ...
    def save(self, path: Path) -> None: ...           # writes model + train_meta.json
    def load(self, path: Path) -> TrainResult: ...    # returns persisted TrainResult
    def policy(self) -> Policy: ...
    def param_count(self) -> int: ...
    def inference_macs(self) -> int | None: ...       # agent self-reports; None if N/A
```

**`save()` always writes `train_meta.json` alongside the model file.** `load()` reads it back and returns the original `TrainResult`. This means `--skip-training` produces records with the *original* training metrics, not zeros. If `train_meta.json` is missing on load, the agent raises rather than silently producing a corrupt record.

**Responsibility split:**

| Intrinsic — agent-reported | Extrinsic — orchestrator-measured |
|---|---|
| `param_count` (see definition below) | `train_time_sec` (wall clock during the *original* training) |
| `inference_macs` | `inf_lat_ms` (wall clock during *this* eval run) |
| `train_env_steps` | `inf_mem_mb` (peak RSS delta during the eval forward passes) |
| `train_opt_steps` | `inf_gpu_mem_mb` (peak CUDA allocation during the eval forward passes; `None` if no CUDA) |
| `diagnostics` | |

**`param_count` definition (must be agreed across agents for fair comparison):** the number of *trainable* parameters that participate in a single inference forward pass on a fully trained model. After any pruning / quantization / freezing.

- PPO/SAC: parameter count of the deployed policy network only (not the value head if it's discarded at deploy).
- ESN: readout weight count only — the random reservoir matrix is **not** counted (it's fixed, not trained).
- NEAT: count of active enabled connections + biases in the winning genome.
- LNN/SNN: standard `numel()` over the deployed module.

**SNN-specific MAC accounting.** Rate-coded SNNs run T internal time-steps per *one* environment step (e.g. T=25 spike steps integrate into one rate-coded output). `fvcore` measures a single spatial forward pass and is unaware of T. The SNN agent's `inference_macs()` MUST therefore return `fvcore_macs * T`, where T is the rate-coding window the agent was deployed with. The agent file documents the value of T used. Without this multiplier the paper would understate SNN inference cost by 25× (or whatever T is), making any SNN-vs-PPO efficiency claim wrong by the same factor.

Document the per-agent definition in each agent file's docstring so the paper can reference it.

The orchestrator never introspects an agent's torch module. Latency is hardware-dependent (orchestrator owns it); MACs are intrinsic (agent owns it).

---

## 5. Exam interface

```python
class Exam(Protocol):
    name: str
    def evaluate(
        self,
        policy: Policy,
        env_fn: Callable[[], gym.Env],
        context: dict,            # train_result, inf_latency_ms, inf_macs, param_count, ...
    ) -> list[ExperimentRecord]: ...
```

### 5.1 ResilienceExam (v1)

Wraps `EnvironmentCritic` and emits one `ExperimentRecord` per `ModType` probed. Each record's `exam.config` contains `{mod_type, s_max, n_episodes, max_iters}`. Each record's `exam.raw` contains `{aurc, s_half, s_max, cliff_slope, points, clean_return}`. Each record's `exam.formula` is `"success := aurc; adapt_score := s_half / s_max"`.

Future exams (`SampleEfficiencyExam`, `StabilityExam`) follow the same protocol and emit their own record sets. The record format does not change.

### 5.2 Diagnostics flow

The exam assembles each `ExperimentRecord`. It copies the agent's `TrainResult.diagnostics` (loss curves, NEAT generation count, ESN spectral radius) directly into the record's top-level `diagnostics` field. The exam MAY merge in exam-specific diagnostics (e.g. bisection iteration count) under a namespaced key like `diagnostics["resilience_probe"]`. Agent-supplied keys are never overwritten. This keeps `diagnostics` as a faithful record of what the agent reported, augmented — not replaced — by the exam.

---

## 6. Records schema

Layered: identity → config → cost → universal performance → exam block → env metrics → diagnostics.

```python
@dataclass
class ExperimentRecord:
    # === Identity & provenance ===
    experiment_id: str           # auto: timestamp + uuid
    timestamp:    str            # ISO UTC
    code_version: str            # git SHA if repo; "untracked" otherwise

    # === Config (training inputs) ===
    config: ExperimentConfig     # agent_name, env_id, train_seed, total_timesteps, hyperparams

    # === Hardware (expanded) ===
    hardware: HardwareInfo       # cpu, gpu, ram_gb, precision, backend, energy_efficiency_w

    # === Training cost ===
    train_time_sec:  float
    train_env_steps: int
    train_opt_steps: int
    param_count:     int

    # === Inference cost (universal, not an exam) ===
    inf_lat_ms:     float
    inf_macs:       int | None
    inf_mem_mb:     float | None    # peak RSS delta during eval forward passes
    inf_gpu_mem_mb: float | None    # peak CUDA allocation during eval; None if no CUDA

    # === Universal performance (always measured) ===
    clean_return:     float        # unperturbed mean return — raw skill baseline
    clean_return_std: float        # std across the clean-baseline episodes (consistency signal)

    # === Universal semantic metrics (formulas mutable; computed by the exam) ===
    success:     float           # [0,1]
    adapt_score: float | None    # [0,1] or None if exam doesn't measure it

    # === Exam block (all exam-specific stuff here) ===
    exam: ExamBlock              # {name, config, raw, formula}

    # === Env-specific creative metrics (env-bound, not exam-bound) ===
    env_metrics: dict            # action_smoothness, energy_consumed, time_to_first_balance, ...

    # === Agent-specific diagnostics ===
    diagnostics: dict            # loss curves, generation counts, reservoir spectral radius, ...

    notes: str = ""
```

```python
@dataclass
class HardwareInfo:
    cpu: str = ""
    gpu: str = ""                # auto-detected: torch.cuda.get_device_name(0), MPS, or "none"
    ram_gb: float = 0.0
    precision: str = "fp32"      # "fp32" | "fp16" | "bf16" | "int8"
    backend: str = "cpu"         # "cuda" | "cpu" | "mps" | "rocm" | "xla"
    energy_efficiency_w: float | None = None  # best-effort, NVIDIA-SMI avg if available
```

```python
@dataclass
class ExamBlock:
    name: str                    # "resilience" | "sample_efficiency" | ...
    config: dict                 # exam-specific config (e.g. ModType, s_max)
    raw: dict                    # frozen measurements (e.g. aurc, s_half, cliff_slope, points)
    formula: str                 # text label of how success/adapt_score were computed
```

### Schema renames from the existing code

- `ExperimentConfig.seed` → `ExperimentConfig.train_seed` (clearer: it's the training seed, not the eval seed which lives in the exam).
- `inf_latency_ms` → `inf_lat_ms` (concise).

### Layering rules

- **Adding new semantic metrics is cheap.** They derive from raw measurements.
- **Changing raw measurement format is expensive.** Errs on the side of recording too much raw data.
- **Adding new exams is free.** They write to `exam.config` and `exam.raw`; the rest of the schema is untouched.

### Env-specific creative metrics (`env_metrics`)

A `task_metrics.py` module owns these, with one function per `env_id`. For Pendulum-v1:

```python
{
    "action_smoothness":     float,  # mean |a_t - a_{t-1}| across clean-baseline episodes
    "energy_consumed":       float,  # mean of sum |a_t| per episode
    "time_to_first_upright": int | None,  # steps to |theta| < 0.2 rad first time, None if never
}
```

These are collected during the clean baseline rollout (s=0) inside the resilience probe and attached to every record from that probe. Adding a new env = registering one function.

---

## 7. The orchestrator (`run_experiments.py`)

### 7.1 ROSTER (the manifest)

A constant at the top of the file, easy to edit, CLI-overridable:

```python
ROSTER = {
    "agents":  ["PPO", "SAC", "ESN", "LNN", "NEAT", "SNN"],
    "envs":    ["Pendulum-v1"],
    "seeds":   [0],
    "exams":   ["resilience"],
    "timesteps_default": 100_000,
}
```

### 7.2 Per-cell pipeline

For each `(agent, env, seed)`:

1. **Train phase.**
   - If `weights/<agent>/<env>/seed<N>/` exists and not `--force-train`: `train_result = agent.load(path)`.
   - Else: `train_result = agent.train(env_fn, total_timesteps, seed)` then `agent.save(path)`.
   - Either way `train_result` carries real, frozen numbers.

2. **Inference cost phase** (all four are measured in a single 1000-forward-pass loop):
   - Warm up the policy with 100 dummy observations (populate caches, CUDA jit, etc.).
   - Record baseline RSS (`psutil.Process().memory_info().rss`) and, if CUDA available, `torch.cuda.reset_peak_memory_stats()`.
   - Time N=1000 forward passes → `inf_lat_ms` (median ms per call, not mean — robust to GC stalls).
   - After the loop: `inf_mem_mb = (peak_rss - baseline_rss) / 1e6`, `inf_gpu_mem_mb = torch.cuda.max_memory_allocated() / 1e6` if CUDA else `None`.
   - `inf_macs = agent.inference_macs()` — agent self-reports (intrinsic).

   Caveat: RSS deltas on Windows are noisy (process memory baseline drifts). The paper documents this and reports `inf_mem_mb` as best-effort.

3. **Exam phase.** For each enabled exam, call `exam.evaluate(policy, env_fn, context={...})`. Each returned record is appended to `results/results.jsonl`. The completed cell is logged to `results/run_log.jsonl`.

### 7.3 Resumability

`results/run_log.jsonl` accumulates one line per completed `(agent, env, seed, exam)`. On startup the orchestrator reads it and skips completed cells unless `--force-eval` or `--force-train` is set. `results.jsonl` is append-only — rows are never rewritten.

### 7.4 Failure isolation

Each cell is wrapped in `try/except`. On failure, the cell identifier and full traceback go to `results/errors.log`. The orchestrator continues to the next cell. `--max-failures N` bails out after N cell failures (default: unlimited).

### 7.5 CLI flags

```
--agents PPO SAC                   # subset of roster agents
--envs Pendulum-v1                 # subset of roster envs
--seeds 0 1 2                      # which seeds
--exams resilience                 # which exams
--timesteps N                      # override training budget
--skip-training                    # reuse saved weights (train_result loaded from disk)
--force-train                      # retrain even if weights exist
--force-eval                       # re-run exams even if logged
--output-dir results               # where to write
--dry-run                          # print plan, do nothing
--max-failures N                   # bail after N failures (default: unlimited)
```

### 7.6 Live progress

Each cell prints a one-liner and updates `results/status.txt` (so it can be tailed from another shell). Format:

```
[ 2/6 ] SAC × Pendulum-v1 × seed=0  ──  training (100k steps)         12m elapsed
[ 2/6 ] SAC × Pendulum-v1 × seed=0  ──  exam=resilience FLICKER       s_half=0.41
[ 2/6 ] SAC × Pendulum-v1 × seed=0  ──  done                          ETA: 4h 22m
```

Cell count for the v1 ROSTER: 6 agents × 1 env × 1 seed × 1 exam = **6 cells**. Each resilience cell internally produces 4 records (one per ModType), so `results.jsonl` will accumulate 24 rows for a full v1 run. The orchestrator counts cells, not records.

`results/run_log.jsonl` schema (one line per completed cell):
```json
{"cell_id": "PPO_Pendulum-v1_seed0_resilience", "completed_at": "2026-05-25T14:32:11Z", "n_records": 4}
```

---

## 8. Dependencies

Updated `requirements.txt` for v1:

```
gymnasium[classic-control]>=0.29
stable-baselines3>=2.3.0
numpy>=2.0                    
scipy>=1.10
psutil>=5.9
torch>=2.0
matplotlib>=3.7
reservoirpy>=0.3.12            # ESN
snntorch>=0.9                  # SNN
ncps>=1.0                      # LNN (liquid networks)
neat-python>=0.92              # NEAT
fvcore>=0.1.5                  # MACs for torch-based agents (PPO/SAC/LNN/SNN report via this)
```

### Optional dependencies

```
nvidia-ml-py>=13.595           # GPU power draw (W) — populates energy_efficiency_w
                               # Note: import name is still `pynvml` (drop-in).
                               # The legacy `pynvml` PyPI package is deprecated; do NOT install it.
                               # If absent or no CUDA GPU present, energy_efficiency_w stays None.
```

`wandb` removed until actually used.

---

## 9. Testing strategy

A new `tests/` directory with regression tests for the contracts the framework depends on. Not a full unit-test suite — targeted tests for the things that would silently break the paper if they failed.

| Test | What it asserts |
|---|---|
| `test_policy_reset.py` | Two episodes on identical seeds produce identical returns for a stateful agent, ONLY if `policy.reset()` is called between them. Catches the memory leak. |
| `test_wrappers_identity.py` | All wrappers at `strength=0` produce bit-identical observations/actions to the unwrapped env. |
| `test_critic_fit.py` | `_fit_logistic` recovers known `(s0, k)` parameters from a synthetic sigmoid within 5% tolerance. |
| `test_records_roundtrip.py` | `ExperimentRecord → to_dict → json.dumps → json.loads → from_dict` is lossless. |
| `test_physics_shift_loud_failure.py` | `PhysicsShiftWrapper` on an env with no known params raises `RuntimeError` (does not silently no-op). |
| `test_train_meta_persistence.py` | `agent.save(path); agent.load(path)` round-trips `TrainResult` identically. |

---

## 10. Reproducibility & local-machine independence

- All default paths resolve from `Path(__file__).parent` — running `python run_experiments.py` from any CWD produces the same output layout.
- `code_version` field captures git SHA when available; `"untracked"` otherwise.
- Seeds are explicit on every randomized call (`env.reset(seed=...)`, `model = PPO(..., seed=...)`).
- The ROSTER constant + `run_experiments.py` is the single command that reproduces the entire paper's data.
- `results/run_log.jsonl` allows safe interruption and resumption — no work is lost on crash.

---

## 11. Out of scope for v1

- Sample-efficiency, generalization, adaptation, stability, transfer exams (architecture supports them; v1 implements only resilience).
- Custom environments (Discordant Bandit, Drifting World, Sensor Fusion Maze). v1 uses `Pendulum-v1` only.
- wandb integration. The records.jsonl + offline plots cover the v1 paper.
- LaTeX paper itself. Code must be working first.
- JAX scalability. Optional far-future.
- Deep Active Inference agent. Far-future.

---

## 12. What "ready to run" looks like

After implementation:

```bash
# fresh clone, no weights, no results
cd folder1
pip install -r requirements.txt
pytest tests/                                # all tests pass
python run_experiments.py --dry-run          # prints 24-cell plan
python run_experiments.py --agents PPO --timesteps 10000   # smoke test, ~5 min
python run_experiments.py                    # full v1 run, ~40 hours
python plot_resilience.py                    # generates plots from results/results.jsonl (or --tag <tag>)
```

The smoke test step is the first real verification. If PPO completes one cell cleanly, the pipeline works.

---

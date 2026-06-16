"""Orchestrator — runs the cartesian product of (agents × envs × seeds × exams).

Usage:
    python run_experiments.py                                    # full ROSTER run
    python run_experiments.py --agents PPO --timesteps 10000     # smoke test
    python run_experiments.py --timesteps 100000 --tag 100k      # namespace outputs to results/100k/, weights/100k/
    python run_experiments.py --skip-training                    # reuse saved weights
    python run_experiments.py --dry-run                          # print plan only
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

from agents.base import AgentProtocol
from agents.ppo_agent import PPOAgent
from agents.esn_agent import ESNAgent
from exams.resilience import ResilienceExam
from records import HardwareInfo, ExperimentConfig


ROSTER = {
    "agents":  ["PPO", "ESN"],     # add SAC, LNN, SNN, NEAT in follow-on plans
    "envs":    ["Pendulum-v1"],
    "seeds":   [0],
    "exams":   ["resilience"],
    "timesteps_default": 100_000,
}

AGENT_FACTORIES = {
    "PPO": PPOAgent,
    "ESN": ESNAgent,
}

EXAM_FACTORIES = {
    "resilience": ResilienceExam,
}

WEIGHTS_DIR = PROJECT_ROOT / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_JSONL = RESULTS_DIR / "results.jsonl"
RUN_LOG_JSONL = RESULTS_DIR / "run_log.jsonl"
ERRORS_LOG = RESULTS_DIR / "errors.log"
STATUS_TXT = RESULTS_DIR / "status.txt"


def _apply_tag(tag: str | None) -> None:
    """Namespace all outputs under results/<tag>/ and weights/<tag>/. None = root paths."""
    global WEIGHTS_DIR, RESULTS_DIR, RESULTS_JSONL, RUN_LOG_JSONL, ERRORS_LOG, STATUS_TXT
    if not tag:
        return
    WEIGHTS_DIR = PROJECT_ROOT / "weights" / tag
    RESULTS_DIR = PROJECT_ROOT / "results" / tag
    RESULTS_JSONL = RESULTS_DIR / "results.jsonl"
    RUN_LOG_JSONL = RESULTS_DIR / "run_log.jsonl"
    ERRORS_LOG = RESULTS_DIR / "errors.log"
    STATUS_TXT = RESULTS_DIR / "status.txt"


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
        return out or "untracked"
    except Exception:
        return "untracked"


def _completed_cells() -> set[str]:
    if not RUN_LOG_JSONL.exists():
        return set()
    cells = set()
    for line in RUN_LOG_JSONL.read_text().splitlines():
        if line.strip():
            cells.add(json.loads(line)["cell_id"])
    return cells


def _log_cell_done(cell_id: str, n_records: int) -> None:
    RUN_LOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
    entry = {"cell_id": cell_id, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "n_records": n_records}
    with RUN_LOG_JSONL.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _log_error(cell_id: str, tb: str) -> None:
    ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ERRORS_LOG.open("a") as fh:
        fh.write(f"=== {cell_id} @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")
        fh.write(tb + "\n\n")


def _write_status(line: str) -> None:
    STATUS_TXT.parent.mkdir(parents=True, exist_ok=True)
    STATUS_TXT.write_text(line + "\n")


def _measure_inference_cost(policy, env_fn, n_warmup: int = 100, n_timed: int = 1000) -> dict:
    """Time N forward passes; measure peak memory delta."""
    import psutil, os
    env = env_fn()
    obs, _ = env.reset(seed=0)
    env.close()

    for _ in range(n_warmup):
        _ = policy(obs)

    proc = psutil.Process(os.getpid())
    base_rss = proc.memory_info().rss
    use_cuda = False
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            use_cuda = True
    except ImportError:
        pass

    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        _ = policy(obs)
        times.append(time.perf_counter() - t0)

    lat_ms = float(np.median(times)) * 1000.0
    peak_rss = proc.memory_info().rss
    inf_mem_mb = float(peak_rss - base_rss) / 1e6 if peak_rss > base_rss else 0.0

    inf_gpu_mem_mb = None
    if use_cuda:
        import torch
        inf_gpu_mem_mb = float(torch.cuda.max_memory_allocated()) / 1e6

    return {"inf_lat_ms": lat_ms, "inf_mem_mb": inf_mem_mb, "inf_gpu_mem_mb": inf_gpu_mem_mb}


def run_cell(agent_name: str, env_id: str, seed: int, exam_name: str,
             total_timesteps: int, force_train: bool, force_eval: bool) -> int:
    """Run one (agent, env, seed, exam) cell. Returns number of records written."""
    cell_id = f"{agent_name}_{env_id}_seed{seed}_{exam_name}"
    weights_path = WEIGHTS_DIR / agent_name / env_id / f"seed{seed}"

    def env_fn():
        return gym.make(env_id)

    AgentCls = AGENT_FACTORIES[agent_name]
    agent: AgentProtocol = AgentCls()

    if weights_path.exists() and not force_train:
        _write_status(f"{cell_id}: loading saved weights")
        train_result = agent.load(weights_path)
    else:
        _write_status(f"{cell_id}: training ({total_timesteps} steps)")
        train_result = agent.train(env_fn, total_timesteps=total_timesteps, seed=seed)
        agent.save(weights_path)

    _write_status(f"{cell_id}: measuring inference cost")
    policy = agent.policy()
    inf = _measure_inference_cost(policy, env_fn)
    inf["inf_macs"] = agent.inference_macs()

    ExamCls = EXAM_FACTORIES[exam_name]
    exam = ExamCls()
    _write_status(f"{cell_id}: running exam={exam_name}")

    cfg = ExperimentConfig(
        agent_name=agent_name, env_id=env_id, train_seed=seed,
        total_timesteps=total_timesteps,
        hyperparams=train_result.diagnostics.get("hyperparams", {}),
    )

    context = {
        "config":         cfg,
        "train_result":   train_result,
        "param_count":    agent.param_count(),
        "hardware":       HardwareInfo(),
        "code_version":   _git_sha(),
        **inf,
    }

    records = exam.evaluate(policy, env_fn, context)
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    for rec in records:
        rec.append_jsonl(RESULTS_JSONL)
    _log_cell_done(cell_id, len(records))
    return len(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=None)
    ap.add_argument("--envs", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--exams", nargs="+", default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--skip-training", action="store_true",
                    help="alias for --force-train=false; weights must exist on disk")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-failures", type=int, default=10**9)
    ap.add_argument("--tag", default=None,
                    help="Namespace outputs: results/<tag>/, weights/<tag>/. Lets you compare runs without clobber.")
    args = ap.parse_args()

    _apply_tag(args.tag)

    agents = args.agents or ROSTER["agents"]
    envs = args.envs or ROSTER["envs"]
    seeds = args.seeds or ROSTER["seeds"]
    exams = args.exams or ROSTER["exams"]
    timesteps = args.timesteps or ROSTER["timesteps_default"]

    plan = [(a, e, s, x) for a in agents for e in envs for s in seeds for x in exams]
    done = _completed_cells() if not args.force_eval else set()
    pending = [c for c in plan if f"{c[0]}_{c[1]}_seed{c[2]}_{c[3]}" not in done]

    print(f"Plan: {len(plan)} cells total, {len(pending)} pending, {len(done)} already done.")
    for i, (a, e, s, x) in enumerate(pending, 1):
        print(f"  [{i}/{len(pending)}] {a} × {e} × seed={s} × {x}")
    if args.dry_run:
        return

    failures = 0
    t_start = time.perf_counter()
    for i, (a, e, s, x) in enumerate(pending, 1):
        cell_id = f"{a}_{e}_seed{s}_{x}"
        header = f"[{i}/{len(pending)}] {a} × {e} × seed={s} × {x}"
        print(f"\n=== {header} ===")
        try:
            n = run_cell(a, e, s, x, total_timesteps=timesteps,
                         force_train=args.force_train,
                         force_eval=args.force_eval)
            elapsed = time.perf_counter() - t_start
            print(f"    done — {n} records — elapsed {elapsed/60:.1f} min")
        except Exception as exc:
            failures += 1
            tb = traceback.format_exc()
            _log_error(cell_id, tb)
            print(f"    FAILED: {exc}")
            if failures >= args.max_failures:
                print(f"Aborting: reached --max-failures={args.max_failures}")
                sys.exit(1)
    print(f"\nFinished. Wrote {RESULTS_JSONL.resolve()}")


if __name__ == "__main__":
    main()

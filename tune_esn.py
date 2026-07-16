"""tune_esn.py -- hyperparameter screening for ESNAgent (CMA-ES on readout).

One research goal: find hyperparameters where CMA-ES stops stalling on
Pendulum. The esn-probe baseline plateaus at clean ~= -915 within ~8
generations (loss_curve in results/esn-probe); this screens the knobs
one-factor-at-a-time around the defaults, plus a few motivated combos.

Honesty rules baked in:
  * Every config gets the SAME env-step budget (the school's cost currency),
    not the same generation count.
  * Final score = mean return over held-out eval episodes (seeds 900000+,
    disjoint from training eval seeds), NOT the optimizer's own best_return --
    es.best is the luckiest individual on 5 noisy episodes (selection bias).
  * env.reset() and policy.reset() are paired per episode (CLAUDE.md S 4).

Usage:
    python tune_esn.py                          # full screen, 500k steps, seeds 0 1
    python tune_esn.py --configs base in0.3     # subset
    python tune_esn.py --steps 100000 --seeds 0 # quick pass

Output: results/esn-tune/tune_results.jsonl (one row per config x seed;
deliberately NOT named results.jsonl -- different schema from
ExperimentRecord, must stay invisible to plots.py / gui.py loaders).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.esn_agent import ESNAgent, DEFAULT_HP

OUT_DIR = PROJECT_ROOT / "results" / "esn-tune"
ENV_ID = "Pendulum-v1"
HELDOUT_SEED_BASE = 900_000   # far above training eval seeds (seed*1000 + gens*eps)
HELDOUT_EPISODES = 30

# name -> hp overrides. One factor at a time around DEFAULT_HP, plus combos.
CONFIGS: dict[str, dict] = {
    "base":    {},
    # spectral radius: memory horizon / edge of chaos. Pendulum is fully
    # observable, so long memory may be pure liability.
    "sr0.3":   {"spectral_radius": 0.3},
    "sr0.6":   {"spectral_radius": 0.6},
    "sr1.1":   {"spectral_radius": 1.1},
    # leak rate: state update time-constant. 1.0 = fastest dynamics.
    "lr0.7":   {"leak_rate": 0.7},
    "lr1.0":   {"leak_rate": 1.0},
    # input scaling: theta_dot spans [-8, 8]; at 1.0 it saturates tanh units.
    "in0.1":   {"input_scaling": 0.1},
    "in0.3":   {"input_scaling": 0.3},
    # reservoir size: feature richness vs CMA-ES search dimension.
    "N50":     {"reservoir_size": 50},
    "N200":    {"reservoir_size": 200},
    # CMA-ES internals. Default popsize rule for 101-dim is ~17; 12 is under.
    "pop6":    {"cma_popsize": 6},
    "pop20":   {"cma_popsize": 20},
    "sig0.1":  {"cma_sigma_init": 0.1},
    "sig0.5":  {"cma_sigma_init": 0.5},
    # fitness noise: more episodes per individual, fewer generations per budget.
    "eps10":   {"eps_per_eval": 10},
    # combos: the "Pendulum is Markovian" bet -- fast, unsaturated, low-memory.
    "fast":    {"spectral_radius": 0.6, "leak_rate": 1.0, "input_scaling": 0.3},
    "fast200": {"spectral_radius": 0.6, "leak_rate": 1.0, "input_scaling": 0.3,
                "reservoir_size": 200, "cma_popsize": 20},
    # round 2 -- the three consistent round-1 winners combined (short memory,
    # small search space, more generations per budget), plus one-step variants.
    "combo":        {"spectral_radius": 0.3, "reservoir_size": 50, "cma_popsize": 6},
    "combo_sig0.5": {"spectral_radius": 0.3, "reservoir_size": 50, "cma_popsize": 6,
                     "cma_sigma_init": 0.5},
    "combo_sr0.1":  {"spectral_radius": 0.1, "reservoir_size": 50, "cma_popsize": 6},
    "combo_N25":    {"spectral_radius": 0.3, "reservoir_size": 25, "cma_popsize": 6},
    # round 3 -- code-level hypotheses from reading the installed sources:
    # (a) dense Win: reservoirpy's input_connectivity defaults to 0.1, so with
    #     3 obs dims ~73% of units get no direct observation input (0.9^3);
    #     classic ESN practice (Lukosevicius guide) sparsifies the RECURRENT
    #     matrix, not the input matrix.
    # (b) xfavorite deployment: es.best.x is the luckiest 5-episode individual
    #     (proven: identical deployed weights at 500k and 2.5M); pycma docs
    #     call result.xfavorite the best available estimate of the optimum.
    "combo_dense": {"spectral_radius": 0.3, "reservoir_size": 50, "cma_popsize": 6,
                    "input_connectivity": 1.0},
    "combo_xfav":  {"spectral_radius": 0.3, "reservoir_size": 50, "cma_popsize": 6,
                    "cma_deploy": "xfavorite"},
    "combo_dx":    {"spectral_radius": 0.3, "reservoir_size": 50, "cma_popsize": 6,
                    "input_connectivity": 1.0, "cma_deploy": "xfavorite"},
}

# round 4 -- local refinement around the confirmed winner, one delta per
# config. Includes extending the two grid edges the winner sits on:
# input_scaling's marginal was still rising at 1.0 (never tested above), and
# leak_rate only had two levels.
_WINNER = {"spectral_radius": 0.3, "reservoir_size": 50, "cma_popsize": 6}
CONFIGS.update({
    "r4_in1.5":  {**_WINNER, "input_scaling": 1.5},
    "r4_in2.0":  {**_WINNER, "input_scaling": 2.0},
    "r4_in3.0":  {**_WINNER, "input_scaling": 3.0},
    "r4_lk0.1":  {**_WINNER, "leak_rate": 0.1},
    "r4_lk0.2":  {**_WINNER, "leak_rate": 0.2},
    "r4_lk0.5":  {**_WINNER, "leak_rate": 0.5},
    "r4_sr0.2":  {**_WINNER, "spectral_radius": 0.2},
    "r4_sr0.45": {**_WINNER, "spectral_radius": 0.45},
    "r4_p4":     {**_WINNER, "cma_popsize": 4},
    "r4_N40":    {**_WINNER, "reservoir_size": 40},
    "r4_N65":    {**_WINNER, "reservoir_size": 65},
})


def grid_configs() -> dict[str, dict]:
    """Full factorial over the knobs that showed signal in rounds 1-3.
    4 sr x 3 N x 2 popsize x 3 input_scaling x 2 leak_rate = 144 configs.
    cma_deploy stays "best" (xfavorite lost the round-3 A/B at these budgets)."""
    out: dict[str, dict] = {}
    for sr in (0.1, 0.3, 0.6, 0.9):
        for n in (25, 50, 100):
            for pop in (6, 12):
                for scale in (0.1, 0.3, 1.0):
                    for lk in (0.3, 1.0):
                        name = f"g_sr{sr}_N{n}_p{pop}_in{scale}_lk{lk}"
                        out[name] = {"spectral_radius": sr, "reservoir_size": n,
                                     "cma_popsize": pop, "input_scaling": scale,
                                     "leak_rate": lk}
    return out


def heldout_eval(agent: ESNAgent, n_episodes: int = HELDOUT_EPISODES) -> tuple[float, float]:
    """Mean/std return of the trained policy on held-out seeds, resets paired."""
    env = gym.make(ENV_ID)
    policy = agent.policy()
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=HELDOUT_SEED_BASE + i)
        policy.reset()
        ep, done = 0.0, False
        while not done:
            obs, r, term, trunc, _ = env.step(policy(obs))
            ep += float(r)
            done = bool(term or trunc)
        returns.append(ep)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def run_config(name: str, overrides: dict, steps: int, seed: int) -> dict:
    agent = ESNAgent(hyperparams=overrides)
    t0 = time.perf_counter()
    tr = agent.train(lambda: gym.make(ENV_ID), total_timesteps=steps, seed=seed)
    mean_ret, std_ret = heldout_eval(agent)
    row = {
        "config": name,
        "hp": {**DEFAULT_HP, **overrides},
        "train_seed": seed,
        "budget_env_steps": steps,
        "actual_env_steps": tr.train_env_steps,
        "generations": tr.train_opt_steps,
        "train_time_sec": round(time.perf_counter() - t0, 2),
        "heldout_mean": round(mean_ret, 1),
        "heldout_std": round(std_ret, 1),
        "heldout_episodes": HELDOUT_EPISODES,
        "optimizer_best_return": round(tr.diagnostics.get("best_return", float("nan")), 1),
        "loss_curve": [round(x, 1) for x in tr.diagnostics.get("loss_curve", [])],
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--configs", nargs="+", default=None,
                    help=f"subset of: {' '.join(CONFIGS)}")
    ap.add_argument("--grid", action="store_true",
                    help="run the full factorial grid instead of the named configs")
    args = ap.parse_args()

    pool = grid_configs() if args.grid else CONFIGS
    names = args.configs or list(pool)
    unknown = [n for n in names if n not in pool]
    if unknown:
        ap.error(f"unknown config(s): {unknown}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "tune_results.jsonl"

    rows = []
    total = len(names) * len(args.seeds)
    i = 0
    for name in names:
        for seed in args.seeds:
            i += 1
            row = run_config(name, pool[name], args.steps, seed)
            rows.append(row)
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"[{i}/{total}] {name:>8} seed{seed}  heldout {row['heldout_mean']:>8.1f} "
                  f"+- {row['heldout_std']:>5.1f}  (optimizer said {row['optimizer_best_return']:>8.1f}, "
                  f"{row['generations']} gens, {row['train_time_sec']}s)", flush=True)

    # leaderboard: mean heldout across seeds per config
    print("\n=== leaderboard (mean heldout return across seeds) ===")
    by_cfg: dict[str, list[float]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r["heldout_mean"])
    board = sorted(by_cfg.items(), key=lambda kv: -np.mean(kv[1]))
    for name, vals in board:
        spread = f" (per-seed: {', '.join(f'{v:.0f}' for v in vals)})" if len(vals) > 1 else ""
        print(f"  {name:>8}  {np.mean(vals):>8.1f}{spread}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

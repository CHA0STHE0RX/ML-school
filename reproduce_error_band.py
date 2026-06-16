"""Reproduce the instrument error-band numbers reported in ML-paper1 §2.5 / §4.

The "error band" is the run-to-run spread of the resilience scores (s_half, AURC,
absolute_aurc) on a SINGLE frozen policy as the critic seed varies. It is the
instrument's own measurement noise: any cross-budget or cross-family difference
in §3 must exceed it to be real (Agarwal et al. 2021).

Each critic seed reseeds every evaluation episode (critic.py:_eval_policy) and the
GAUSSIAN_NOISE scale rollout (wrappers.py, scale_seed), so a fixed policy probed
under different critic seeds yields the score distribution we summarize here. Same
seed -> identical scores (checked at the end); the spread comes only from changing it.

Defaults probe the converged 500k/seed0 PPO checkpoint across 30 critic seeds and
all four axes, matching the numbers in the paper (AURC sd ~= 0.03, abs_aurc sd ~= 45).

Run:
    python reproduce_error_band.py
    python reproduce_error_band.py --policy-dir weights/2_5m/PPO/Pendulum-v1/seed0 --n-seeds 50
"""
from __future__ import annotations
import argparse
import statistics as st
import warnings
from collections import defaultdict
from pathlib import Path

import gymnasium as gym

from agents.ppo_agent import PPOAgent
from critic import EnvironmentCritic
from exams.resilience import S_MAX_BY_MOD

PROJECT_ROOT = Path(__file__).resolve().parent


def probe_once(policy, env_fn, mod_type, s_max, seed, n_episodes, max_iters):
    """One resilience probe; returns (aurc, s_half, absolute_aurc, fit_rmse, n_warnings)."""
    critic = EnvironmentCritic(
        base_env_fn=env_fn, mod_type=mod_type, s_max=s_max,
        n_episodes=n_episodes, max_iters=max_iters, seed=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = critic.probe(policy)
        n_warn = sum(1 for w in caught if issubclass(w.category, RuntimeWarning))
    return report.aurc, report.s_half, report.absolute_aurc, report.fit_rmse, n_warn


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy-dir", default="weights/500k/PPO/Pendulum-v1/seed0",
                    help="checkpoint dir (relative to project root) holding model.zip + train_meta.json")
    ap.add_argument("--env", default="Pendulum-v1")
    ap.add_argument("--n-seeds", type=int, default=30, help="number of critic seeds to sweep")
    ap.add_argument("--n-episodes", type=int, default=5, help="episodes per probe point (exam default)")
    ap.add_argument("--max-iters", type=int, default=6, help="bisection iterations (exam default)")
    args = ap.parse_args()

    policy_dir = (PROJECT_ROOT / args.policy_dir).resolve()
    agent = PPOAgent()
    agent.load(policy_dir)
    policy = agent.policy()
    env_fn = lambda: gym.make(args.env)

    print(f"Reseeding error band: {args.policy_dir}, {args.n_seeds} critic seeds, "
          f"{len(S_MAX_BY_MOD)} axes ({args.n_episodes} eps/point, {args.max_iters} bisection iters)\n")

    scores = defaultdict(lambda: {"aurc": [], "s_half": [], "abs": [], "rmse": []})
    warn_total = defaultdict(int)
    for mod_type, s_max in S_MAX_BY_MOD.items():
        for seed in range(args.n_seeds):
            aurc, s_half, abs_aurc, rmse, n_warn = probe_once(
                policy, env_fn, mod_type, s_max, seed, args.n_episodes, args.max_iters)
            d = scores[mod_type.name]
            d["aurc"].append(aurc); d["s_half"].append(s_half)
            d["abs"].append(abs_aurc); d["rmse"].append(rmse)
            warn_total[mod_type.name] += n_warn

    hdr = f"{'axis':<16}{'AURC mean':>11}{'AURC sd':>9}{'s_half mean':>13}{'s_half sd':>11}{'abs_aurc sd':>13}{'warns':>7}"
    print(hdr)
    print("-" * len(hdr))
    aurc_sds, shalf_sds = [], []
    for mod_type in S_MAX_BY_MOD:
        d = scores[mod_type.name]
        aurc_sd, shalf_sd = st.stdev(d["aurc"]), st.stdev(d["s_half"])
        aurc_sds.append(aurc_sd); shalf_sds.append(shalf_sd)
        print(f"{mod_type.name:<16}{st.mean(d['aurc']):>11.3f}{aurc_sd:>9.3f}"
              f"{st.mean(d['s_half']):>13.3f}{shalf_sd:>11.3f}"
              f"{st.stdev(d['abs']):>13.1f}{warn_total[mod_type.name]:>7}")

    print(f"\nmean across-seed sd over the {len(S_MAX_BY_MOD)} axes:  "
          f"AURC {st.mean(aurc_sds):.3f}   s_half {st.mean(shalf_sds):.3f}")

    # Reproducibility: same critic seed must give identical scores (the spread above
    # comes only from varying the seed, not from nondeterminism at a fixed one).
    mod0, smax0 = next(iter(S_MAX_BY_MOD.items()))
    a1, *_ = probe_once(policy, env_fn, mod0, smax0, 7, args.n_episodes, args.max_iters)
    a2, *_ = probe_once(policy, env_fn, mod0, smax0, 7, args.n_episodes, args.max_iters)
    verdict = "identical" if a1 == a2 else "DIFFERS (nondeterminism bug)"
    print(f"\nsame-seed reproducibility ({mod0.name} seed 7, AURC): {a1:.6f} vs {a2:.6f}  -> {verdict}")


if __name__ == "__main__":
    main()

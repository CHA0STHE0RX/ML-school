"""One-shot learnability check for SwingupDoublePendulum-v0.

Trains PPO once and compares swing-up performance to a random-action baseline.
Success is measured by PHYSICAL tip height (peak_height / uptime_fraction), which
is reward-agnostic, so the comparison is fair regardless of the reward weights.

Reward note: a swing-up must build high joint velocities to pump energy upward,
so we do NOT penalize velocity here (the C3 smoke showed a heavy velocity penalty
dominates the return for any energetic policy). Only a tiny control cost remains.

    python calibrate_swingup.py
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym

from swingup_env import register_swingup
from agents.ppo_agent import PPOAgent
from agents.base import StatelessPolicy
from task_metrics import _swingup_double_pendulum_metrics

BUDGET = 300_000
SEED = 0
CTRL_COST = 1e-4
VEL_COST = 0.0

register_swingup()


def env_fn():
    return gym.make("SwingupDoublePendulum-v0", ctrl_cost=CTRL_COST, vel_cost=VEL_COST)


def main() -> None:
    agent = PPOAgent()
    tr = agent.train(env_fn, total_timesteps=BUDGET, seed=SEED)
    trained = agent.policy()
    rng = np.random.default_rng(0)
    random_pol = StatelessPolicy(lambda obs: rng.uniform(-1, 1, 2).astype(np.float32))

    mt = _swingup_double_pendulum_metrics(env_fn, trained, n_episodes=10, seed=123)
    mr = _swingup_double_pendulum_metrics(env_fn, random_pol, n_episodes=10, seed=123)

    print(f"\n=== SwingupDoublePendulum-v0 learnability (budget={BUDGET}, seed={SEED}) ===")
    print(f"train_time = {tr.train_time_sec:.0f}s  "
          f"(ctrl_cost={CTRL_COST}, vel_cost={VEL_COST})")
    print(f"{'metric':<22}{'TRAINED':>10}{'RANDOM':>10}")
    for k in ("peak_height", "uptime_fraction", "upright_success_rate", "energy_consumed"):
        print(f"{k:<22}{mt[k]:>10.3f}{mr[k]:>10.3f}")
    learned = mt["peak_height"] > mr["peak_height"] + 0.3 or mt["uptime_fraction"] > 0.05
    print(f"\nLEARNED SWING-UP: {learned}")


if __name__ == "__main__":
    main()

"""Per-environment creative metrics (action_smoothness, energy_consumed, ...).

Adding a new env = register one function under METRIC_COLLECTORS.
"""
from __future__ import annotations
from typing import Any, Callable
import gymnasium as gym
import numpy as np
from agents.base import Policy


def _pendulum_metrics(env_fn: Callable[[], gym.Env], policy: Policy,
                      n_episodes: int, seed: int) -> dict[str, Any]:
    """For Pendulum-v1: action_smoothness, energy_consumed, time_to_first_upright."""
    smooth_means, energies, first_upright_steps = [], [], []
    for i in range(n_episodes):
        env = env_fn()
        obs, _ = env.reset(seed=seed + i)
        if hasattr(policy, "reset"):
            policy.reset()
        prev_a = None
        diffs, abs_actions = [], []
        first_upright = None
        step = 0
        done = False
        while not done:
            a = np.asarray(policy(obs), dtype=np.float32)
            if prev_a is not None:
                diffs.append(float(np.abs(a - prev_a).mean()))
            abs_actions.append(float(np.abs(a).sum()))
            obs, r, term, trunc, _ = env.step(a)
            # Pendulum obs = [cos(theta), sin(theta), thetadot] — upright when cos(theta) > 0.98
            if first_upright is None and float(obs[0]) > 0.98:
                first_upright = step
            prev_a = a
            step += 1
            done = term or trunc
        env.close()
        smooth_means.append(float(np.mean(diffs)) if diffs else 0.0)
        energies.append(float(np.sum(abs_actions)))
        first_upright_steps.append(first_upright)

    upright_with_value = [x for x in first_upright_steps if x is not None]
    return {
        "action_smoothness": float(np.mean(smooth_means)),
        "energy_consumed":   float(np.mean(energies)),
        "time_to_first_upright": int(np.mean(upright_with_value)) if upright_with_value else None,
    }


METRIC_COLLECTORS: dict[str, Callable[..., dict[str, Any]]] = {
    "Pendulum-v1": _pendulum_metrics,
}


def collect_env_metrics(env_id: str, env_fn, policy: Policy,
                        n_episodes: int, seed: int) -> dict[str, Any]:
    fn = METRIC_COLLECTORS.get(env_id)
    if fn is None:
        return {}
    return fn(env_fn, policy, n_episodes, seed)

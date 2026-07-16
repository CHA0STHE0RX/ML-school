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
    n_eps = len(first_upright_steps)
    return {
        "action_smoothness": float(np.mean(smooth_means)),
        "energy_consumed":   float(np.mean(energies)),
        "time_to_first_upright": int(np.mean(upright_with_value)) if upright_with_value else None,
        "time_to_first_upright_n": len(upright_with_value),
        "upright_success_rate": float(len(upright_with_value) / n_eps) if n_eps else 0.0,
    }


UPRIGHT_TIP_Z = 0.9  # tip-height threshold for "upright" (max reachable ~1.2)


def _swingup_double_pendulum_metrics(env_fn: Callable[[], gym.Env], policy: Policy,
                                     n_episodes: int, seed: int) -> dict[str, Any]:
    """For SwingupDoublePendulum-v0. Leads with unconditioned channels; the
    conditioned time_to_first_upright carries its denominator (survivorship)."""
    energies, smooths, peaks, uptimes, first_upright = [], [], [], [], []
    for i in range(n_episodes):
        env = env_fn()
        obs, _ = env.reset(seed=seed + i)
        if hasattr(policy, "reset"):
            policy.reset()
        prev_a, diffs, abs_actions, tip_heights = None, [], [], []
        up_steps, first, step, done = 0, None, 0, False
        while not done:
            a = np.asarray(policy(obs), dtype=np.float32)
            if prev_a is not None:
                diffs.append(float(np.abs(a - prev_a).mean()))
            abs_actions.append(float(np.abs(a).sum()))
            obs, r, term, trunc, info = env.step(a)
            tip = float(info["tip_height"])
            tip_heights.append(tip)
            if tip > UPRIGHT_TIP_Z:
                up_steps += 1
                if first is None:
                    first = step
            prev_a = a
            step += 1
            done = term or trunc
        env.close()
        energies.append(float(np.sum(abs_actions)))
        smooths.append(float(np.mean(diffs)) if diffs else 0.0)
        peaks.append(float(np.max(tip_heights)) if tip_heights else 0.0)
        uptimes.append(up_steps / step if step else 0.0)
        first_upright.append(first)

    n_eps = len(first_upright)
    succ = [x for x in first_upright if x is not None]
    return {
        "energy_consumed": float(np.mean(energies)),
        "action_smoothness": float(np.mean(smooths)),
        "peak_height": float(np.mean(peaks)),
        "uptime_fraction": float(np.mean(uptimes)),
        "upright_success_rate": float(len(succ) / n_eps) if n_eps else 0.0,
        "time_to_first_upright": int(np.mean(succ)) if succ else None,
        "time_to_first_upright_n": len(succ),
    }



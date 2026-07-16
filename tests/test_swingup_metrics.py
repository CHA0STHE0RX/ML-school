"""Swing-up creative metrics: unconditioned channels + conditioned-with-denominator."""
from __future__ import annotations
import pytest

pytest.importorskip("mujoco")

import numpy as np
import gymnasium as gym
from swingup_env import register_swingup
from agents.base import StatelessPolicy
from task_metrics import _swingup_double_pendulum_metrics

register_swingup()


def _env_fn():
    return gym.make("SwingupDoublePendulum-v0")


def test_swingup_metrics_contract():
    policy = StatelessPolicy(lambda obs: np.zeros(2, dtype=np.float32))
    m = _swingup_double_pendulum_metrics(_env_fn, policy, n_episodes=2, seed=0)
    for k in ("energy_consumed", "action_smoothness", "peak_height",
              "uptime_fraction", "upright_success_rate",
              "time_to_first_upright", "time_to_first_upright_n"):
        assert k in m
    assert m["energy_consumed"] >= 0.0
    assert -1.3 <= m["peak_height"] <= 1.3
    assert 0.0 <= m["uptime_fraction"] <= 1.0
    assert 0.0 <= m["upright_success_rate"] <= 1.0
    assert isinstance(m["time_to_first_upright_n"], int)
    assert m["time_to_first_upright"] is None or isinstance(m["time_to_first_upright"], int)


def test_zero_policy_never_reaches_upright():
    # Hanging under zero torque -> never above the upright threshold.
    policy = StatelessPolicy(lambda obs: np.zeros(2, dtype=np.float32))
    m = _swingup_double_pendulum_metrics(_env_fn, policy, n_episodes=2, seed=0)
    assert m["upright_success_rate"] == 0.0
    assert m["time_to_first_upright"] is None
    assert m["peak_height"] < 0.0  # stays in the lower half

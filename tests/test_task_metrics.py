"""task_metrics: per-env creative metrics (collector functions, no registry)."""
from __future__ import annotations
import gymnasium as gym
import numpy as np
from agents.base import StatelessPolicy
from task_metrics import _pendulum_metrics


def _env_fn():
    return gym.make("Pendulum-v1")


def test_pendulum_metrics_collected():
    policy = StatelessPolicy(lambda obs: np.array([0.0], dtype=np.float32))
    m = _pendulum_metrics(_env_fn, policy, n_episodes=2, seed=0)
    assert "action_smoothness" in m
    assert "energy_consumed" in m
    assert "time_to_first_upright" in m
    assert m["action_smoothness"] >= 0
    assert m["energy_consumed"] >= 0


def test_pendulum_metrics_survivorship_companions():
    policy = StatelessPolicy(lambda obs: np.array([0.0], dtype=np.float32))
    m = _pendulum_metrics(_env_fn, policy, n_episodes=3, seed=0)
    assert "upright_success_rate" in m
    assert 0.0 <= m["upright_success_rate"] <= 1.0
    assert "time_to_first_upright_n" in m
    assert isinstance(m["time_to_first_upright_n"], int)
    assert "time_to_first_upright" in m and "energy_consumed" in m

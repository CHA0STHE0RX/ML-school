"""task_metrics: per-env creative metrics."""
from __future__ import annotations
import gymnasium as gym
import numpy as np
from agents.base import StatelessPolicy
from task_metrics import collect_env_metrics


def test_pendulum_metrics_collected():
    def env_fn():
        return gym.make("Pendulum-v1")
    policy = StatelessPolicy(lambda obs: np.array([0.0], dtype=np.float32))
    m = collect_env_metrics("Pendulum-v1", env_fn, policy, n_episodes=2, seed=0)
    assert "action_smoothness" in m
    assert "energy_consumed" in m
    assert "time_to_first_upright" in m
    assert m["action_smoothness"] >= 0
    assert m["energy_consumed"] >= 0


def test_unknown_env_returns_empty_dict():
    def env_fn():
        return gym.make("Pendulum-v1")  # placeholder
    policy = StatelessPolicy(lambda obs: np.array([0.0], dtype=np.float32))
    m = collect_env_metrics("UnknownEnv-v0", env_fn, policy, n_episodes=1, seed=0)
    assert m == {}

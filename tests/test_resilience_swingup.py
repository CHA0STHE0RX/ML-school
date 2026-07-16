"""ResilienceExam runs end-to-end on SwingupDoublePendulum-v0 via its profile."""
from __future__ import annotations
import pytest

pytest.importorskip("mujoco")

import numpy as np
import gymnasium as gym
from agents.base import StatelessPolicy, TrainResult
from records import ExperimentConfig
from exams.resilience import ResilienceExam
from swingup_env import register_swingup

register_swingup()


def _env_fn():
    return gym.make("SwingupDoublePendulum-v0")


def _ctx():
    return {
        "config": ExperimentConfig(agent_name="PPO", env_id="SwingupDoublePendulum-v0",
                                   train_seed=0, total_timesteps=1),
        "train_result": TrainResult(0.0, 0, 0, {}),
        "param_count": 1, "inf_lat_ms": 0.0, "code_version": "test",
    }


def test_resilience_runs_on_swingup():
    # Deterministic, non-constant policy (sign of the two joint velocities).
    policy = StatelessPolicy(
        lambda obs: np.array([np.sign(obs[4]), np.sign(obs[5])], dtype=np.float32))
    records = ResilienceExam(n_episodes=1, max_iters=2).evaluate(policy, _env_fn, _ctx())
    assert len(records) == 4
    for r in records:
        assert 0.0 <= r.success <= 1.0
        assert r.exam.raw["points"]
        assert "peak_height" in r.env_metrics
        assert "upright_success_rate" in r.env_metrics

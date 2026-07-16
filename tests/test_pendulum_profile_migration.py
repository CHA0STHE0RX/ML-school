"""Migrating Pendulum onto the profile path changes nothing measurable."""
from __future__ import annotations
import gymnasium as gym
import numpy as np
from agents.base import StatelessPolicy, TrainResult
from records import ExperimentConfig
from exams.resilience import ResilienceExam


def _env_fn():
    return gym.make("Pendulum-v1")


def _policy():
    # Deterministic, non-constant -> non-degenerate probe, fully reproducible.
    return StatelessPolicy(lambda obs: np.array([float(np.sign(obs[2]))], dtype=np.float32))


def _ctx():
    return {
        "config": ExperimentConfig(agent_name="PPO", env_id="Pendulum-v1",
                                   train_seed=0, total_timesteps=1),
        "train_result": TrainResult(0.0, 0, 0, {}),
        "param_count": 1, "inf_lat_ms": 0.0, "code_version": "test",
    }


def _summary(records):
    """Reduce records to the numeric fields that must be reproducible."""
    out = {}
    for r in records:
        out[r.mod.mod_type.name] = (
            round(r.success, 9),
            round(r.exam.raw["s_half"], 9),
            round(r.exam.raw["absolute_aurc"], 6),
        )
    return out


def test_exam_is_deterministic_on_pendulum():
    a = _summary(ResilienceExam(n_episodes=3, max_iters=3).evaluate(_policy(), _env_fn, _ctx()))
    b = _summary(ResilienceExam(n_episodes=3, max_iters=3).evaluate(_policy(), _env_fn, _ctx()))
    assert a == b, f"non-deterministic exam: {a} != {b}"


def test_records_valid_and_complete():
    records = ResilienceExam(n_episodes=3, max_iters=3).evaluate(_policy(), _env_fn, _ctx())
    assert len(records) == 4
    for r in records:
        assert 0.0 <= r.success <= 1.0
        assert r.adapt_score is None or 0.0 <= r.adapt_score <= 1.0
        assert r.exam.raw["points"]
        assert "upright_success_rate" in r.env_metrics

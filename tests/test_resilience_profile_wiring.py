"""ResilienceExam reads s_max / physics_knob / metrics from the env profile."""
from __future__ import annotations
import gymnasium as gym
import numpy as np
from agents.base import StatelessPolicy, TrainResult
from records import ExperimentConfig
from exams.resilience import ResilienceExam
from env_profiles import get_profile


def _ctx(env_id):
    return {
        "config": ExperimentConfig(agent_name="PPO", env_id=env_id,
                                   train_seed=0, total_timesteps=1),
        "train_result": TrainResult(train_time_sec=0.0, train_env_steps=0,
                                    train_opt_steps=0, diagnostics={}),
        "param_count": 1, "inf_lat_ms": 0.0, "code_version": "test",
    }


def test_exam_uses_profile_s_max_on_pendulum():
    def env_fn():
        return gym.make("Pendulum-v1")
    # A non-constant policy so the probe is not degenerate.
    policy = StatelessPolicy(lambda obs: np.array([float(np.sign(obs[2]))], dtype=np.float32))
    exam = ResilienceExam(n_episodes=2, max_iters=2)
    records = exam.evaluate(policy, env_fn, _ctx("Pendulum-v1"))

    assert len(records) == 4
    prof = get_profile("Pendulum-v1")
    by_mod = {r.mod.mod_type: r for r in records}
    for mod, s_max in prof.s_max_by_mod.items():
        assert np.isclose(by_mod[mod].mod.mod_strength, s_max)
        assert "action_smoothness" in by_mod[mod].env_metrics


def test_collect_env_metrics_is_removed():
    import task_metrics
    assert not hasattr(task_metrics, "collect_env_metrics")
    assert not hasattr(task_metrics, "METRIC_COLLECTORS")

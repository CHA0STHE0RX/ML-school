"""Agent.save() and Agent.load() must preserve TrainResult identically."""
from __future__ import annotations
from pathlib import Path
import pytest
import gymnasium as gym
from agents.ppo_agent import PPOAgent


def _env_fn():
    return gym.make("Pendulum-v1")


def test_ppo_train_save_load_preserves_metrics(tmp_path: Path):
    agent = PPOAgent()
    original = agent.train(_env_fn, total_timesteps=512, seed=0)
    agent.save(tmp_path)

    fresh = PPOAgent()
    loaded = fresh.load(tmp_path)
    assert loaded.train_time_sec == original.train_time_sec
    assert loaded.train_env_steps == original.train_env_steps
    assert loaded.train_opt_steps == original.train_opt_steps
    assert loaded.diagnostics == original.diagnostics


def test_load_raises_on_missing_meta(tmp_path: Path):
    agent = PPOAgent()
    agent.train(_env_fn, total_timesteps=512, seed=0)
    agent.save(tmp_path)
    (tmp_path / "train_meta.json").unlink()

    fresh = PPOAgent()
    with pytest.raises(FileNotFoundError, match="train_meta"):
        fresh.load(tmp_path)

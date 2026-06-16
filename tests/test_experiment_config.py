"""ExperimentConfig field naming and validation."""
from __future__ import annotations
import pytest
from records import ExperimentConfig


def test_train_seed_field_exists():
    cfg = ExperimentConfig(agent_name="X", env_id="Y", train_seed=42, total_timesteps=1000)
    assert cfg.train_seed == 42


def test_validation_rejects_zero_timesteps():
    with pytest.raises(ValueError, match="total_timesteps"):
        ExperimentConfig(agent_name="X", env_id="Y", train_seed=0, total_timesteps=0)


def test_validation_rejects_empty_agent_name():
    with pytest.raises(ValueError, match="agent_name"):
        ExperimentConfig(agent_name="", env_id="Y", train_seed=0, total_timesteps=1)

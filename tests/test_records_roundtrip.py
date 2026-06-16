"""ExperimentRecord JSONL round-trip and new field validation."""
from __future__ import annotations
import json
from pathlib import Path
from records import (
    ExperimentRecord, ExperimentConfig, EnvironmentMod, ExamBlock, ModType,
)


def _make_record() -> ExperimentRecord:
    return ExperimentRecord(
        code_version="abc1234",
        config=ExperimentConfig(
            agent_name="PPO", env_id="Pendulum-v1",
            train_seed=0, total_timesteps=100_000,
            hyperparams={"lr": 3e-4},
        ),
        mod=EnvironmentMod(ModType.FLICKER, 0.9, "test"),
        train_time_sec=12.3,
        train_env_steps=100_000,
        train_opt_steps=15_625,
        param_count=4481,
        inf_lat_ms=0.42,
        inf_macs=4480,
        inf_mem_mb=12.0,
        inf_gpu_mem_mb=None,
        clean_return=-180.5,
        clean_return_std=15.2,
        success=0.62,
        adapt_score=0.45,
        exam=ExamBlock(
            name="resilience",
            config={"mod_type": "FLICKER", "s_max": 0.9},
            raw={"aurc": 0.62, "s_half": 0.405, "cliff_slope": 8.1, "points": []},
            formula="success := aurc; adapt_score := s_half / s_max",
        ),
        env_metrics={"action_smoothness": 0.12, "energy_consumed": 340.5},
        diagnostics={"loss_curve": [0.5, 0.3, 0.1]},
        notes="smoke test",
    )


def test_roundtrip_through_jsonl(tmp_path: Path):
    rec = _make_record()
    path = tmp_path / "results.jsonl"
    rec.append_jsonl(path)

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])

    assert loaded["config"]["agent_name"] == "PPO"
    assert loaded["config"]["train_seed"] == 0
    assert loaded["inf_lat_ms"] == 0.42
    assert loaded["inf_gpu_mem_mb"] is None
    assert loaded["clean_return_std"] == 15.2
    assert loaded["exam"]["name"] == "resilience"
    assert loaded["exam"]["raw"]["aurc"] == 0.62
    assert loaded["env_metrics"]["action_smoothness"] == 0.12


def test_two_records_append(tmp_path: Path):
    rec = _make_record()
    path = tmp_path / "results.jsonl"
    rec.append_jsonl(path)
    rec.append_jsonl(path)
    assert len(path.read_text().splitlines()) == 2


def test_success_validation_rejects_out_of_range():
    import pytest
    with pytest.raises(ValueError, match="success"):
        ExperimentRecord(
            config=ExperimentConfig("X", "Y", 0, 1),
            mod=EnvironmentMod(ModType.NONE, 0.0),
            success=1.5,
            exam=ExamBlock(name="test"),
        )


def test_mod_optional_for_non_perturbing_exams(tmp_path: Path):
    """Exams that don't perturb the environment (sample efficiency, memory, etc.)
    leave mod=None instead of fabricating a meaningless NONE/0.0 placeholder."""
    rec = ExperimentRecord(
        config=ExperimentConfig("PPO", "Pendulum-v1", 0, 1000),
        success=0.5,
        exam=ExamBlock(name="sample_efficiency"),
    )
    assert rec.mod is None

    path = tmp_path / "results.jsonl"
    rec.append_jsonl(path)
    loaded = json.loads(path.read_text().splitlines()[0])
    assert loaded["mod"] is None

""" Experiment tracking dataclasses for uniform judgement.
Usage:

    from records import ExperimentConfig, EnvironmentMod, ModType, ExperimentRecord

    record = ExperimentRecord(
        config=ExperimentConfig(
            agent_name="PPO", env_id="Pendulum-v1", train_seed=42, total_timesteps=50_000,
            hyperparams={"learning_rate": 3e-4},),
        mod=EnvironmentMod(ModType.FLICKER, 0.1),
        success=0.85,
        env_metrics={"action_smoothness": 0.12, "energy_consumed": 340.5},)
    print(record.to_dict())"""
from __future__ import annotations
import json
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any

class ModType(Enum):
    NONE = auto()
    FLICKER = auto()
    PHYSICS_SHIFT = auto()
    GAUSSIAN_NOISE = auto()
    ACTION_DELAY = auto()

@dataclass(frozen=True)
class ExperimentConfig:
    agent_name: str
    env_id: str
    train_seed: int
    total_timesteps: int
    hyperparams: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if not self.agent_name:
            raise ValueError("agent_name must not be empty")
        if not self.env_id:
            raise ValueError("env_id must not be empty")

@dataclass
class EnvironmentMod:
    mod_type: ModType = ModType.NONE
    mod_strength: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if self.mod_strength < 0.0:
            raise ValueError("mod_strength must be >= 0.0")

@dataclass
class ExamBlock:
    """Exam-specific block on an ExperimentRecord. Different exams populate this differently."""
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    formula: str = ""

@dataclass
class HardwareInfo:
    cpu: str = ""
    gpu: str = ""
    ram_gb: float = 0.0
    precision: str = "fp32"          # "fp32" | "fp16" | "bf16" | "int8"
    backend: str = "cpu"             # "cuda" | "cpu" | "mps" | "rocm" | "xla"
    energy_efficiency_w: float | None = None  # avg watts during training, None if unavailable

    def __post_init__(self) -> None:
        if not self.cpu:
            self.cpu = platform.processor() or platform.machine() or "unknown"
        if self.ram_gb <= 0.0:
            try:
                import psutil  # type: ignore[import-untyped]
                self.ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
            except ImportError:
                self.ram_gb = -1.0
        if not self.gpu:
            self.gpu, detected_backend = _detect_gpu_and_backend()
            if self.backend == "cpu":  # only override default
                self.backend = detected_backend


def _detect_gpu_and_backend() -> tuple[str, str]:
    """Returns (gpu_name, backend)"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0), "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS", "mps"
    except ImportError:
        pass
    return "none", "cpu"

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass
class ExperimentRecord:
    """One row in results.jsonl. Universal shape across all agents and all exams."""
    experiment_id: str = ""
    timestamp: str = field(default_factory=_utcnow_iso)
    code_version: str = "untracked"

    config: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(
            agent_name="x", env_id="x", train_seed=0, total_timesteps=1))
    mod: EnvironmentMod | None = None
    hardware: HardwareInfo = field(default_factory=HardwareInfo)

    train_time_sec: float = 0.0
    train_env_steps: int = 0
    train_opt_steps: int = 0
    param_count: int = 0

    inf_lat_ms: float = 0.0
    inf_macs: int | None = None
    inf_mem_mb: float | None = None
    inf_gpu_mem_mb: float | None = None

    clean_return: float = 0.0
    clean_return_std: float = 0.0

    success: float = 0.0
    adapt_score: float | None = None

    exam: ExamBlock = field(default_factory=lambda: ExamBlock(name="none"))

    env_metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    notes: str = ""

    def __post_init__(self) -> None:
        if not self.experiment_id:
            self.experiment_id = (
                datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
        if not 0.0 <= self.success <= 1.0:
            raise ValueError(f"success must be in [0, 1], got {self.success}")
        if self.adapt_score is not None and not 0.0 <= self.adapt_score <= 1.0:
            raise ValueError(f"adapt_score must be in [0, 1] or None, got {self.adapt_score}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if isinstance(d.get("mod"), dict) and "mod_type" in d["mod"]:
            d["mod"]["mod_type"] = self.mod.mod_type.name
        return d

    def append_jsonl(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(self.to_dict(), default=str) + "\n")

@dataclass
class Timer:
    elapsed: float = 0.0
    _start: float = field(default=0.0, init=False, repr=False)

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
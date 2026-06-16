"""Agent black-box protocol: Policy, TrainResult, AgentProtocol.

Every agent in ML-school satisfies this contract. The orchestrator and exams
never look inside the agent.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable
import numpy as np


class Policy:
    """Episode-stateful policy. reset() called between episodes, __call__ on each step.

    Stateless policies (PPO/SAC) override reset() as a no-op.
    Stateful policies (ESN, LNN, SNN) clear internal state in reset().
    """
    def reset(self) -> None:
        raise NotImplementedError

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class StatelessPolicy(Policy):
    """Wraps a bare callable; reset() is a no-op. Use for PPO/SAC/feedforward NEAT."""
    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]):
        self._fn = fn

    def reset(self) -> None:
        pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self._fn(obs)


@dataclass
class TrainResult:
    """Frozen training metrics. Saved to train_meta.json alongside model weights."""
    train_time_sec: float
    train_env_steps: int
    train_opt_steps: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AgentProtocol(Protocol):
    """Every agent satisfies this. The framework never introspects internals."""
    def train(self, env_fn: Callable[[], Any], total_timesteps: int, seed: int) -> TrainResult: ...
    def save(self, path: Path) -> None: ...                         # writes model + train_meta.json
    def load(self, path: Path) -> TrainResult: ...                  # returns persisted TrainResult
    def policy(self) -> Policy: ...
    def param_count(self) -> int: ...
    def inference_macs(self) -> int | None: ...                     # None if not measurable

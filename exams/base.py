"""Exam protocol — capability tests grade agents identically and emit records."""
from __future__ import annotations
from typing import Any, Callable, Protocol, runtime_checkable
import gymnasium as gym
from agents.base import Policy
from records import ExperimentRecord


@runtime_checkable
class Exam(Protocol):
    """Each exam tests one capability and emits one or more ExperimentRecord rows."""
    name: str
    def evaluate(
        self,
        policy: Policy,
        env_fn: Callable[[], gym.Env],
        context: dict[str, Any],
    ) -> list[ExperimentRecord]: ...

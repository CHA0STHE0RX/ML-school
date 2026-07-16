"""EnvProfile — the single source of truth for what each env needs adapted.

Consulted by run_experiments (make_fn, render) and exams/resilience
(s_max_by_mod, metrics_fn, physics_knob). Depends only on records + gymnasium +
task_metrics (references collector functions). It is NEVER imported by
wrappers.py, so the dependency graph stays acyclic (CLAUDE.md §10).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import gymnasium as gym

from records import ModType, ClassicAttrs, MujocoGravity, PhysicsKnob, DEFAULT_PHYSICS_ATTRS
from task_metrics import _pendulum_metrics, _swingup_double_pendulum_metrics


@dataclass(frozen=True)
class RenderConfig:
    """Reserved for the watchability spec: camera config / fps / overlay."""
    camera: dict[str, Any] = field(default_factory=dict)
    fps: int = 30
    overlay_strength: bool = True


@dataclass(frozen=True)
class EnvProfile:
    """Everything the framework needs adapted for one environment."""
    env_id: str
    make_fn: Callable[[], gym.Env]
    s_max_by_mod: dict[ModType, float]
    physics_knob: PhysicsKnob
    metrics_fn: Optional[Callable[..., dict[str, Any]]]
    action_kind: Literal["continuous", "discrete"]
    render: Optional[RenderConfig] = None
    notes: str = ""


PENDULUM_S_MAX: dict[ModType, float] = {
    ModType.FLICKER: 0.9,
    ModType.GAUSSIAN_NOISE: 1.5,
    ModType.ACTION_DELAY: 1.0,
    ModType.PHYSICS_SHIFT: 0.5,
}


def _make_swingup() -> gym.Env:
    # Lazy import: keeps env_profiles importable without MuJoCo for Pendulum-only runs.
    from swingup_env import register_swingup
    register_swingup()
    return gym.make("SwingupDoublePendulum-v0")


# Empirically calibrated by calibrate_s_max.py from the 2.5M reference policy:
# s_max = 2.5 * s_half (keeps s_half ~mid-range so the curve is well-resolved).
# Re-run that script if the reference policy or env changes. The strong policy is
# notably fragile to obs noise (s_half 0.047) and robust to gravity (0.273).
# Re-calibrated 2026-07-05 with the sim-explosion guard: the pre-guard
# PHYSICS_SHIFT value (0.98, from s_half 0.391) was inflated by MuJoCo's
# auto-reset teleporting exploded episodes to the upright goal state.
SWINGUP_S_MAX: dict[ModType, float] = {
    ModType.FLICKER: 0.3,
    ModType.GAUSSIAN_NOISE: 0.12,
    ModType.ACTION_DELAY: 0.2,
    ModType.PHYSICS_SHIFT: 0.68,
}


PROFILES: dict[str, EnvProfile] = {
    "Pendulum-v1": EnvProfile(
        env_id="Pendulum-v1",
        make_fn=lambda: gym.make("Pendulum-v1"),
        s_max_by_mod=dict(PENDULUM_S_MAX),
        physics_knob=ClassicAttrs(DEFAULT_PHYSICS_ATTRS),
        metrics_fn=_pendulum_metrics,
        action_kind="continuous",
        render=None,
        notes="Original calibration env. ClassicAttrs physics shift over (m, l, g).",
    ),
    "SwingupDoublePendulum-v0": EnvProfile(
        env_id="SwingupDoublePendulum-v0",
        make_fn=_make_swingup,
        s_max_by_mod=dict(SWINGUP_S_MAX),
        physics_knob=MujocoGravity(),
        metrics_fn=_swingup_double_pendulum_metrics,
        action_kind="continuous",
        render=RenderConfig(
            camera={"distance": 4.0, "azimuth": 90, "elevation": -10,
                    "lookat": [0.0, 0.0, -0.3]},
            fps=20,
        ),
        notes="Fixed-base double pendulum swing-up. MuJoCo-gravity physics shift. "
              "s_max values are initial; refine via calibrate_swingup.py.",
    ),
}


def get_profile(env_id: str) -> EnvProfile | None:
    return PROFILES.get(env_id)

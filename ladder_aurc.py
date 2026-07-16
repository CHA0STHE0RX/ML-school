"""Ladder-normalized AURC -- a cross-budget AND cross-env comparable resilience
metric, derived post-hoc from stored probe points (CLAUDE.md invariant #4:
raw frozen, semantics mutable).

The critic's `success` is INTRA-policy: it normalizes each policy to its own
[worst, clean] band, so a weak policy (little to lose) scores misleadingly high.
This module re-normalizes to ENV-FIXED endpoints instead:

    norm(r) = (r - floor) / (ceiling - floor)

  floor   = min(do-nothing, random) mean return on the env  (the ladder bottom)
  ceiling = the strongest policy's clean return on the env  (the ladder top)

Both endpoints are policy-INDEPENDENT, so the result is dimensionless [0,1] and
comparable across budgets and envs. A weak policy then correctly scores low
(its clean sits far below the ceiling). The ceiling cannot be known at probe
time (the strongest policy may not exist yet), which is why this is a post-hoc
derivation rather than a change to the critic.
"""
from __future__ import annotations
from typing import Any

import numpy as np
import gymnasium as gym

from agents.base import StatelessPolicy
from critic import _eval_policy, _aurc_and_shape


def ladder_floor(env_fn, n_episodes: int = 5, seed: int = 0) -> float:
    """Strength-independent floor: the lower of a do-nothing (zero-action) and a
    random (uniform-in-bounds) policy's mean clean return on the env."""
    env = env_fn()
    space = env.action_space
    lo = np.asarray(space.low, dtype=np.float32)
    hi = np.asarray(space.high, dtype=np.float32)
    env.close()

    zero = StatelessPolicy(lambda obs: np.zeros(space.shape, dtype=np.float32))
    rng = np.random.default_rng(seed)
    rand = StatelessPolicy(lambda obs: rng.uniform(lo, hi).astype(np.float32))

    do_nothing, _ = _eval_policy(env_fn, zero, n_episodes, seed)
    random_ret, _ = _eval_policy(env_fn, rand, n_episodes, seed)
    return float(min(do_nothing, random_ret))


def _get(p: Any, key: str) -> float:
    return p[key] if isinstance(p, dict) else getattr(p, key)


def ladder_aurc(points: list, s_max: float, floor: float, ceiling: float) -> float:
    """Recompute AURC for one mod from stored probe points, normalized to the
    fixed [floor, ceiling] band. `points` items expose .strength/.mean_return
    (ProbePoint) or ['strength']/['mean_return'] (a record's exam.raw points)."""
    pts = sorted(points, key=lambda p: _get(p, "strength"))
    s = np.array([_get(p, "strength") for p in pts], dtype=float)
    r = np.array([_get(p, "mean_return") for p in pts], dtype=float)
    nr = np.clip((r - floor) / max(ceiling - floor, 1e-9), 0.0, 1.0)
    return _aurc_and_shape(s, nr, s_max)[0]

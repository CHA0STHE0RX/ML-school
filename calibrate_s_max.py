"""Empirical s_max calibration: set each mod's strength ceiling from data.

Rule: s_max = C * s_half, where s_half is where the REFERENCE policy's normalized
return crosses 0.5, located on a coarse geometric strength grid. This keeps
s_half near the middle of the probed range (well-resolved curve, no step-like
fits). It is a ONE-TIME per-env calibration -- it does NOT add cost to the
benchmark's per-probe bisection (respecting CLAUDE.md S6.2). Use the STRONGEST
expected policy as the reference so weaker agents stay inside the range.

    python calibrate_s_max.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import gymnasium as gym

from swingup_env import register_swingup
from agents.ppo_agent import PPOAgent
from critic import _eval_policy
from wrappers import apply_mod
from env_profiles import get_profile
from records import ModType

C = 2.5
# all <= 0.9 so FlickerWrapper (which requires strength in [0,1]) is safe;
# brackets the observed swing-up breaking points (~0.05-0.26).
GRID = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.9)
ENV = "SwingupDoublePendulum-v0"
# Anchored to the project root so the script works from any CWD (CLAUDE.md §4).
REF_WEIGHTS = Path(__file__).resolve().parent / "weights/swingup_2_5m/PPO/SwingupDoublePendulum-v0/seed0"
MODS = (ModType.FLICKER, ModType.GAUSSIAN_NOISE, ModType.ACTION_DELAY, ModType.PHYSICS_SHIFT)


def _s_half(policy, mod, knob, seed: int = 1, n_eps: int = 5) -> float:
    base = lambda: gym.make(ENV)
    rets = []
    for s in GRID:
        rets.append(_eval_policy(lambda s=s: apply_mod(base(), mod, s, knob), policy, n_eps, seed)[0])
    rets = np.asarray(rets)
    clean, worst = rets[0], rets.min()
    norm = np.clip((rets - worst) / max(clean - worst, 1e-9), 0.0, 1.0)
    g = np.asarray(GRID)
    below = np.where(norm <= 0.5)[0]
    if not below.size:
        return float(g[-1])          # never breaks within the grid
    i = below[0]
    if i == 0:
        return float(g[0])           # already broken at s=0+
    x0, x1, y0, y1 = g[i - 1], g[i], norm[i - 1], norm[i]
    return float(x1 if y1 == y0 else x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))


def main() -> None:
    register_swingup()
    agent = PPOAgent()
    agent.load(REF_WEIGHTS)
    policy = agent.policy()
    knob = get_profile(ENV).physics_knob

    print(f"empirical s_max for {ENV}  (C={C}, ref={REF_WEIGHTS.parent.parent.parent.name})")
    out = {}
    for mod in MODS:
        sh = _s_half(policy, mod, knob)
        s_max = float(np.clip(round(C * sh, 2), 0.1, 2.0))
        out[mod] = s_max
        print(f"  {mod.name:<16} s_half={sh:.3f}  ->  s_max={s_max}")

    print("\nSWINGUP_S_MAX = {")
    for mod in MODS:
        print(f"    ModType.{mod.name}: {out[mod]},")
    print("}")


if __name__ == "__main__":
    main()

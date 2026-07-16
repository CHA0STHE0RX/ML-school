"""watch.py -- render a policy's behaviour to a GIF so you can SEE it act.

One file, one research goal: watch a trained policy (clean or under a
perturbation), and across training budgets if you point it at a sweep's weights.
Uses the env profile's render camera when it has one.

Requires the optional `imageio` package (not in requirements.txt):
    pip install imageio

Examples:
    # clean rollout of a trained policy
    python watch.py \
        --weights weights/swingup_2_5m/PPO/SwingupDoublePendulum-v0/seed0

    # under a perturbation
    python watch.py --weights <path> --mod PHYSICS_SHIFT --strength 0.4

    # the whole budget progression from a sweep (one gif per budget)
    python watch.py --env SwingupDoublePendulum-v0 \
        --progression swingup_50k swingup_100k swingup_250k swingup_500k swingup_1m swingup_2_5m
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym

from agents.ppo_agent import PPOAgent
from env_profiles import get_profile
from records import ModType
from wrappers import apply_mod

# Register optional custom envs if their deps are present (no-op otherwise).
try:
    from swingup_env import register_swingup
    register_swingup()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "results" / "render"


def _load_imageio():
    """imageio is an optional dependency — only watch.py needs it."""
    try:
        import imageio.v2 as imageio
        return imageio
    except ImportError as e:
        raise SystemExit(
            "watch.py needs the optional `imageio` package to write GIFs.\n"
            "Install it with:  pip install imageio"
        ) from e


def _make_render_env(env_id: str, mod: str | None, strength: float) -> gym.Env:
    prof = get_profile(env_id)
    kwargs = {"render_mode": "rgb_array"}
    if prof is not None and prof.render is not None:
        kwargs["default_camera_config"] = prof.render.camera
    env = gym.make(env_id, **kwargs)
    if mod and strength > 0.0:
        knob = prof.physics_knob if prof is not None else None
        env = apply_mod(env, ModType[mod], strength, knob)
    return env


def render_gif(env_id: str, weights: Path, out_path: Path,
               mod: str | None = None, strength: float = 0.0,
               n_steps: int = 300, seed: int = 0) -> dict:
    """Render one rollout to a GIF. Returns simple stats for a sanity print."""
    prof = get_profile(env_id)
    fps = prof.render.fps if (prof is not None and prof.render is not None) else 30

    agent = PPOAgent()
    agent.load(weights)
    policy = agent.policy()

    env = _make_render_env(env_id, mod, strength)
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    frames = [env.render()]
    tips = []
    for _ in range(n_steps):
        obs, _, term, trunc, info = env.step(policy(obs))
        frames.append(env.render())
        if "tip_height" in info:
            tips.append(info["tip_height"])
        if term or trunc:
            break
    env.close()

    imageio = _load_imageio()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(out_path, frames, fps=fps)
    except TypeError:
        imageio.mimsave(out_path, frames, duration=1.0 / fps)

    stats = {"frames": len(frames)}
    if tips:
        stats["frac_up"] = float(np.mean([t > 0.9 for t in tips]))
        stats["peak"] = float(np.max(tips))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="SwingupDoublePendulum-v0")
    ap.add_argument("--weights", default=None, help="path to a saved policy dir")
    ap.add_argument("--mod", default=None, help="ModType name, e.g. PHYSICS_SHIFT")
    ap.add_argument("--strength", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--progression", nargs="+", default=None,
                    help="sweep tags to render one gif each (weights/<tag>/PPO/<env>/seed<seed>)")
    args = ap.parse_args()

    if args.progression:
        for tag in args.progression:
            wp = PROJECT_ROOT / "weights" / tag / "PPO" / args.env / f"seed{args.seed}"
            if not wp.exists():
                print(f"missing {wp}")
                continue
            out = OUT_DIR / f"{tag}.gif"
            st = render_gif(args.env, wp, out, args.mod, args.strength, args.steps, args.seed)
            print(f"{tag:>16} -> {out}  {st}")
        return

    if not args.weights:
        ap.error("provide --weights, or --progression <tags...>")
    suffix = f"_{args.mod}_{args.strength}" if args.mod else "_clean"
    out = Path(args.out) if args.out else OUT_DIR / f"{args.env}{suffix}.gif"
    st = render_gif(args.env, Path(args.weights), out, args.mod, args.strength, args.steps, args.seed)
    print(f"{out}  {st}")


if __name__ == "__main__":
    main()

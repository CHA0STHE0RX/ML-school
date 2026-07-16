"""EnvProfile registry: the single source of truth for per-env adaptation."""
from __future__ import annotations
import sys
import gymnasium as gym
from records import ModType
from env_profiles import PROFILES, EnvProfile, get_profile

_ALL_MODS = (ModType.FLICKER, ModType.GAUSSIAN_NOISE,
             ModType.ACTION_DELAY, ModType.PHYSICS_SHIFT)


def test_all_profiles_wellformed():
    for env_id, p in PROFILES.items():
        assert isinstance(p, EnvProfile)
        try:
            env = p.make_fn()
        except ImportError:
            continue  # env needs an optional dep (e.g. mujoco) not installed
        try:
            if p.action_kind == "continuous":
                assert isinstance(env.action_space, gym.spaces.Box)
            else:
                assert isinstance(env.action_space, gym.spaces.Discrete)
            for mod in _ALL_MODS:
                assert mod in p.s_max_by_mod, f"{env_id} missing s_max for {mod}"
        finally:
            env.close()


def test_pendulum_profile_matches_legacy_constants():
    p = get_profile("Pendulum-v1")
    assert p is not None
    assert p.s_max_by_mod == {
        ModType.FLICKER: 0.9, ModType.GAUSSIAN_NOISE: 1.5,
        ModType.ACTION_DELAY: 1.0, ModType.PHYSICS_SHIFT: 0.5,
    }
    assert p.action_kind == "continuous"


def test_get_profile_unknown_returns_none():
    assert get_profile("NoSuchEnv-v0") is None


def test_wrappers_does_not_import_env_profiles():
    # Acyclic dep graph (CLAUDE.md §10): importing wrappers must not pull in env_profiles.
    for m in ("wrappers", "env_profiles"):
        sys.modules.pop(m, None)
    import wrappers  # noqa: F401
    assert "env_profiles" not in sys.modules

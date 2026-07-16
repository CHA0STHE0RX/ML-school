"""SwingupDoublePendulum-v0 has a wellformed EnvProfile."""
from __future__ import annotations
import pytest

pytest.importorskip("mujoco")

from records import ModType, MujocoGravity
from env_profiles import get_profile

_ALL_MODS = (ModType.FLICKER, ModType.GAUSSIAN_NOISE,
             ModType.ACTION_DELAY, ModType.PHYSICS_SHIFT)


def test_swingup_profile_present_and_buildable():
    p = get_profile("SwingupDoublePendulum-v0")
    assert p is not None
    assert p.action_kind == "continuous"
    assert isinstance(p.physics_knob, MujocoGravity)
    assert p.metrics_fn is not None
    for mod in _ALL_MODS:
        assert mod in p.s_max_by_mod
    env = p.make_fn()
    try:
        assert env.observation_space.shape == (6,)
        assert env.action_space.shape == (2,)
    finally:
        env.close()

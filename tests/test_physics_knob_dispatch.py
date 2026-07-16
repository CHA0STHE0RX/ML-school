"""PhysicsShiftWrapper dispatches on the declared knob; legacy path is the default."""
from __future__ import annotations
import pytest
import gymnasium as gym
import numpy as np
from records import ModType, ClassicAttrs, MujocoGravity
from wrappers import apply_mod, PhysicsShiftWrapper


def test_default_knob_is_legacy_classic_path():
    # No knob -> ClassicAttrs(DEFAULT) -> Pendulum has m,l,g -> no raise.
    PhysicsShiftWrapper(gym.make("Pendulum-v1"), 0.3)


def test_classic_attrs_missing_name_raises():
    with pytest.raises(RuntimeError, match="no known physics parameters"):
        PhysicsShiftWrapper(gym.make("Pendulum-v1"), 0.3, ClassicAttrs(("nonexistent_attr",)))


def test_mujoco_gravity_on_nonmujoco_raises():
    with pytest.raises(RuntimeError, match="MuJoCo"):
        PhysicsShiftWrapper(gym.make("Pendulum-v1"), 0.3, MujocoGravity())


def test_mujoco_gravity_scales_heavier_exact():
    pytest.importorskip("mujoco")
    env = gym.make("InvertedDoublePendulum-v5")
    default_gz = float(env.unwrapped.model.opt.gravity[2])
    s = 0.4
    w = PhysicsShiftWrapper(env, s, MujocoGravity())
    w.reset(seed=0)  # first reset flips _flip False->True -> heavier (1+s)
    assert np.isclose(float(w.env.unwrapped.model.opt.gravity[2]), default_gz * (1 + s))


def test_apply_mod_strength_zero_returns_unwrapped():
    pytest.importorskip("mujoco")
    env = gym.make("InvertedDoublePendulum-v5")
    out = apply_mod(env, ModType.PHYSICS_SHIFT, 0.0, MujocoGravity())
    assert out is env  # strength 0 -> no wrapper, gravity untouched

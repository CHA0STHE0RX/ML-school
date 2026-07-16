"""PhysicsKnob descriptors live in records (the dep root) so wrappers can use them."""
from __future__ import annotations
from records import ClassicAttrs, MujocoGravity, DEFAULT_PHYSICS_ATTRS


def test_classic_attrs_default_names():
    assert ClassicAttrs().names == DEFAULT_PHYSICS_ATTRS
    assert "gravity" in DEFAULT_PHYSICS_ATTRS and "g" in DEFAULT_PHYSICS_ATTRS


def test_classic_attrs_custom_names():
    assert ClassicAttrs(("g", "l")).names == ("g", "l")


def test_classic_attrs_is_frozen_and_eq():
    assert ClassicAttrs(("g",)) == ClassicAttrs(("g",))


def test_mujoco_gravity_constructs_and_eq():
    assert MujocoGravity() == MujocoGravity()

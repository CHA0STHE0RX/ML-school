"""SwingupDoublePendulum-v0: fixed-base double pendulum swing-up env."""
from __future__ import annotations
import pytest

pytest.importorskip("mujoco")

import numpy as np
import gymnasium as gym
from swingup_env import SwingupDoublePendulumEnv, register_swingup

register_swingup()


def test_registration_and_spaces():
    env = gym.make("SwingupDoublePendulum-v0")
    try:
        assert env.observation_space.shape == (6,)
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (2,)
        assert np.allclose(env.action_space.low, -1.0)
        assert np.allclose(env.action_space.high, 1.0)
    finally:
        env.close()


def test_starts_hanging():
    env = SwingupDoublePendulumEnv()
    obs, _ = env.reset(seed=0)
    # obs = [sin1, sin2, cos1, cos2, v1, v2]; hanging -> cos(shoulder)=-1
    assert obs[2] < -0.97, f"shoulder not hanging: cos(shoulder)={obs[2]}"
    tip_z = float(env.data.site_xpos[env._tip][2])
    assert tip_z < -1.0, f"tip not hanging low: {tip_z}"
    env.close()


def test_no_early_termination():
    env = SwingupDoublePendulumEnv()
    env.reset(seed=0)
    for _ in range(300):
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        assert terminated is False
        assert truncated is False  # raw class has no TimeLimit
    env.close()


def test_reward_increases_with_height():
    env = SwingupDoublePendulumEnv()
    env.reset(seed=0)
    env.set_state(np.array([0.0, 0.0]), np.zeros(2))       # both up -> tip high
    r_up, _ = env._get_rew(np.zeros(2))
    env.set_state(np.array([np.pi, 0.0]), np.zeros(2))     # hanging -> tip low
    r_down, _ = env._get_rew(np.zeros(2))
    assert r_up > r_down, f"reward not increasing with height: up={r_up} down={r_down}"
    env.close()


@pytest.mark.parametrize("ratio,expected", [(0.0, 0.0), (0.2, 20.0), (1.0, 100.0)])
def test_weak_wrist_gear_scaling(ratio, expected):
    env = SwingupDoublePendulumEnv(shoulder_ratio=ratio)
    sid = env.model.actuator("m_shoulder").id
    eid = env.model.actuator("m_elbow").id
    assert np.isclose(float(env.model.actuator_gear[sid, 0]), expected)
    assert np.isclose(float(env.model.actuator_gear[eid, 0]), 100.0)
    env.close()


def test_zero_ratio_shoulder_is_passive():
    """ratio=0 -> shoulder torque has no authority (pure Acrobot).

    Start at the exact hanging equilibrium (no reset noise) so the only thing
    that could move the shoulder joint is its own actuator; at ratio=0 the
    joint must stay put even under a full shoulder command.
    """
    env = SwingupDoublePendulumEnv(shoulder_ratio=0.0)
    env.reset(seed=0)
    env.set_state(np.array([np.pi, 0.0]), np.zeros(2))  # exact equilibrium
    for _ in range(30):
        env.step(np.array([1.0, 0.0], dtype=np.float32))  # full shoulder command
    assert np.isclose(float(env.data.qpos[0]), np.pi, atol=1e-2)
    env.close()


def test_deterministic():
    def roll():
        env = SwingupDoublePendulumEnv()
        obs, _ = env.reset(seed=0)
        log = [obs.copy()]
        rng = np.random.default_rng(0)
        for _ in range(20):
            obs, *_ = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
            log.append(obs.copy())
        env.close()
        return np.asarray(log)
    np.testing.assert_array_equal(roll(), roll())


def test_mujoco_gravity_scales_on_swingup():
    from records import MujocoGravity
    from wrappers import PhysicsShiftWrapper
    env = SwingupDoublePendulumEnv()
    gz = float(env.unwrapped.model.opt.gravity[2])
    w = PhysicsShiftWrapper(env, 0.5, MujocoGravity())
    w.reset(seed=0)  # heavier-only first reset -> (1 + 0.5)
    assert np.isclose(float(w.env.unwrapped.model.opt.gravity[2]), gz * 1.5)
    env.close()


def test_sim_explosion_terminates_without_teleport_bonus():
    """MuJoCo auto-resets an unstable sim to defaults — which for this model is
    BOTH POLES UPRIGHT (tip_z = +1.2, the reward-maximal state). An exploded
    episode must therefore terminate at the reward floor, not silently collect
    the goal reward for breaking the simulator (MUJOCO_LOG.TXT shows these
    explosions happen in practice under heavy perturbation)."""
    env = SwingupDoublePendulumEnv()
    env.reset(seed=0)
    env.set_state(np.array([np.pi, 0.0]), np.array([np.nan, 0.0]))  # hanging, NaN velocity
    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is True
    assert info.get("sim_exploded") is True
    assert np.isfinite(obs).all()
    assert reward <= -1.1, f"exploded step must score at the floor, got {reward}"
    env.close()

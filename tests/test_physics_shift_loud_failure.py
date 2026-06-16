"""PhysicsShiftWrapper must raise on envs with no recognized physics params."""
from __future__ import annotations
import pytest
import gymnasium as gym
from wrappers import PhysicsShiftWrapper


class FakeEnv(gym.Env):
    """Env with no gravity/mass/length attributes."""
    metadata = {}
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}
    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


def test_raises_on_env_with_no_known_params():
    env = FakeEnv()
    with pytest.raises(RuntimeError, match="no known physics parameters"):
        PhysicsShiftWrapper(env, strength=0.5)


def test_no_raise_on_pendulum():
    env = gym.make("Pendulum-v1")
    PhysicsShiftWrapper(env, strength=0.3)

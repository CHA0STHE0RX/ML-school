"""All wrappers must be bit-identity passthroughs at strength=0."""
from __future__ import annotations
import gymnasium as gym
import numpy as np
from records import ModType
from wrappers import apply_mod


SEED = 1234


def _rollout(env, n_steps=50):
    obs, _ = env.reset(seed=SEED)
    obs_log = [obs.copy()]
    for _ in range(n_steps):
        a = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(a)
        obs_log.append(obs.copy())
        if term or trunc:
            break
    return np.stack(obs_log)


def test_flicker_identity_at_zero():
    base = gym.make("Pendulum-v1")
    base.action_space.seed(SEED)
    wrapped = apply_mod(gym.make("Pendulum-v1"), ModType.FLICKER, 0.0)
    wrapped.action_space.seed(SEED)
    np.testing.assert_array_equal(_rollout(base), _rollout(wrapped))


def test_gaussian_noise_identity_at_zero():
    base = gym.make("Pendulum-v1")
    base.action_space.seed(SEED)
    wrapped = apply_mod(gym.make("Pendulum-v1"), ModType.GAUSSIAN_NOISE, 0.0)
    wrapped.action_space.seed(SEED)
    np.testing.assert_array_equal(_rollout(base), _rollout(wrapped))


def test_action_delay_identity_at_zero():
    base = gym.make("Pendulum-v1")
    base.action_space.seed(SEED)
    wrapped = apply_mod(gym.make("Pendulum-v1"), ModType.ACTION_DELAY, 0.0)
    wrapped.action_space.seed(SEED)
    np.testing.assert_array_equal(_rollout(base), _rollout(wrapped))

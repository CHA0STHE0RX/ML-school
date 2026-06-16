"""PPOAgent: train a tiny model, sanity-check the protocol."""
from __future__ import annotations
import gymnasium as gym
import numpy as np
from agents.ppo_agent import PPOAgent
from agents.base import Policy


def _env_fn():
    return gym.make("Pendulum-v1")


def test_ppo_train_then_predict():
    agent = PPOAgent()
    result = agent.train(_env_fn, total_timesteps=512, seed=0)  # tiny budget for unit test
    assert result.train_time_sec > 0
    assert result.train_env_steps >= 512
    assert result.train_opt_steps > 0

    policy = agent.policy()
    assert isinstance(policy, Policy)
    policy.reset()
    obs, _ = _env_fn().reset(seed=0)
    a = policy(obs)
    assert a.shape == (1,)  # Pendulum action shape


def test_ppo_param_count_positive():
    agent = PPOAgent()
    agent.train(_env_fn, total_timesteps=512, seed=0)
    assert agent.param_count() > 0


def test_ppo_inference_macs_positive():
    agent = PPOAgent()
    agent.train(_env_fn, total_timesteps=512, seed=0)
    macs = agent.inference_macs()
    assert macs is None or macs > 0  # fvcore may not work in all environments

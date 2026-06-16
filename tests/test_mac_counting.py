"""inference_macs() methodology check.

Both agents claim to report 'multiply-accumulates in the deployed forward pass,
linear layers only, no biases, no activations.' PPO arrives at this via fvcore
auto-tracing; ESN counts by hand. This test pins down that both methodologies
agree with a hand calculation for known network shapes - so cross-family
inf_macs comparisons are honest.
"""
from __future__ import annotations
import gymnasium as gym
import pytest
from agents.ppo_agent import PPOAgent
from agents.esn_agent import ESNAgent


def test_ppo_macs_match_manual_count():
    """PPO MlpPolicy on Pendulum-v1: default policy branch is 3 -> 64 -> 64,
    action head 64 -> 1. Value branch (3 -> 64 -> 64 -> 1) is excluded per
    CLAUDE.md S 4 - discarded at inference time.

    Pendulum obs_dim=3, act_dim=1.
    Manual:
      mlp_extractor.policy_net  (3 -> 64 -> 64)  = 3*64 + 64*64 = 4288
      action_net                (64 -> 1)        = 64
      total                                       = 4352
    fvcore counts linear-layer multiply-accumulates only (no biases, no activations).
    """
    agent = PPOAgent()
    agent.train(lambda: gym.make("Pendulum-v1"), total_timesteps=2048, seed=0)
    reported = agent.inference_macs()
    assert reported is not None, "PPO inference_macs returned None"
    manual = (3 * 64 + 64 * 64) + 64 * 1
    assert reported == manual, (
        f"PPO MACs mismatch: fvcore reports {reported}, manual count is {manual}. "
        f"Either fvcore is missing a layer, or the MlpExtractor shape changed."
    )


def test_esn_macs_match_manual_count():
    """ESN default: 100-unit reservoir, rc_connectivity=0.1, on Pendulum (obs=3, act=1).
    Manual: input_proj (3*100) + recurrent (100*100*0.1) + readout (100*1)
          = 300 + 1000 + 100 = 1400.
    """
    agent = ESNAgent({"cma_popsize": 5, "eps_per_eval": 1})
    agent.train(lambda: gym.make("Pendulum-v1"), total_timesteps=1500, seed=0)
    reported = agent.inference_macs()
    assert reported is not None, "ESN inference_macs returned None"
    N = 100
    manual = 3 * N + int(N * N * 0.1) + N * 1
    assert reported == manual, (
        f"ESN MACs mismatch: hand-count reports {reported}, manual is {manual}. "
        f"Check input_dim, reservoir_size, rc_connectivity, or act_dim."
    )

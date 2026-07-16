"""Ladder-normalized AURC: fixed [floor, ceiling] band -> weak policies score low."""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from ladder_aurc import ladder_aurc, ladder_floor


def test_weak_policy_scores_mid_not_high():
    # A weak policy: clean ~50, returns barely move under perturbation. With its
    # OWN clean as ceiling it would read ~1.0; against a fixed high ceiling it
    # sits where its capability actually is (~half the floor->best band).
    pts = [{"strength": s, "mean_return": 50.0} for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    a = ladder_aurc(pts, 1.0, floor=-1200.0, ceiling=1000.0)
    assert 0.5 < a < 0.65  # (50 - -1200)/(1000 - -1200) ~= 0.57, roughly flat


def test_strong_policy_that_breaks_scores_lower():
    # Strong clean (= ceiling) but collapses to the floor under perturbation.
    pts = [{"strength": 0.0, "mean_return": 1000.0},
           {"strength": 0.5, "mean_return": -100.0},
           {"strength": 1.0, "mean_return": -1200.0}]
    a = ladder_aurc(pts, 1.0, floor=-1200.0, ceiling=1000.0)
    assert a < 0.6  # loses most of its capability across the range


def test_robust_strong_policy_scores_high():
    # Strong clean, barely degrades -> stays near the top of the band.
    pts = [{"strength": s, "mean_return": 950.0} for s in (0.0, 0.5, 1.0)]
    a = ladder_aurc(pts, 1.0, floor=-1200.0, ceiling=1000.0)
    assert a > 0.9


def test_ladder_floor_is_finite_min():
    def env_fn():
        return gym.make("Pendulum-v1")
    f = ladder_floor(env_fn, n_episodes=2, seed=0)
    assert np.isfinite(f)

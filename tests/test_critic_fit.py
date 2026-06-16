"""Critic logistic fit recovers known params from synthetic curve."""
from __future__ import annotations
import numpy as np
from critic import _fit_logistic


def test_recovers_known_s0_and_k():
    rng = np.random.default_rng(42)
    s0_true, k_true = 0.45, 12.0
    strengths = np.linspace(0, 1, 21)
    truth = 1.0 / (1.0 + np.exp(k_true * (strengths - s0_true)))
    noisy = np.clip(truth + rng.normal(0, 0.02, size=truth.shape), 0, 1)
    s0_hat, k_hat = _fit_logistic(strengths, noisy)
    assert abs(s0_hat - s0_true) < 0.05
    assert abs(k_hat - k_true) < 2.0


def test_fallback_on_flat_curve():
    strengths = np.linspace(0, 1, 5)
    flat = np.ones_like(strengths) * 0.5
    s0_hat, k_hat = _fit_logistic(strengths, flat)
    assert 0.0 <= s0_hat <= 1.0
    assert k_hat > 0


def _rmse(strengths, norm):
    # mirrors the residual the critic records as fit_rmse
    s0, k = _fit_logistic(strengths, norm)
    fitted = 1.0 / (1.0 + np.exp(k * (strengths - s0)))
    return float(np.sqrt(np.mean((fitted - norm) ** 2)))


def test_fit_rmse_low_for_logistic_high_for_nonmonotone():
    s = np.linspace(0, 1, 9)
    clean = 1.0 / (1.0 + np.exp(12.0 * (s - 0.4)))   # a true logistic
    bumpy = np.array([1.0, 0.9, 0.3, 0.15, 0.4, 0.7, 0.55, 0.2, 0.05])  # dips then recovers
    assert _rmse(s, clean) < 0.05      # logistic fits itself
    assert _rmse(s, bumpy) > 0.15      # non-monotone curve trips the warn threshold

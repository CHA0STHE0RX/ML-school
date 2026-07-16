"""When the logistic misfits a non-monotone curve, the critic falls back to a
shape-agnostic interpolation instead of reporting nonsense logistic params."""
from __future__ import annotations
import numpy as np
from critic import _aurc_and_shape, NONLOGISTIC_RMSE


def test_logistic_curve_uses_logistic():
    s = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    nr = 1.0 / (1.0 + np.exp(15.0 * (s - 0.5)))  # clean logistic crossing at 0.5
    aurc, s_half, k, rmse, method = _aurc_and_shape(s, nr, 1.0)
    assert method == "logistic"
    assert rmse <= NONLOGISTIC_RMSE
    assert 0.0 <= aurc <= 1.0
    assert abs(s_half - 0.5) < 0.1


def test_nonmonotone_curve_falls_back():
    s = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    nr = np.array([1.0, 0.3, 0.9, 0.2, 0.5, 0.0])  # up-down-up: logistic cannot fit
    aurc, s_half, k, rmse, method = _aurc_and_shape(s, nr, 1.0)
    assert rmse > NONLOGISTIC_RMSE
    assert method == "pchip_fallback"
    assert 0.0 <= aurc <= 1.0
    assert 0.0 <= s_half <= 1.0


def test_no_overflow_on_extreme_slope():
    # near-step curve -> huge k; must not emit an overflow warning or NaN.
    s = np.array([0.0, 0.05, 0.1, 0.15, 1.0])
    nr = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn any RuntimeWarning (overflow) into an error
        aurc, s_half, k, rmse, method = _aurc_and_shape(s, nr, 1.0)
    assert np.isfinite(aurc) and 0.0 <= aurc <= 1.0

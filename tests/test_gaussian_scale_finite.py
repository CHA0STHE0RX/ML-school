"""GaussianObsNoise scale estimation must survive non-finite warmup observations.

An unstable sim (MuJoCo NaN burst) during the fixed-seed warmup rollout would
otherwise put NaN into the cached per-dim scale and poison every subsequent
observation of the probe.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym

from wrappers import GaussianObsNoise


class _NaNBurstEnv(gym.Env):
    """Emits a NaN observation every third step, finite ones otherwise."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self):
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(3, dtype=np.float64), {}

    def step(self, action):
        self._t += 1
        if self._t % 3 == 0:
            obs = np.full(3, np.nan)
        else:
            obs = self.np_random.normal(size=3)
        return obs, 0.0, False, False, {}


def test_scale_finite_despite_nan_warmup_rows():
    w = GaussianObsNoise(_NaNBurstEnv(), strength=0.5)
    w.reset(seed=0)  # triggers _estimate_scale over the NaN-bursting warmup
    assert w._scale is not None
    assert np.isfinite(w._scale).all(), f"scale poisoned by NaN rows: {w._scale}"


def test_scale_defaults_to_ones_when_all_rows_nonfinite():
    class _AllNaNEnv(_NaNBurstEnv):
        def reset(self, *, seed=None, options=None):
            super(_NaNBurstEnv, self).reset(seed=seed)
            self._t = 0
            return np.full(3, np.nan), {}

        def step(self, action):
            return np.full(3, np.nan), 0.0, False, False, {}

    w = GaussianObsNoise(_AllNaNEnv(), strength=0.5)
    w.reset(seed=0)
    assert np.isfinite(w._scale).all()
    assert (w._scale == 1.0).all()

"""EnvironmentCritic: bisection probe + logistic fit -> (s_half, AURC, cliff_slope)."""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import gymnasium as gym
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from records import ModType
from wrappers import apply_mod

PolicyFn = Callable[[np.ndarray], np.ndarray]

@dataclass
class ProbePoint:
    strength: float
    mean_return: float
    std_return: float
    n_episodes: int

@dataclass
class RReport:
    mod_type: ModType
    clean_return: float
    s_max: float
    s_half: float           # strength at 50% normalized return
    aurc: float             # area under fitted curve, normalized to [0, 1] (intra-policy)
    absolute_aurc: float    # area under raw-return curve (env reward units); cross-policy comparable
    cliff_slope: float      # logistic k; high = cliff, low = !cliff
    fit_rmse: float = 0.0    # RMSE of fitted curve vs probe points; high = logistic misfits this curve
    fit_method: str = "logistic"  # "logistic" | "pchip_fallback" (used when the logistic misfits)
    points: list[ProbePoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"mod_type": self.mod_type.name, "clean_return": self.clean_return,
                "s_max": self.s_max, "s_half": self.s_half, "aurc": self.aurc,
                "absolute_aurc": self.absolute_aurc, "cliff_slope": self.cliff_slope,
                "fit_rmse": self.fit_rmse, "fit_method": self.fit_method,
                "points": [p.__dict__ for p in self.points]}

def _eval_policy(env_fn, policy, n_episodes: int, seed: int):
    """Evaluate policy across n_episodes. Calls policy.reset() at each episode boundary.

    `policy` must satisfy the Policy protocol: __call__(obs) -> action, and reset() -> None.
    A bare callable can be wrapped in StatelessPolicy().
    """
    returns = []
    for i in range(n_episodes):
        env = env_fn()
        obs, _ = env.reset(seed=seed + i)
        if hasattr(policy, "reset"):
            policy.reset()
        done, ep_ret = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy(obs))
            ep_ret += float(r)
            done = term or trunc
        env.close()
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns))

def _fit_logistic(strengths: np.ndarray, norm_returns: np.ndarray) -> tuple[float, float]:
    """Fit f(s) = 1 / (1 + exp(k*(s-s0))). Returns (s0, k). Falls back to interpolation on failure."""
    f = lambda s, s0, k: 1.0 / (1.0 + np.exp(k * (s - s0)))
    # Probe points arrive in evaluation order (0, s_max, then bisection mids) —
    # sort by strength before interpolating, np.interp needs monotone xp.
    order = np.argsort(strengths)
    s_sorted, nr_sorted = strengths[order], norm_returns[order]
    s0_init = (
        float(np.interp(0.5, nr_sorted[::-1], s_sorted[::-1]))
        if norm_returns.min() < 0.5 < norm_returns.max()
        else float(strengths.mean())
    )
    try:
        popt, _ = curve_fit(f, strengths, norm_returns, p0=[s0_init, 10.0], maxfev=2000)
        return float(popt[0]), float(popt[1])
    except Exception:
        return s0_init, 10.0


NONLOGISTIC_RMSE = 0.15  # above this the logistic misfits; use the shape-agnostic fallback


def _logistic(s, s0, k):
    # exponent clamped so an extreme (near-step) slope cannot overflow exp().
    return 1.0 / (1.0 + np.exp(np.clip(k * (s - s0), -50.0, 50.0)))


def _aurc_and_shape(strengths: np.ndarray, norm_returns: np.ndarray, s_max: float):
    """Return (aurc, s_half, cliff_slope, fit_rmse, fit_method).

    Fit the logistic and integrate it. If it misfits (fit_rmse > NONLOGISTIC_RMSE
    -- the curve is non-monotone or non-logistic), fall back to a shape-agnostic
    PCHIP interpolation of the probe points so AURC and s_half still describe the
    actual curve instead of a wrong functional form.
    """
    s0, k = _fit_logistic(strengths, norm_returns)
    fitted = _logistic(strengths, s0, k)
    fit_rmse = float(np.sqrt(np.mean((fitted - norm_returns) ** 2)))
    grid = np.linspace(0.0, s_max, 200)

    if fit_rmse <= NONLOGISTIC_RMSE:
        curve = _logistic(grid, s0, k)
        aurc = float(trapezoid(curve, grid) / s_max)
        return aurc, float(np.clip(s0, 0.0, s_max)), float(k), fit_rmse, "logistic"

    # Fallback: interpolate the points directly, no functional-form assumption.
    order = np.argsort(strengths)
    xs, ys = strengths[order], norm_returns[order]
    xs_u = np.unique(xs)
    ys_u = np.array([ys[xs == x].mean() for x in xs_u])
    curve = np.clip(PchipInterpolator(xs_u, ys_u)(grid), 0.0, 1.0)
    aurc = float(trapezoid(curve, grid) / s_max)
    below = np.where(curve <= 0.5)[0]
    s_half = float(grid[below[0]]) if below.size else float(np.clip(s0, 0.0, s_max))
    return aurc, s_half, float(k), fit_rmse, "pchip_fallback"

class EnvironmentCritic:
    """Bisection probe over a single perturbation axis."""

    def __init__(self, base_env_fn: Callable[[], gym.Env], mod_type: ModType,
                 s_max: float = 1.0, n_episodes: int = 5, max_iters: int = 6,
                 tol: float = 0.02, seed: int = 0, physics_knob=None):
        self.base_env_fn = base_env_fn
        self.mod_type = mod_type
        self.s_max = s_max
        self.n_episodes = n_episodes
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed
        self.physics_knob = physics_knob

    def _env_at(self, s: float):
        return lambda: apply_mod(self.base_env_fn(), self.mod_type, s, self.physics_knob)

    def probe(self, policy: PolicyFn) -> RReport:
        points: list[ProbePoint] = []

        def eval_at(s: float) -> float:
            m, sd = _eval_policy(self._env_at(s), policy, self.n_episodes, self.seed)
            points.append(ProbePoint(s, m, sd, self.n_episodes))
            return m

        r_clean = eval_at(0.0)
        r_worst = eval_at(self.s_max)

        denom = max(r_clean - r_worst, 1e-9)
        if r_clean - r_worst < 0.1 * abs(r_clean) + 1e-9:
            warnings.warn(
                f"Critic probe degenerate: clean_return ({r_clean:.3f}) and worst "
                f"({r_worst:.3f}) are too close. AURC may be meaningless. Mod={self.mod_type.name}, "
                f"s_max={self.s_max}. Either policy is constant or s_max is too small.",
                RuntimeWarning,
                stacklevel=2,
            )
        norm = lambda r: float(np.clip((r - r_worst) / denom, 0.0, 1.0))

        # bisect toward 50% crossing
        lo, hi = 0.0, self.s_max
        for _ in range(self.max_iters):
            if hi - lo < self.tol:
                break
            mid = 0.5 * (lo + hi)
            if norm(eval_at(mid)) > 0.5:
                lo = mid
            else:
                hi = mid
        # alt: golden-section search, or uniform grid of max_iters+2 points
        strengths = np.array([p.strength for p in points])
        norm_returns = np.array([norm(p.mean_return) for p in points])
        aurc, s_half, cliff_slope, fit_rmse, fit_method = _aurc_and_shape(
            strengths, norm_returns, self.s_max)
        if fit_rmse > NONLOGISTIC_RMSE:
            warnings.warn(
                f"Logistic fit residual high (rmse={fit_rmse:.3f}) for Mod={self.mod_type.name}: "
                f"curve is non-monotone/non-logistic -- using {fit_method} for AURC/s_half.",
                RuntimeWarning, stacklevel=2,
            )

        # absolute_aurc: mean raw return over the strength range, in env reward units.
        # Computed by trapezoid on raw probe points (not the fit). Lets you compare
        # absolute robustness between policies whose clean baselines differ.
        pts_sorted = sorted(points, key=lambda p: p.strength)
        raw_strengths = np.array([p.strength for p in pts_sorted])
        raw_returns = np.array([p.mean_return for p in pts_sorted])
        absolute_aurc = float(trapezoid(raw_returns, raw_strengths) / self.s_max)

        return RReport(
            mod_type=self.mod_type, clean_return=r_clean, s_max=self.s_max,
            s_half=s_half, aurc=aurc,
            absolute_aurc=absolute_aurc, cliff_slope=cliff_slope,
            fit_rmse=fit_rmse, fit_method=fit_method, points=pts_sorted,
        )

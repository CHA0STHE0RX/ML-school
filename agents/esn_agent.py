"""ESNAgent - Echo State Network with linear readout, trained by CMA-ES.

The reservoir is a fixed random recurrent net (reservoirpy). Only the readout
weights are trained. CMA-ES optimizes the readout to maximize mean episode
return across `eps_per_eval` episodes per individual. All individuals within
one generation see the same set of evaluation seeds (advances per-generation,
not per-individual) so CMA-ES compares apples to apples.

param_count: readout weights only (reservoir is fixed at init, not trained).
inference_macs: manually counted - reservoirpy ops are not fvcore-traceable.
reset() on the returned policy clears the reservoir state vector (required
between episodes per the stateful-policy isolation invariant, CLAUDE.md S 4).
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Callable
import numpy as np
import cma
from reservoirpy.nodes import Reservoir
from agents.base import Policy, TrainResult


DEFAULT_HP = {
    "reservoir_size": 100,
    "spectral_radius": 0.9,
    "leak_rate": 0.3,
    "input_scaling": 1.0,
    "rc_connectivity": 0.1,
    "cma_popsize": 12,
    "cma_sigma_init": 0.3,
    "eps_per_eval": 5,
}


class _ESNPolicy(Policy):
    """Stateful policy. reset() clears the reservoir state vector."""
    def __init__(self, reservoir: Reservoir, readout_W: np.ndarray,
                 action_low: np.ndarray, action_high: np.ndarray):
        self._r = reservoir
        self._W = readout_W
        self._lo = action_low
        self._hi = action_high
        self._scale = (action_high - action_low) / 2.0
        self._offset = (action_high + action_low) / 2.0

    def reset(self) -> None:
        self._r.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        state = self._r(obs.astype(np.float32))
        action = state @ self._W[:-1] + self._W[-1]
        action = np.tanh(action) * self._scale + self._offset
        return action.astype(np.float32)


class ESNAgent:
    def __init__(self, hyperparams: dict[str, Any] | None = None):
        self.hp = {**DEFAULT_HP, **(hyperparams or {})}
        self._reservoir: Reservoir | None = None
        self._readout_W: np.ndarray | None = None
        self._action_low: np.ndarray | None = None
        self._action_high: np.ndarray | None = None
        self._obs_dim: int | None = None
        self._act_dim: int | None = None
        self._init_seed: int | None = None
        self._train_result: TrainResult | None = None

    def _make_reservoir(self, seed: int) -> Reservoir:
        return Reservoir(
            units=self.hp["reservoir_size"],
            sr=self.hp["spectral_radius"],
            lr=self.hp["leak_rate"],
            input_scaling=self.hp["input_scaling"],
            rc_connectivity=self.hp["rc_connectivity"],
            seed=seed,
        )

    def _eval_individual(self, env, W: np.ndarray, eval_seed_base: int
                         ) -> tuple[float, int]:
        """eps_per_eval episodes; returns (negated_mean_return, total_steps).
        Negated because CMA-ES minimizes."""
        eps = self.hp["eps_per_eval"]
        returns: list[float] = []
        total_steps = 0
        scale = (self._action_high - self._action_low) / 2.0
        offset = (self._action_high + self._action_low) / 2.0
        for i in range(eps):
            obs, _ = env.reset(seed=eval_seed_base + i)
            self._reservoir.reset()
            ep_return = 0.0
            done = False
            while not done:
                state = self._reservoir(obs.astype(np.float32))
                action = state @ W[:-1] + W[-1]
                action = (np.tanh(action) * scale + offset).astype(np.float32)
                obs, reward, term, trunc, _ = env.step(action)
                ep_return += float(reward)
                total_steps += 1
                done = bool(term or trunc)
            returns.append(ep_return)
        return -float(np.mean(returns)), total_steps

    def train(self, env_fn: Callable[[], Any], total_timesteps: int, seed: int
              ) -> TrainResult:
        env = env_fn()
        self._obs_dim = int(env.observation_space.shape[0])
        self._act_dim = int(env.action_space.shape[0])
        self._action_low = np.asarray(env.action_space.low, dtype=np.float32)
        self._action_high = np.asarray(env.action_space.high, dtype=np.float32)
        self._init_seed = int(seed)
        self._reservoir = self._make_reservoir(seed)
        # Lazy init: first call materializes W. Then reset back to clean state.
        _ = self._reservoir(np.zeros(self._obs_dim, dtype=np.float32))
        self._reservoir.reset()

        N = self.hp["reservoir_size"]
        n_params = (N + 1) * self._act_dim

        rng = np.random.default_rng(seed)
        x0 = rng.normal(0, 0.1, size=n_params)
        es = cma.CMAEvolutionStrategy(
            x0, self.hp["cma_sigma_init"],
            {"popsize": self.hp["cma_popsize"], "seed": seed + 1,
             "verbose": -9, "maxfevals": float("inf")},
        )

        env_steps = 0
        opt_steps = 0
        gen_base = seed * 1000 + 1  # eval-seed base, advances per generation
        loss_curve: list[float] = []
        t0 = time.perf_counter()

        while env_steps < total_timesteps and not es.stop():
            solutions = es.ask()
            fitnesses: list[float] = []
            for sol in solutions:
                W = sol.reshape(N + 1, self._act_dim)
                neg_return, steps = self._eval_individual(env, W, gen_base)
                fitnesses.append(neg_return)
                env_steps += steps
                if env_steps >= total_timesteps:
                    break
            if len(fitnesses) >= es.sp.weights.mu:
                es.tell(solutions[:len(fitnesses)], fitnesses)
                opt_steps += 1
                loss_curve.append(-min(fitnesses))  # best (highest) return this gen
            gen_base += self.hp["eps_per_eval"]

        train_time = time.perf_counter() - t0
        env.close()

        best_x = es.best.x if (es.best is not None and es.best.x is not None) else x0
        self._readout_W = best_x.reshape(N + 1, self._act_dim).astype(np.float32)

        self._train_result = TrainResult(
            train_time_sec=train_time,
            train_env_steps=env_steps,
            train_opt_steps=opt_steps,
            diagnostics={
                "hyperparams": dict(self.hp),
                "best_return": float(-es.best.f) if es.best is not None else 0.0,
                "loss_curve": loss_curve,
            },
        )
        return self._train_result

    def save(self, path: Path) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        if self._readout_W is None or self._train_result is None:
            raise RuntimeError("Cannot save: ESNAgent has not been trained yet.")
        np.savez(
            path / "model.npz",
            readout_W=self._readout_W,
            action_low=self._action_low,
            action_high=self._action_high,
            obs_dim=np.array([self._obs_dim]),
            act_dim=np.array([self._act_dim]),
            init_seed=np.array([self._init_seed]),
        )
        meta = {
            "train_time_sec": self._train_result.train_time_sec,
            "train_env_steps": self._train_result.train_env_steps,
            "train_opt_steps": self._train_result.train_opt_steps,
            "diagnostics": self._train_result.diagnostics,
        }
        (path / "train_meta.json").write_text(json.dumps(meta))

    def load(self, path: Path) -> TrainResult:
        path = Path(path)
        model_path = path / "model.npz"
        meta_path = path / "train_meta.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"ESNAgent.load: missing model.npz or train_meta.json at {path}. "
                f"Refusing to produce a record with zeroed training metrics."
            )
        data = np.load(model_path)
        self._readout_W = data["readout_W"]
        self._action_low = data["action_low"]
        self._action_high = data["action_high"]
        self._obs_dim = int(data["obs_dim"][0])
        self._act_dim = int(data["act_dim"][0])
        self._init_seed = int(data["init_seed"][0])
        meta = json.loads(meta_path.read_text())
        # Restore hyperparams so the rebuilt reservoir matches the trained one.
        saved_hp = meta.get("diagnostics", {}).get("hyperparams", {})
        self.hp = {**DEFAULT_HP, **saved_hp}
        # Rebuild reservoir from init_seed (deterministic given the seed + hp).
        self._reservoir = self._make_reservoir(self._init_seed)
        _ = self._reservoir(np.zeros(self._obs_dim, dtype=np.float32))
        self._reservoir.reset()
        self._train_result = TrainResult(
            train_time_sec=meta["train_time_sec"],
            train_env_steps=meta["train_env_steps"],
            train_opt_steps=meta["train_opt_steps"],
            diagnostics=meta.get("diagnostics", {}),
        )
        return self._train_result

    def policy(self) -> Policy:
        if self._reservoir is None or self._readout_W is None:
            raise RuntimeError("policy() called before train() or load()")
        return _ESNPolicy(self._reservoir, self._readout_W,
                          self._action_low, self._action_high)

    def param_count(self) -> int:
        """Readout weights only. Reservoir is fixed at init, never trained."""
        if self._readout_W is None:
            return 0
        return int(self._readout_W.size)

    def inference_macs(self) -> int | None:
        """input_proj (obs_dim x N) + recurrent (N x N x density) + readout (N x act_dim).
        reservoirpy ops aren't fvcore-traceable, so we count by hand."""
        if self._reservoir is None or self._readout_W is None:
            return None
        N = self.hp["reservoir_size"]
        density = self.hp.get("rc_connectivity", 1.0)
        macs_input = self._obs_dim * N
        macs_recurrent = int(N * N * density)
        macs_readout = N * self._act_dim
        return macs_input + macs_recurrent + macs_readout

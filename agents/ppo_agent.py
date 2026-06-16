"""PPOAgent — Stable-Baselines3 PPO wrapped behind the AgentProtocol.

param_count: the deployed policy MLP's trainable parameters (excludes the value head).
inference_macs: fvcore.FlopCountAnalysis on the policy net (per the MAC = 1 FLOP convention).
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Callable
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from agents.base import AgentProtocol, Policy, StatelessPolicy, TrainResult


DEFAULT_HP = {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
              "gamma": 0.99, "gae_lambda": 0.95, "n_epochs": 10}


class PPOAgent:
    """SB3 PPO wrapper. Stateless at inference (MLP)."""

    def __init__(self, hyperparams: dict[str, Any] | None = None):
        self.hyperparams = {**DEFAULT_HP, **(hyperparams or {})}
        self.model: PPO | None = None
        self._train_result: TrainResult | None = None

    def train(self, env_fn: Callable[[], Any], total_timesteps: int, seed: int) -> TrainResult:
        vec_env = DummyVecEnv([env_fn])
        self.model = PPO("MlpPolicy", vec_env, seed=seed, verbose=0,
                         device="cpu", **self.hyperparams)

        t0 = time.perf_counter()
        self.model.learn(total_timesteps=total_timesteps, progress_bar=False)
        train_time = time.perf_counter() - t0

        env_steps = int(self.model.num_timesteps)
        n_steps = self.hyperparams["n_steps"]
        n_epochs = self.hyperparams["n_epochs"]
        batch_size = self.hyperparams["batch_size"]
        rollouts = env_steps // n_steps
        minibatches_per_rollout = max(n_steps // batch_size, 1)
        opt_steps = rollouts * n_epochs * minibatches_per_rollout

        self._train_result = TrainResult(
            train_time_sec=train_time,
            train_env_steps=env_steps,
            train_opt_steps=opt_steps,
            diagnostics={"hyperparams": dict(self.hyperparams)},
        )
        return self._train_result

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.model is None or self._train_result is None:
            raise RuntimeError("Cannot save: PPOAgent has not been trained yet.")
        self.model.save(path / "model.zip")
        meta = {
            "train_time_sec": self._train_result.train_time_sec,
            "train_env_steps": self._train_result.train_env_steps,
            "train_opt_steps": self._train_result.train_opt_steps,
            "diagnostics": self._train_result.diagnostics,
        }
        (path / "train_meta.json").write_text(json.dumps(meta))

    def load(self, path: Path) -> TrainResult:
        path = Path(path)
        model_path = path / "model.zip"
        meta_path = path / "train_meta.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"PPOAgent.load: missing model.zip or train_meta.json at {path}. "
                f"Refusing to produce a record with zeroed training metrics."
            )
        self.model = PPO.load(model_path, device="cpu")
        meta = json.loads(meta_path.read_text())
        self._train_result = TrainResult(
            train_time_sec=meta["train_time_sec"],
            train_env_steps=meta["train_env_steps"],
            train_opt_steps=meta["train_opt_steps"],
            diagnostics=meta.get("diagnostics", {}),
        )
        return self._train_result

    def policy(self) -> Policy:
        if self.model is None:
            raise RuntimeError("policy() called before train() or load()")
        model = self.model
        def predict(obs: np.ndarray) -> np.ndarray:
            action, _ = model.predict(obs, deterministic=True)
            return action
        return StatelessPolicy(predict)

    def param_count(self) -> int:
        """Trainable params in the deployed inference path: policy branch of the
        mlp_extractor + action head. Excludes the value branch AND value head
        (value-estimation work discarded at inference), mirroring inference_macs."""
        if self.model is None:
            return 0
        deployed_modules = [self.model.policy.mlp_extractor.policy_net, self.model.policy.action_net]
        total = 0
        for mod in deployed_modules:
            for p in mod.parameters():
                if p.requires_grad:
                    total += p.numel()
        return total

    def inference_macs(self) -> int | None:
        """MACs per forward pass (fvcore convention: 1 MAC = 1 FLOP).

        Deployed inference path: policy branch of mlp_extractor + action_net.
        Excludes value_net AND the value branch of mlp_extractor per CLAUDE.md
        S 4 - both are value-estimation work discarded at inference time.
        SB3's MlpExtractor runs pi and vf branches in parallel; counting both
        would inflate the deployed footprint by ~2x for symmetric architectures.
        """
        if self.model is None:
            return None
        try:
            from fvcore.nn import FlopCountAnalysis
            obs_dim = self.model.observation_space.shape[0]
            obs = torch.zeros((1, obs_dim))
            mlp = self.model.policy.mlp_extractor
            # MlpExtractor has policy_net and value_net submodules (each a Sequential).
            # Trace only the policy branch.
            pi_macs = int(FlopCountAnalysis(mlp.policy_net, obs).total())
            with torch.no_grad():
                latent_pi = mlp.policy_net(obs)
            head_macs = int(FlopCountAnalysis(self.model.policy.action_net, latent_pi).total())
            return pi_macs + head_macs
        except Exception:
            return None

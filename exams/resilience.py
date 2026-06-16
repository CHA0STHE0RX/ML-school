"""ResilienceExam — wraps EnvironmentCritic, emits one record per ModType."""
from __future__ import annotations
from typing import Any, Callable
import gymnasium as gym
import numpy as np
from agents.base import Policy
from critic import EnvironmentCritic, RReport
from records import (
    ExperimentRecord, ExperimentConfig, EnvironmentMod, ExamBlock, ModType, HardwareInfo,
)

S_MAX_BY_MOD = {
    ModType.FLICKER:        0.9,
    ModType.GAUSSIAN_NOISE: 1.5,
    ModType.ACTION_DELAY:   1.0,
    ModType.PHYSICS_SHIFT:  0.5,
}


class ResilienceExam:
    name = "resilience"

    def __init__(self, n_episodes: int = 5, max_iters: int = 6):
        self.n_episodes = n_episodes
        self.max_iters = max_iters

    def evaluate(
        self,
        policy: Policy,
        env_fn: Callable[[], gym.Env],
        context: dict[str, Any],
    ) -> list[ExperimentRecord]:
        """Probe each ModType. Returns one record per mod."""
        records: list[ExperimentRecord] = []
        cfg: ExperimentConfig = context["config"]

        from task_metrics import collect_env_metrics
        env_metrics = collect_env_metrics(cfg.env_id, env_fn, policy, n_episodes=self.n_episodes,
                                          seed=cfg.train_seed + 999)

        for mod_type, s_max in S_MAX_BY_MOD.items():
            critic = EnvironmentCritic(
                base_env_fn=env_fn, mod_type=mod_type, s_max=s_max,
                n_episodes=self.n_episodes, max_iters=self.max_iters,
                seed=cfg.train_seed + 1,
            )
            try:
                report: RReport = critic.probe(policy)
            except RuntimeError as e:
                records.append(self._error_record(cfg, mod_type, s_max, str(e), context, env_metrics))
                continue

            clean_pts = [p for p in report.points if p.strength == 0.0]
            clean_std = clean_pts[0].std_return if clean_pts else 0.0

            success = float(np.clip(report.aurc, 0.0, 1.0))
            adapt = float(report.s_half / report.s_max) if report.s_max > 0 else 0.0

            rec = ExperimentRecord(
                code_version=context.get("code_version", "untracked"),
                config=cfg,
                mod=EnvironmentMod(mod_type, s_max, f"resilience probe up to s_max={s_max}"),
                hardware=context.get("hardware", HardwareInfo()),
                train_time_sec=context["train_result"].train_time_sec,
                train_env_steps=context["train_result"].train_env_steps,
                train_opt_steps=context["train_result"].train_opt_steps,
                param_count=context["param_count"],
                inf_lat_ms=context["inf_lat_ms"],
                inf_macs=context.get("inf_macs"),
                inf_mem_mb=context.get("inf_mem_mb"),
                inf_gpu_mem_mb=context.get("inf_gpu_mem_mb"),
                clean_return=report.clean_return,
                clean_return_std=float(clean_std),
                success=success,
                adapt_score=adapt,
                exam=ExamBlock(
                    name=self.name,
                    config={"mod_type": mod_type.name, "s_max": s_max,
                            "n_episodes": self.n_episodes, "max_iters": self.max_iters},
                    raw={"aurc": report.aurc, "s_half": report.s_half,
                         "s_max": report.s_max, "cliff_slope": report.cliff_slope,
                         "absolute_aurc": report.absolute_aurc,
                         "fit_rmse": report.fit_rmse,
                         "clean_return": report.clean_return,
                         "points": [p.__dict__ for p in report.points]},
                    formula="success := aurc; adapt_score := s_half / s_max",
                ),
                env_metrics=env_metrics,
                diagnostics=dict(context["train_result"].diagnostics),
            )
            records.append(rec)
        return records

    def _error_record(self, cfg, mod_type, s_max, err, context, env_metrics) -> ExperimentRecord:
        return ExperimentRecord(
            code_version=context.get("code_version", "untracked"),
            config=cfg,
            mod=EnvironmentMod(mod_type, s_max, f"FAILED: {err}"),
            hardware=context.get("hardware", HardwareInfo()),
            train_time_sec=context["train_result"].train_time_sec,
            train_env_steps=context["train_result"].train_env_steps,
            train_opt_steps=context["train_result"].train_opt_steps,
            param_count=context["param_count"],
            inf_lat_ms=context["inf_lat_ms"],
            inf_macs=context.get("inf_macs"),
            inf_mem_mb=context.get("inf_mem_mb"),
            inf_gpu_mem_mb=context.get("inf_gpu_mem_mb"),
            success=0.0,
            adapt_score=None,
            exam=ExamBlock(name=self.name,
                           config={"mod_type": mod_type.name, "s_max": s_max},
                           raw={}, formula=""),
            env_metrics=env_metrics,
            diagnostics=dict(context["train_result"].diagnostics),
            notes=f"Exam failed: {err}",
        )

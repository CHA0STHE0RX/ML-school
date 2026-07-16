"""Run PPO on SwingupDoublePendulum-v0 through the real orchestrator pipeline.

run_cell builds envs via get_profile(env_id).make_fn, which lazily imports and
registers the swing-up env — no manual registration is needed here.

Output is namespaced to results/swingup_ppo/ and weights/swingup_ppo/ so it does
not touch the main results.jsonl.

    python run_swingup_ppo.py
"""
import run_experiments as rx

rx._apply_tag("swingup_ppo")
n = rx.run_cell(
    "PPO", "SwingupDoublePendulum-v0", 0, "resilience",
    total_timesteps=300_000, force_train=False, skip_training=False,
)
print(f"records written: {n}")

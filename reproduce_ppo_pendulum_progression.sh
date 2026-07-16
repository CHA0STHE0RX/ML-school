#!/usr/bin/env bash
# Reproduce results/plots/progression/progression_*_PPO_Pendulum-v1.png
# end-to-end from a clean checkout. (Bash port of the original .ps1.)
#
# Trains PPO on Pendulum-v1 across 6 budgets x 4 seeds:
#   seeds = {0, 42, 1729, 31415}  -- seed 0 is the original exploratory run;
#   the three spread-out seeds (Hardy-Ramanujan, first digits of pi) replicate
#   the progression curve for variance bounds.
#   (Mersenne Twister decorrelates adjacent seed integers, so any 4 distinct
#   ints would be statistically equivalent; the choice is aesthetic.)
#
# Runtime: ~2.5 hrs on CPU (PPO/Pendulum stays on CPU even with CUDA torch
# installed because PPOAgent forces device="cpu" -- tiny MLP on Pendulum is
# faster on CPU than GPU due to kernel-launch overhead).
#
# Produces ~96 records (4 mods x 6 budgets x 4 seeds) at results/<tag>/results.jsonl
# and figures at results/plots/progression/:
#   - progression_PPO_Pendulum-v1.png            (2x2 bundle, includes coupling panel)
#   - progression_triptych_PPO_Pendulum-v1.png   (4-row stacked: clean | AURC | absolute_aurc | hardware)
set -euo pipefail
cd "$(dirname "$0")"

# Use PY=... to point at a specific interpreter (e.g. the ML-school conda env);
# defaults to whatever `python` is on PATH (an already-activated env).
PY="${PY:-python}"

budgets=(50000 100000 250000 500000 1000000 2500000)
tags=(50k 100k 250k 500k 1m 2_5m)
seeds=(0 42 1729 31415)

for i in "${!budgets[@]}"; do
    echo "=== Training PPO @ ${budgets[$i]} steps (tag=${tags[$i]}) x ${#seeds[@]} seeds ==="
    "$PY" run_experiments.py \
        --agents PPO --envs Pendulum-v1 \
        --seeds "${seeds[@]}" \
        --timesteps "${budgets[$i]}" --tag "${tags[$i]}" --force-train
done

echo "=== Rendering progression plots ==="
"$PY" plot_progression.py

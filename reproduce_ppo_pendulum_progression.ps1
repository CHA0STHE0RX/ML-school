# Reproduce results/plots/progression/progression_*_PPO_Pendulum-v1.png
# end-to-end from a clean checkout.
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

# Prefer the project venv if present (per README: python -m venv ML-school);
# otherwise use whatever `python` is on PATH (e.g. an already-activated env).
$py = if (Test-Path "ML-school/Scripts/python.exe") { "ML-school/Scripts/python.exe" } else { "python" }
$budgets = @(50000, 100000, 250000, 500000, 1000000, 2500000)
$tags    = @("50k",  "100k",  "250k",  "500k",  "1m",   "2_5m")
$seeds   = @(0, 42, 1729, 31415)

for ($i = 0; $i -lt $budgets.Length; $i++) {
    Write-Host "=== Training PPO @ $($budgets[$i]) steps (tag=$($tags[$i])) x $($seeds.Length) seeds ==="
    & $py run_experiments.py `
        --agents PPO --envs Pendulum-v1 `
        --seeds $seeds `
        --timesteps $budgets[$i] --tag $tags[$i] --force-train
    if (-not $?) { Write-Error "Run $($tags[$i]) failed"; exit 1 }
}

Write-Host "=== Rendering progression plots ==="
& $py plot_progression.py

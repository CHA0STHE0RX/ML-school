"""Resilience-exam visualization for one (agent, env) cell. ↑↓
Produces per-mod sigmoid curves, an AURC leaderboard, and the breaking-point
scatter — all three answer the same research question: "what is this agent's
resilience signature on this env?"

Run:
    python plot_resilience.py --tag 100k
    python plot_resilience.py --in results/250k/results.jsonl --out results/250k/plots
"""
from __future__ import annotations
from plots import (
    load, resolve_paths, standard_argparser,
    draw_curves_per_mod, draw_leaderboard, draw_breaking_point,
)


def main():
    args = standard_argparser(__doc__.splitlines()[0]).parse_args()
    inp, out_dir = resolve_paths(args.tag, args.inp, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load(inp)
    print(f"Loaded {len(records)} records. Writing to {out_dir}/")
    draw_curves_per_mod(records, out_dir)
    draw_leaderboard(records, out_dir)
    draw_breaking_point(records, out_dir)


if __name__ == "__main__":
    main()

"""Cost vs capability across families — the commensurability plane. ↑↓
Places architecturally-unrelated agents on one inference-cost vs resilience
plane: the cross-family cost/capability trade-off. One research question:
"what does each family cost to run, and what resilience does that buy?"

Run:
    python plot_efficiency.py --tag 500k
    python plot_efficiency.py --in results/500k/results.jsonl --out results/500k/plots
"""
from __future__ import annotations
from plots import load, resolve_paths, standard_argparser, draw_cost_performance


def main():
    args = standard_argparser(__doc__.splitlines()[0]).parse_args()
    inp, out_dir = resolve_paths(args.tag, args.inp, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load(inp)
    print(f"Loaded {len(records)} records. Writing to {out_dir}/")
    draw_cost_performance(records, out_dir)


if __name__ == "__main__":
    main()

"""Training-progression visualization (CLAUDE.md §7.1). ↑↓
One figure per (agent, env): resilience-vs-train_env_steps (per mod),
clean_return-vs-train_env_steps, train_time_sec-vs-train_env_steps, and the
resilience ↔ clean_return coupling — all answering one research question:
"how does an agent's resilience signature evolve with training budget?"

Run:
    python plot_progression.py                              # auto-discover all results/<tag>/
    python plot_progression.py --tags 50k 100k 250k         # specific budgets only
    python plot_progression.py --out results/plots/progression
"""
from __future__ import annotations
import argparse
from pathlib import Path
from plots import (PROJECT_ROOT, load_progression,
                   draw_progression_bundle, draw_progression_triptych)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", nargs="+", default=None,
                    help="tags (subdirs under results/) to combine; omit to auto-discover")
    ap.add_argument("--out", default=None,
                    help="output directory (default: results/plots/progression/)")
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else (PROJECT_ROOT / "results" / "plots" / "progression")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_progression(args.tags)
    n_tags = len({r["_tag"] for r in records})
    print(f"Loaded {len(records)} records across {n_tags} budget(s). Writing to {out_dir}/")
    draw_progression_bundle(records, out_dir)
    draw_progression_triptych(records, out_dir)


if __name__ == "__main__":
    main()

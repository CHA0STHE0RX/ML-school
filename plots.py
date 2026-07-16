"""Shared utilities and plot primitives for every plot_*.py script. ↑↓

Each plot_<research-goal>.py composes these primitives to produce one
research output. Group by functionality (what question are we answering?),
not by what kind of chart it is.

Example: plot_resilience.py uses draw_curves + draw_leaderboard +
draw_breaking_point — all three serve one goal ("show the resilience
signature of an agent on an env"). A future plot_progression.py would
draw a different chart entirely, but is also one file because it serves
one goal.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IN = PROJECT_ROOT / "results" / "results.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "results" / "plots"


# ---------- loaders / arg parsing / path resolution ----------

def load(path: Path) -> list[dict]:
    """Load a JSONL of ExperimentRecord dicts."""
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def norm_points(raw: dict):
    """From exam.raw, return (strengths, normalized_returns) ready to plot.

    Normalizes against the critic's worst — the return at s_max (the largest
    probed strength) — so the plotted points share the normalization that
    produced the stored s_half/aurc. (min-over-points would diverge from the
    critic on a non-monotone curve.)
    """
    pts = sorted(raw["points"], key=lambda p: p["strength"])
    worst = pts[-1]["mean_return"]                 # return at s_max == critic's r_worst
    denom = max(raw["clean_return"] - worst, 1e-9)
    s = np.array([p["strength"] for p in pts])
    r = np.array([(p["mean_return"] - worst) / denom for p in pts]).clip(0, 1)
    return s, r


def load_progression(tags: list[str] | None = None) -> list[dict]:
    """Load records across multiple tagged runs. Auto-discovers all results/<tag>/results.jsonl
    if `tags` is None. Each record gets `_tag` attached so plotters can identify the budget."""
    results_dir = PROJECT_ROOT / "results"
    if tags:
        pairs = [(t, results_dir / t / "results.jsonl") for t in tags]
    else:
        pairs = sorted((p.parent.name, p) for p in results_dir.glob("*/results.jsonl"))
    out: list[dict] = []
    for tag, path in pairs:
        if not path.exists():
            print(f"  skipping tag={tag}: {path} not found")
            continue
        for rec in load(path):
            rec["_tag"] = tag
            out.append(rec)
    return out


def resolve_paths(tag: str | None, inp: str | None, out: str | None) -> tuple[Path, Path]:
    """Resolve --in/--out, with --tag as a shortcut to results/<tag>/{results.jsonl, plots/}."""
    if tag:
        return (PROJECT_ROOT / "results" / tag / "results.jsonl",
                PROJECT_ROOT / "results" / tag / "plots")
    return (Path(inp) if inp else DEFAULT_IN,
            Path(out) if out else DEFAULT_OUT)


def standard_argparser(description: str) -> argparse.ArgumentParser:
    """Argparser shared by every plot_<name>.py script."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--in", dest="inp", default=None,
                    help="path to results.jsonl (overrides --tag)")
    ap.add_argument("--out", default=None,
                    help="output directory for PNGs (overrides --tag)")
    ap.add_argument("--tag", default=None,
                    help="read results/<tag>/results.jsonl, write to results/<tag>/plots/")
    return ap


# ---------- plot primitives (drawing functions) ----------

def draw_curves_per_mod(records: list[dict], out_dir: Path) -> None:
    """One PNG per ModType: logistic fit + bisection probe points."""
    by_mod: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_mod[r["exam"]["config"]["mod_type"]].append(r)

    for mod, rows in by_mod.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in rows:
            raw = r["exam"]["raw"]
            s, rn = norm_points(raw)
            label = f'{r["config"]["agent_name"]} (s½={raw["s_half"]:.2f}, AURC={raw["aurc"]:.2f})'
            grid = np.linspace(0, raw["s_max"], 200)
            if raw.get("fit_method") == "pchip_fallback":
                # This record's aurc/s_half came from the shape-agnostic PCHIP
                # fallback (logistic misfit) — draw that same shape, dashed;
                # drawing the rejected logistic would not match the stored aurc.
                su, idx = np.unique(s, return_index=True)
                fit = np.clip(PchipInterpolator(su, rn[idx])(grid), 0.0, 1.0)
                line, = ax.plot(grid, fit, lw=2, ls="--", label=label + " [pchip]")
            else:
                z = np.clip(raw["cliff_slope"] * (grid - raw["s_half"]), -50.0, 50.0)
                fit = 1.0 / (1.0 + np.exp(z))
                line, = ax.plot(grid, fit, lw=2, label=label)
            ax.scatter(s, rn, s=28, color=line.get_color(), edgecolor="white", zorder=3)
            ax.axvline(raw["s_half"], color=line.get_color(), ls=":", alpha=0.4)
        ax.axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.5)
        ax.set_xlabel("perturbation strength")
        ax.set_ylabel("normalized return")
        ax.set_title(f"Resilience under {mod}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.2)
        out = out_dir / f"curve_{mod.lower()}.png"
        fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
        print(f"  wrote {out}")


def draw_leaderboard(records: list[dict], out_dir: Path) -> None:
    """Grouped bar: AURC per (agent, mod). ↑ the better."""
    agents = sorted({r["config"]["agent_name"] for r in records})
    mods = sorted({r["exam"]["config"]["mod_type"] for r in records})
    # Mean over all records per (agent, mod) — a file can hold several seeds;
    # keeping only one of them (last-writer-wins) would misstate the leaderboard.
    data: dict[str, dict[str, list[float]]] = {a: {m: [] for m in mods} for a in agents}
    for r in records:
        data[r["config"]["agent_name"]][r["exam"]["config"]["mod_type"]].append(
            r["exam"]["raw"]["aurc"])

    x = np.arange(len(mods))
    w = 0.8 / max(len(agents), 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(mods)), 4))
    for i, a in enumerate(agents):
        ax.bar(x + i * w, [float(np.mean(data[a][m])) if data[a][m] else 0.0 for m in mods],
               width=w, label=a)
    ax.set_xticks(x + w * (len(agents) - 1) / 2)
    ax.set_xticklabels(mods, rotation=20, ha="right")
    ax.set_ylabel("AURC ; ↑ the better")
    ax.set_title("Leaderboard - AURC per perturbation")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.2)
    out = out_dir / "leaderboard_aurc.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


def _agg_by_budget(rows: list[dict], value_fn) -> tuple[list[float], list[float], list[float]]:
    """Group rows by _tag, return (xs, means, stds) sorted by mean train_env_steps.
    Lets multi-seed progression plots show mean+/-std bands instead of N tangled lines."""
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tag[r.get("_tag", "_default")].append(r)
    triples = []
    for tag, group in by_tag.items():
        xs = [r["train_env_steps"] for r in group]
        ys = [value_fn(r) for r in group]
        triples.append((float(np.mean(xs)), float(np.mean(ys)), float(np.std(ys))))
    triples.sort()
    return ([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples])


def draw_progression_bundle(records: list[dict], out_dir: Path) -> None:
    """One 2x2 figure per (agent, env): the §7.1 progression-curve bundle.

    Top-left:  AURC (resilience) vs train_env_steps — one line per ModType.
    Top-right: clean_return vs train_env_steps.
    Bottom-L:  train_time_sec vs train_env_steps (training cost).
    Bottom-R:  resilience ↔ clean_return parametric trajectory (one line per mod).
    """
    if not records:
        print("  no records — skipping progression bundle")
        return

    bundles: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in records:
        key = (r["config"]["agent_name"], r["config"]["env_id"],
               r["exam"]["config"]["mod_type"])
        bundles[key].append(r)
    for key in bundles:
        bundles[key].sort(key=lambda r: r["train_env_steps"])

    by_agent_env: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(dict)
    for (a, e, m), rows in bundles.items():
        by_agent_env[(a, e)][m] = rows

    for (agent, env), mod_to_rows in sorted(by_agent_env.items()):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle(f"Training progression — {agent} × {env}", fontsize=12)
        ax_res, ax_clean, ax_cost, ax_couple = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        for mod, rows in sorted(mod_to_rows.items()):
            xs, means, stds = _agg_by_budget(rows, lambda r: r["exam"]["raw"]["aurc"])
            line, = ax_res.plot(xs, means, "o-", label=mod, lw=1.8)
            ax_res.fill_between(xs, np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                color=line.get_color(), alpha=0.15)
        ax_res.set_xscale("log")
        ax_res.set_xlabel("train_env_steps"); ax_res.set_ylabel("AURC (resilience)")
        ax_res.set_title("Resilience progression — per mod (mean ±1σ across seeds)")
        ax_res.set_ylim(0, 1); ax_res.legend(fontsize=8); ax_res.grid(alpha=0.2)

        # clean_return / train_time are agent-level; aggregate across all mod rows.
        all_rows = [r for rows in mod_to_rows.values() for r in rows]
        xs, mc, sc = _agg_by_budget(all_rows, lambda r: r["clean_return"])
        ax_clean.plot(xs, mc, "o-", color="black", lw=1.8)
        ax_clean.fill_between(xs, np.array(mc) - np.array(sc),
                              np.array(mc) + np.array(sc), color="black", alpha=0.12)
        ax_clean.set_xscale("log")
        ax_clean.set_xlabel("train_env_steps"); ax_clean.set_ylabel("clean_return (env units)")
        ax_clean.set_title("Clean baseline progression (mean ±1σ across seeds)")
        ax_clean.grid(alpha=0.2)

        xs, mt, st = _agg_by_budget(all_rows, lambda r: r["train_time_sec"])
        ax_cost.plot(xs, mt, "o-", color="darkred", lw=1.8)
        ax_cost.fill_between(xs, np.array(mt) - np.array(st),
                             np.array(mt) + np.array(st), color="darkred", alpha=0.15)
        ax_cost.set_xscale("log"); ax_cost.set_yscale("log")
        ax_cost.set_xlabel("train_env_steps"); ax_cost.set_ylabel("train_time_sec")
        ax_cost.set_title("Training cost (wall clock)"); ax_cost.grid(alpha=0.2)

        for mod, rows in sorted(mod_to_rows.items()):
            _, mean_clean, _ = _agg_by_budget(rows, lambda r: r["clean_return"])
            _, mean_aurc, _ = _agg_by_budget(rows, lambda r: r["exam"]["raw"]["aurc"])
            ax_couple.plot(mean_clean, mean_aurc, "o-", label=mod, lw=1.4, alpha=0.85)
        ax_couple.set_xlabel("clean_return"); ax_couple.set_ylabel("AURC (resilience)")
        ax_couple.set_title("Coupling: resilience vs clean_return\n(each marker = one training budget; lines connect in budget order)")
        ax_couple.set_ylim(0, 1); ax_couple.legend(fontsize=8); ax_couple.grid(alpha=0.2)

        fig.tight_layout()
        slug = f"{agent}_{env}".replace("/", "_")
        out = out_dir / f"progression_{slug}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"  wrote {out}")


def draw_progression_triptych(records: list[dict], out_dir: Path) -> None:
    """One 3-row figure per (agent, env): clean / mod returns / hardware, shared x-axis.

    Stacked vertically (not side-by-side) so peaks and troughs at the same
    train_env_steps value line up in a column — read straight down from any budget
    to see what was happening to clean_return, resilience, and cost simultaneously.
    Vertical gridlines mark every probed budget.
    """
    if not records:
        print("  no records — skipping progression triptych")
        return

    bundles: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in records:
        key = (r["config"]["agent_name"], r["config"]["env_id"],
               r["exam"]["config"]["mod_type"])
        bundles[key].append(r)
    for key in bundles:
        bundles[key].sort(key=lambda r: r["train_env_steps"])

    by_agent_env: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(dict)
    for (a, e, m), rows in bundles.items():
        by_agent_env[(a, e)][m] = rows

    for (agent, env), mod_to_rows in sorted(by_agent_env.items()):
        fig, (ax_clean, ax_res, ax_abs, ax_hw) = plt.subplots(
            4, 1, figsize=(10, 12), sharex=True,
            gridspec_kw={"hspace": 0.12})
        fig.suptitle(f"Progression triptych: {agent} × {env}", fontsize=13)

        all_rows = [r for rows in mod_to_rows.values() for r in rows]
        budgets, mc, sc = _agg_by_budget(all_rows, lambda r: r["clean_return"])
        ax_clean.plot(budgets, mc, "o-", color="black", lw=1.8)
        ax_clean.fill_between(budgets, np.array(mc) - np.array(sc),
                              np.array(mc) + np.array(sc), color="black", alpha=0.12)
        ax_clean.set_ylabel("clean_return (env units)")
        ax_clean.set_title("Clean baseline (mean ±1σ across seeds)", loc="left", fontsize=10)
        ax_clean.grid(alpha=0.2)

        for mod, rows in sorted(mod_to_rows.items()):
            xs, m, s = _agg_by_budget(rows, lambda r: r["exam"]["raw"]["aurc"])
            line, = ax_res.plot(xs, m, "o-", label=mod, lw=1.8)
            ax_res.fill_between(xs, np.array(m) - np.array(s),
                                np.array(m) + np.array(s),
                                color=line.get_color(), alpha=0.15)
        ax_res.set_ylabel("AURC (intra-policy, ∈[0,1])")
        ax_res.set_title("Mod returns: AURC (normalized per row, can mislead across budgets)",
                         loc="left", fontsize=10)
        ax_res.set_ylim(0, 1)
        ax_res.legend(fontsize=8, loc="lower right", ncol=2)
        ax_res.grid(alpha=0.2)

        for mod, rows in sorted(mod_to_rows.items()):
            xs, m, s = _agg_by_budget(rows, lambda r: r["exam"]["raw"]["absolute_aurc"])
            line, = ax_abs.plot(xs, m, "o-", label=mod, lw=1.8)
            ax_abs.fill_between(xs, np.array(m) - np.array(s),
                                np.array(m) + np.array(s),
                                color=line.get_color(), alpha=0.15)
        ax_abs.set_ylabel("absolute_aurc (env reward units)")
        ax_abs.set_title("Mod returns: absolute_aurc (raw env units, cross-budget honest)",
                         loc="left", fontsize=10)
        ax_abs.legend(fontsize=8, loc="lower right", ncol=2)
        ax_abs.grid(alpha=0.2)

        _, mt, st = _agg_by_budget(all_rows, lambda r: r["train_time_sec"])
        _, ml, sl = _agg_by_budget(all_rows, lambda r: r["inf_lat_ms"])
        _, mm, sm = _agg_by_budget(all_rows, lambda r: r["inf_mem_mb"])
        ax_hw.plot(budgets, mt, "o-", color="darkred", lw=1.8, label="train_time_sec")
        ax_hw.plot(budgets, ml, "s-", color="steelblue", lw=1.6, label="inf_lat_ms")
        ax_hw.plot(budgets, mm, "^-", color="seagreen", lw=1.4, label="inf_mem_mb")
        ax_hw.set_yscale("log")
        ax_hw.set_ylabel("hardware use (mixed units, log)")
        ax_hw.set_title("Hardware use (mean across seeds)", loc="left", fontsize=10)
        ax_hw.set_xlabel("train_env_steps")
        ax_hw.legend(fontsize=8, loc="center right")
        ax_hw.grid(alpha=0.2, which="both")

        ax_hw.set_xscale("log")
        ax_hw.set_xticks(budgets)
        ax_hw.set_xticklabels([f"{b/1000:g}k" if b < 1_000_000 else f"{b/1_000_000:g}M"
                               for b in budgets])
        for ax in (ax_clean, ax_res, ax_abs, ax_hw):
            for b in budgets:
                ax.axvline(b, color="grey", ls=":", lw=0.6, alpha=0.5)

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        slug = f"{agent}_{env}".replace("/", "_")
        out = out_dir / f"progression_triptych_{slug}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"  wrote {out}")


def draw_cost_performance(records: list[dict], out_dir: Path) -> None:
    """Scatter: inference cost (x, log) vs resilience capability (y), one marker per agent.

    Places architecturally-unrelated families on one cost/capability plane — the
    cross-family trade the framework exists to make visible. x is inf_macs (fixed
    by architecture, not training budget); y is mean absolute_aurc across this
    file's mods+seeds (raw env-reward units, cross-policy honest, ↑ better).
    """
    if not records:
        print("  no records — skipping cost/performance plane")
        return
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_agent[r["config"]["agent_name"]].append(r)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, (agent, rows) in enumerate(sorted(by_agent.items())):
        macs = [r["inf_macs"] for r in rows if r.get("inf_macs")]
        perf = [r["exam"]["raw"]["absolute_aurc"] for r in rows]
        if not macs:
            print(f"  {agent}: no inf_macs — skipping")
            continue
        x = float(np.median(macs))                       # architecture-fixed
        y, ystd = float(np.mean(perf)), float(np.std(perf))
        params = int(np.median([r["param_count"] for r in rows]))
        ax.errorbar(x, y, yerr=ystd, fmt="o", ms=7, color=cmap(i),
                    capsize=4, lw=1.6, label=agent, zorder=3)
        ax.annotate(f"{agent}\n{params:,} params · {int(x):,} MACs",
                    (x, y), textcoords="offset points", xytext=(12, 6),
                    fontsize=8, color=cmap(i))
    ax.set_xscale("log")
    ax.set_xlabel("inf_macs: inference cost (log; ← cheaper)")
    ax.set_ylabel("mean absolute_aurc (env reward units; ↑ more resilient)")
    ax.set_title("Cost vs resilience: one plane, two families")
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=8, loc="lower right")
    out = out_dir / "cost_performance.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


def draw_breaking_point(records: list[dict], out_dir: Path) -> None:
    """Scatter: x=s½/s_max (when it breaks), y=cliff_slope (how it breaks). ↑ cliff, ↓ !cliff."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    markers = {"FLICKER": "o", "GAUSSIAN_NOISE": "s", "ACTION_DELAY": "^", "PHYSICS_SHIFT": "D"}
    agents = sorted({r["config"]["agent_name"] for r in records})
    cmap = plt.get_cmap("tab10")
    color = {a: cmap(i) for i, a in enumerate(agents)}
    for r in records:
        raw = r["exam"]["raw"]
        ax.scatter(raw["s_half"] / raw["s_max"], raw["cliff_slope"],
                   s=70, marker=markers.get(r["exam"]["config"]["mod_type"], "o"),
                   color=color[r["config"]["agent_name"]], edgecolor="black", lw=0.6)
    # legends
    for a, c in color.items():
        ax.scatter([], [], color=c, label=a, s=69, edgecolor="black", lw=0.6)
    for m, mk in markers.items():
        ax.scatter([], [], marker=mk, color="grey", label=m, s=70, edgecolor="black", lw=0.6)
    ax.set_xlabel("breaking point (s½ / s_max)")
    ax.set_ylabel("slope k (↑ cliff, ↓ !cliff)")
    ax.set_title("Breaking point vs failure shape")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(alpha=0.2)
    out = out_dir / "breaking_point.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")

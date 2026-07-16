"""gui.py -- generate a zero-dependency local GUI for the ML-school framework.

One command, one output:

    python gui.py            # writes results/gui.html

Open results/gui.html in any browser (double-click / file:// -- no server,
no packages, works identically on the Windows and Linux side).

Four tabs:
  * Run builder -- compose run_experiments.py / plot_progression.py commands
    from dropdowns and copy the exact CLI string (the terminal stays the
    thing that runs it -- the command IS the reproducibility receipt).
  * Results -- every record across all results/<tag>/results.jsonl in one
    sortable, filterable table; click a row for the fitted resilience curve,
    the formula receipt, and the full record JSON.
  * Plots -- progression chart (metric vs training budget, one line per mod)
    rendered client-side from the embedded records.
  * Watch -- gallery of results/render/*.gif plus a watch.py command builder;
    a record's detail view can prefill it at that record's own s_half.

The HTML embeds a snapshot of the data, so re-run `python gui.py` after new
experiment runs or new renders. Dropdown contents (agents, envs, mods, exams)
are parsed from the source via `ast` at generation time -- no heavy imports,
and the GUI can never drift from the code.

Stdlib only. All CSS/JS inline; GIFs referenced by relative path (the file
lives in results/ so render/ sits next to it).
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RENDER_DIR = RESULTS_DIR / "render"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DEFAULT_OUT = RESULTS_DIR / "gui.html"


# --------------------------------------------------------------------------
# source introspection (ast -- no torch/mujoco imports at generation time)
# --------------------------------------------------------------------------

def _assigned_value(path: Path, varname: str) -> ast.expr | None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        if varname in names:
            return node.value
    return None


def _dict_keys(path: Path, varname: str) -> list[str]:
    node = _assigned_value(path, varname)
    if isinstance(node, ast.Dict):
        return [k.value for k in node.keys if isinstance(k, ast.Constant)]
    return []


def _enum_members(path: Path, classname: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == classname:
            out = []
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    out += [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            return out
    return []


def collect_meta() -> dict:
    meta = {
        "roster": {"agents": ["PPO", "ESN"], "envs": ["Pendulum-v1"],
                   "seeds": [0], "exams": ["resilience"], "timesteps_default": 100_000},
        "agents": ["PPO", "ESN"],
        "envs": ["Pendulum-v1", "SwingupDoublePendulum-v0"],
        "exams": ["resilience"],
        "mods": ["FLICKER", "PHYSICS_SHIFT", "GAUSSIAN_NOISE", "ACTION_DELAY"],
    }
    try:
        roster_node = _assigned_value(PROJECT_ROOT / "run_experiments.py", "ROSTER")
        if roster_node is not None:
            meta["roster"] = ast.literal_eval(roster_node)
        agents = _dict_keys(PROJECT_ROOT / "run_experiments.py", "AGENT_FACTORIES")
        exams = _dict_keys(PROJECT_ROOT / "run_experiments.py", "EXAM_FACTORIES")
        envs = _dict_keys(PROJECT_ROOT / "env_profiles.py", "PROFILES")
        mods = [m for m in _enum_members(PROJECT_ROOT / "records.py", "ModType")
                if m != "NONE"]
        if agents:
            meta["agents"] = agents
        if exams:
            meta["exams"] = exams
        if envs:
            meta["envs"] = envs
        if mods:
            meta["mods"] = mods
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"warning: source introspection failed ({exc}); using fallback lists")
    return meta


# --------------------------------------------------------------------------
# data collection
# --------------------------------------------------------------------------

def _tag_key(tag: str) -> tuple:
    """Sort budget-like tags numerically (50k < 100k < ... < 2_5m), rest after."""
    import re
    m = re.fullmatch(r"(?:(?P<prefix>[a-z_]+?)_)?(?P<num>\d+(?:_\d+)?)(?P<unit>[km])",
                     tag.lower())
    if not m:
        return (2, tag, 0.0)
    num = float(m.group("num").replace("_", "."))
    num *= 1e6 if m.group("unit") == "m" else 1e3
    return (0 if m.group("prefix") is None else 1, m.group("prefix") or "", num)


def load_records() -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    tags: list[str] = []
    sources: list[tuple[str, Path]] = []
    if (RESULTS_DIR / "results.jsonl").exists():
        sources.append(("", RESULTS_DIR / "results.jsonl"))
    if RESULTS_DIR.exists():
        for d in RESULTS_DIR.iterdir():
            f = d / "results.jsonl"
            if d.is_dir() and f.exists():
                sources.append((d.name, f))
    sources.sort(key=lambda s: _tag_key(s[0]))
    bad = 0
    for tag, f in sources:
        n = 0
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            rec["_tag"] = tag
            records.append(rec)
            n += 1
        if n:
            tags.append(tag)
    if bad:
        print(f"warning: skipped {bad} unparseable line(s)")
    return records, tags


def scan_renders() -> list[str]:
    if not RENDER_DIR.exists():
        return []
    return sorted(p.name for p in RENDER_DIR.iterdir()
                  if p.suffix.lower() in {".gif", ".png"})


def scan_weights() -> list[str]:
    if not WEIGHTS_DIR.exists():
        return []
    out = [p.relative_to(PROJECT_ROOT).as_posix()
           for p in WEIGHTS_DIR.glob("**/seed*") if p.is_dir()]
    return sorted(out)


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------
# Placeholders %%NAME%% are substituted by generate(). The palette (light +
# dark categorical slots, chrome/ink tokens) is the validated dataviz
# reference palette; mod->color assignment is fixed by name, never cycled.

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML-school console</title>
<style>
:root {
  --page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --ink-mut: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --accent: #2a78d6;
  --mod-FLICKER: #2a78d6; --mod-GAUSSIAN_NOISE: #1baf7a;
  --mod-PHYSICS_SHIFT: #eda100; --mod-ACTION_DELAY: #008300;
  --mod-NONE: #898781;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-mut: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --accent: #3987e5;
    --mod-FLICKER: #3987e5; --mod-GAUSSIAN_NOISE: #199e70;
    --mod-PHYSICS_SHIFT: #c98500; --mod-ACTION_DELAY: #008300;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1240px; margin: 0 auto; padding: 20px 24px 64px; }
header { color: var(--ink-mut); font-size: 12.5px; margin-bottom: 10px; }
header strong { color: var(--ink-2); }
header code { font-size: 12px; }
code, pre {
  font-family: ui-monospace, "Cascadia Mono", Consolas, Menlo, monospace;
}
.tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--grid); margin-bottom: 20px; }
.tabs button {
  appearance: none; border: none; background: none; color: var(--ink-2);
  font: inherit; padding: 8px 14px; cursor: pointer;
  border-bottom: 2px solid transparent; margin-bottom: -1px;
}
.tabs button[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--accent); font-weight: 600; }
.tabs button:hover { color: var(--ink); }
section.tab { display: none; }
section.tab.active { display: block; }
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px 18px; margin-bottom: 16px;
}
.card h2 { font-size: 15px; margin: 0 0 10px; }
.card h3 { font-size: 13px; margin: 14px 0 6px; color: var(--ink-2); }
.hint { color: var(--ink-mut); font-size: 12.5px; }
.row { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: flex-end; margin: 8px 0; }
.field { display: flex; flex-direction: column; gap: 3px; font-size: 12.5px; color: var(--ink-2); }
.field > span { font-weight: 500; }
input[type=text], input[type=number], select {
  font: inherit; color: var(--ink); background: var(--page);
  border: 1px solid var(--axis); border-radius: 6px; padding: 5px 8px; min-width: 110px;
}
input[type=text].wide { min-width: 340px; }
.checks { display: flex; flex-wrap: wrap; gap: 4px 14px; }
.checks label { display: inline-flex; align-items: center; gap: 5px; color: var(--ink); font-size: 13px; }
.cmd {
  position: relative; background: var(--page); border: 1px solid var(--grid);
  border-radius: 8px; padding: 12px 96px 12px 12px; margin-top: 10px;
  font-family: ui-monospace, "Cascadia Mono", Consolas, Menlo, monospace;
  font-size: 12.5px; white-space: pre-wrap; word-break: break-all;
}
.copy {
  position: absolute; top: 8px; right: 8px; font: 12px system-ui, sans-serif;
  color: var(--ink-2); background: var(--surface); border: 1px solid var(--axis);
  border-radius: 6px; padding: 3px 10px; cursor: pointer;
}
.copy:hover { color: var(--ink); }
.oneliners { display: grid; gap: 6px; }
.oneliners .cmd { margin-top: 0; }
.oneliners .desc { color: var(--ink-mut); font-size: 12px; margin: 2px 0 0 2px; }
/* results table */
.filters { display: flex; flex-wrap: wrap; gap: 8px 12px; margin-bottom: 12px; align-items: flex-end; }
.tablewrap { overflow: auto; max-height: 70vh; border: 1px solid var(--grid); border-radius: 8px; }
table.results { border-collapse: collapse; width: 100%; font-size: 12.5px; }
table.results th, table.results td { padding: 6px 10px; text-align: left; white-space: nowrap; }
table.results th {
  position: sticky; top: 0; background: var(--surface); color: var(--ink-2);
  font-weight: 600; cursor: pointer; user-select: none;
  border-bottom: 1px solid var(--axis); z-index: 1;
}
table.results td { border-bottom: 1px solid var(--grid); }
table.results td.num, table.results th.num { text-align: right; font-variant-numeric: tabular-nums; }
table.results tbody tr.rec:hover { background: color-mix(in srgb, var(--accent) 7%, transparent); cursor: pointer; }
tr.detail > td { background: var(--surface); padding: 14px 16px; white-space: normal; }
.detail-grid { display: grid; grid-template-columns: minmax(320px, 620px) minmax(260px, 1fr); gap: 18px; }
@media (max-width: 900px) { .detail-grid { grid-template-columns: 1fr; } }
.facts { font-size: 12.5px; }
.facts dt { color: var(--ink-mut); float: left; clear: left; width: 120px; }
.facts dd { margin: 0 0 3px 128px; font-variant-numeric: tabular-nums; }
.facts .formula { font-family: ui-monospace, Consolas, monospace; font-size: 11.5px; }
details.json { margin-top: 10px; }
details.json pre {
  max-height: 320px; overflow: auto; background: var(--page);
  border: 1px solid var(--grid); border-radius: 8px; padding: 10px; font-size: 11px;
}
.btn {
  font: 12.5px system-ui, sans-serif; color: var(--ink); background: var(--surface);
  border: 1px solid var(--axis); border-radius: 6px; padding: 4px 12px; cursor: pointer;
}
.btn:hover { border-color: var(--ink-2); }
/* charts */
.chartbox svg { max-width: 100%; height: auto; display: block; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 6px 0 4px; font-size: 12.5px; }
.legend .key { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; color: var(--ink-2); }
.legend .key .sw { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
.legend .key .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.legend .key.off { opacity: 0.35; text-decoration: line-through; }
#tip {
  position: absolute; display: none; pointer-events: none; z-index: 10;
  background: var(--surface); color: var(--ink); border: 1px solid var(--axis);
  border-radius: 7px; padding: 7px 10px; font-size: 12px; max-width: 280px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.18);
}
#tip .t { font-weight: 600; margin-bottom: 2px; }
#tip .m { color: var(--ink-2); font-variant-numeric: tabular-nums; }
.caption { color: var(--ink-mut); font-size: 12px; margin-top: 6px; }
.empty { color: var(--ink-mut); padding: 24px 0; text-align: center; }
/* watch gallery */
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 14px; }
.gallery figure {
  margin: 0; background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 8px; overflow: hidden;
}
.gallery img { width: 100%; border-radius: 6px; display: block; background: var(--page); }
.gallery figcaption {
  font-size: 11.5px; color: var(--ink-2); margin-top: 6px; word-break: break-all;
  font-family: ui-monospace, Consolas, monospace;
}
.note-flash { color: var(--ink-2); font-size: 12.5px; margin-top: 6px; }
</style>
</head>
<body>
<div class="wrap">
<header><strong>ML-school console</strong> · snapshot %%GENERATED%% · %%N_RECORDS%% records ·
  %%N_TAGS%% tags · %%N_RENDERS%% renders · regenerate with <code>python gui.py</code></header>

<nav class="tabs" role="tablist">
  <button role="tab" data-tab="run" aria-selected="true">Run builder</button>
  <button role="tab" data-tab="results" aria-selected="false">Results</button>
  <button role="tab" data-tab="plots" aria-selected="false">Plots</button>
  <button role="tab" data-tab="watch" aria-selected="false">Watch</button>
</nav>

<!-- ============================== RUN BUILDER ============================ -->
<section class="tab active" id="tab-run">
  <div class="card">
    <h2>Interpreter</h2>
    <div class="row">
      <label class="field"><span>python</span>
        <select id="py-select">
          <option value="python">python (active env)</option>
          <option value="~/miniconda3/envs/ML-school/bin/python">Linux conda env</option>
          <option value="ML-school\Scripts\python.exe">Windows venv</option>
        </select>
      </label>
      <span class="hint">used as the prefix of every command below</span>
    </div>
  </div>

  <div class="card" id="run-form">
    <h2>run_experiments.py</h2>
    <h3>agents</h3><div class="checks" id="rb-agents"></div>
    <h3>envs</h3><div class="checks" id="rb-envs"></div>
    <h3>exams</h3><div class="checks" id="rb-exams"></div>
    <div class="row">
      <label class="field"><span>seeds (space-separated)</span>
        <input type="text" id="rb-seeds" value="%%ROSTER_SEEDS%%"></label>
      <label class="field"><span>timesteps</span>
        <input type="number" id="rb-timesteps" placeholder="%%TIMESTEPS_DEFAULT%% (default)"></label>
      <label class="field"><span>tag</span>
        <input type="text" id="rb-tag" placeholder="e.g. 500k"></label>
      <label class="field"><span>max failures</span>
        <input type="number" id="rb-maxfail" placeholder="unlimited"></label>
    </div>
    <div class="checks">
      <label><input type="checkbox" id="rb-skip"> --skip-training (reuse saved weights)</label>
      <label><input type="checkbox" id="rb-force-train"> --force-train</label>
      <label><input type="checkbox" id="rb-force-eval"> --force-eval (ignore run_log resume)</label>
      <label><input type="checkbox" id="rb-dry"> --dry-run (print plan only)</label>
    </div>
    <div class="cmd" id="rb-cmd"></div>
    <div class="hint" id="rb-note"></div>
  </div>

  <div class="card" id="pp-form">
    <h2>plot_progression.py</h2>
    <h3>--tags (order = x-axis order)</h3><div class="checks" id="pp-tags"></div>
    <div class="row">
      <label class="field"><span>--out (optional)</span>
        <input type="text" id="pp-out" placeholder="results/plots/progression"></label>
    </div>
    <div class="cmd" id="pp-cmd"></div>
  </div>

  <div class="card">
    <h2>one-liners</h2>
    <div class="oneliners" id="oneliners"></div>
  </div>
</section>

<!-- ============================== RESULTS ================================ -->
<section class="tab" id="tab-results">
  <div class="card">
    <div class="filters">
      <label class="field"><span>tag</span><select id="f-tag"></select></label>
      <label class="field"><span>agent</span><select id="f-agent"></select></label>
      <label class="field"><span>env</span><select id="f-env"></select></label>
      <label class="field"><span>mod</span><select id="f-mod"></select></label>
      <label class="field"><span>search</span>
        <input type="text" id="f-q" placeholder="id, notes, anything…"></label>
      <span class="hint" id="f-count"></span>
    </div>
    <div class="tablewrap">
      <table class="results">
        <thead><tr id="results-head"></tr></thead>
        <tbody id="results-body"></tbody>
      </table>
    </div>
    <div class="caption">click a column header to sort, a row to open its resilience
      curve + full record. AURC is intra-policy; use abs AURC for cross-budget or
      cross-agent comparisons.</div>
  </div>
</section>

<!-- ============================== PLOTS ================================== -->
<section class="tab" id="tab-plots">
  <div class="card">
    <h2>Progression, metric vs training budget, one line per mod</h2>
    <div class="row">
      <label class="field"><span>env</span><select id="p-env"></select></label>
      <label class="field"><span>agent</span><select id="p-agent"></select></label>
      <label class="field"><span>metric</span>
        <select id="p-metric">
          <option value="absolute_aurc" selected>absolute_aurc (cross-policy, honest across budgets)</option>
          <option value="success">AURC / success (intra-policy)</option>
          <option value="clean_return">clean_return</option>
          <option value="s_half">s_half</option>
          <option value="cliff_slope">cliff_slope</option>
          <option value="inf_lat_ms">inf_lat_ms</option>
          <option value="train_time_sec">train_time_sec</option>
        </select></label>
    </div>
    <div class="legend" id="p-legend"></div>
    <div class="chartbox" id="p-chart"></div>
    <div class="caption" id="p-caption"></div>
  </div>
</section>

<!-- ============================== WATCH ================================== -->
<section class="tab" id="tab-watch">
  <div class="card">
    <h2>watch.py: render a policy to a GIF</h2>
    <div class="hint">needs the optional <code>imageio</code> package. Run the command
      in a terminal, then <code>python gui.py</code> again to refresh the gallery below.</div>
    <div class="row">
      <label class="field"><span>mode</span>
        <select id="w-mode">
          <option value="single">single policy (--weights)</option>
          <option value="progression">budget progression (--progression)</option>
        </select></label>
      <label class="field"><span>env</span><select id="w-env"></select></label>
      <label class="field"><span>mod</span><select id="w-mod"></select></label>
      <label class="field"><span>strength</span>
        <input type="number" id="w-strength" step="0.05" value="0"></label>
      <label class="field"><span>seed</span><input type="number" id="w-seed" value="0"></label>
      <label class="field"><span>steps</span><input type="number" id="w-steps" value="300"></label>
    </div>
    <div class="row" id="w-single-row">
      <label class="field"><span>weights (scanned from weights/)</span>
        <select id="w-weights"></select></label>
      <label class="field"><span>or free-text path (overrides)</span>
        <input type="text" id="w-weights-free" class="wide" placeholder="weights/…/seedN"></label>
    </div>
    <div id="w-prog-row" style="display:none">
      <h3 style="font-size:13px;color:var(--ink-2);margin:6px 0">tags (one GIF each)</h3>
      <div class="checks" id="w-tags"></div>
    </div>
    <div class="cmd" id="w-cmd"></div>
    <div class="note-flash" id="w-note"></div>
  </div>

  <div class="card">
    <h2>rendered rollouts: results/render/</h2>
    <div id="w-gallery"></div>
  </div>
</section>

</div>
<div id="tip"></div>

<script>
"use strict";
const DATA = {
  records: %%RECORDS%%,
  meta: %%META%%,
  tags: %%TAGS%%,
  renders: %%RENDERS%%,
  weights: %%WEIGHTS%%
};

/* ---------------------------- tiny helpers ---------------------------- */
const $  = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const isNum = v => typeof v === "number" && Number.isFinite(v);
const fmt = (v, d = 2) => isNum(v) ? v.toFixed(d) : "—";
const fmtInt = v => isNum(v) ? v.toLocaleString("en-US") : "—";
const compact = v => {
  if (!isNum(v)) return "—";
  const trim = x => String(Math.round(x * 100) / 100);
  if (Math.abs(v) >= 1e6) return trim(v / 1e6) + "M";
  if (Math.abs(v) >= 1e3) return trim(v / 1e3) + "k";
  return trim(v);
};
const tagLabel = t => t === "" ? "(untagged)" : t;
const modVar = m => `var(--mod-${DATA.meta.mods.includes(m) ? m : "NONE"})`;

async function copyText(text, btn) {
  try { await navigator.clipboard.writeText(text); }
  catch (e) {
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta);
    ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (btn) {
    const old = btn.textContent;
    btn.textContent = "copied ✓";
    setTimeout(() => { btn.textContent = old; }, 1200);
  }
}
function attachCopy(box, getText) {
  let btn = box.querySelector(".copy");
  if (!btn) {
    btn = document.createElement("button");
    btn.className = "copy"; btn.textContent = "copy";
    btn.addEventListener("click", ev => { ev.stopPropagation(); copyText(getText(), btn); });
    box.appendChild(btn);
  }
}
function setCmd(boxId, text) {
  const box = $(boxId);
  let code = box.querySelector(".cmdtext");
  if (!code) { code = document.createElement("span"); code.className = "cmdtext"; box.prepend(code); }
  code.textContent = text;
  attachCopy(box, () => box.querySelector(".cmdtext").textContent);
}

/* nice ticks */
function niceNum(x, round) {
  const e = Math.floor(Math.log10(x)), f = x / 10 ** e;
  let nf;
  if (round) nf = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10;
  else nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  return nf * 10 ** e;
}
function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const step = niceNum(niceNum(hi - lo, false) / (n - 1), true);
  const t0 = Math.ceil(lo / step) * step, out = [];
  for (let t = t0; t <= hi + step * 1e-6; t += step) out.push(+t.toFixed(10));
  return out;
}

/* shared tooltip */
const tip = $("#tip");
function showTip(html, ev) {
  tip.innerHTML = html; tip.style.display = "block";
  const pad = 14, w = tip.offsetWidth;
  let x = ev.pageX + pad;
  if (x + w > document.documentElement.clientWidth + window.scrollX - 8) x = ev.pageX - w - pad;
  tip.style.left = x + "px";
  tip.style.top = (ev.pageY + pad) + "px";
}
function hideTip() { tip.style.display = "none"; }
function bindTips(root) {
  $$("[data-tip]", root).forEach(el => {
    el.addEventListener("mousemove", ev => showTip(el.dataset.tip, ev));
    el.addEventListener("mouseleave", hideTip);
  });
}

/* ------------------------------- tabs --------------------------------- */
function showTab(name) {
  $$(".tabs button").forEach(b => b.setAttribute("aria-selected", b.dataset.tab === name));
  $$("section.tab").forEach(s => s.classList.toggle("active", s.id === "tab-" + name));
  if (name === "plots") renderProgression();
  if (history.replaceState) history.replaceState(null, "", "#" + name);
}
$$(".tabs button").forEach(b => b.addEventListener("click", () => showTab(b.dataset.tab)));

/* ---------------------------- run builder ----------------------------- */
function checksInto(elId, values, checkedVals, onchange, name) {
  const el = $(elId);
  el.innerHTML = values.map(v =>
    `<label><input type="checkbox" name="${esc(name)}" value="${esc(v)}"
      ${checkedVals.includes(v) ? "checked" : ""}> ${esc(v === "" ? "(untagged)" : v)}</label>`).join("");
  $$("input", el).forEach(i => i.addEventListener("change", onchange));
}
const checkedVals = elId => $$(`${elId} input:checked`).map(i => i.value);
const pyPrefix = () => $("#py-select").value;

function buildRunCmd() {
  const r = DATA.meta.roster;
  const parts = [pyPrefix(), "run_experiments.py"];
  const agents = checkedVals("#rb-agents");
  const envs = checkedVals("#rb-envs");
  const exams = checkedVals("#rb-exams");
  // always emit checked selections, even when they equal the ROSTER default:
  // the command is the reproducibility receipt, and an implicit default would
  // change meaning if ROSTER changes later
  if (agents.length) parts.push("--agents", ...agents);
  if (envs.length) parts.push("--envs", ...envs);
  if (exams.length) parts.push("--exams", ...exams);
  const seeds = $("#rb-seeds").value.trim();
  if (seeds) parts.push("--seeds", ...seeds.split(/\s+/));
  const ts = $("#rb-timesteps").value.trim();
  if (ts) parts.push("--timesteps", ts);
  const tag = $("#rb-tag").value.trim();
  if (tag) parts.push("--tag", tag);
  if ($("#rb-skip").checked) parts.push("--skip-training");
  if ($("#rb-force-train").checked) parts.push("--force-train");
  if ($("#rb-force-eval").checked) parts.push("--force-eval");
  if ($("#rb-dry").checked) parts.push("--dry-run");
  const mf = $("#rb-maxfail").value.trim();
  if (mf) parts.push("--max-failures", mf);
  setCmd("#rb-cmd", parts.join(" "));
  const notes = [];
  if (agents.length === 0 || envs.length === 0 || exams.length === 0)
    notes.push("⚠ empty selection falls back to the ROSTER default, check at least one box to be explicit.");
  if ($("#rb-skip").checked && $("#rb-force-train").checked)
    notes.push("⚠ --skip-training and --force-train together: force-train wins per cell.");
  const cells = (agents.length || r.agents.length) * (envs.length || r.envs.length)
              * (seeds ? seeds.split(/\s+/).length : r.seeds.length)
              * (exams.length || r.exams.length);
  notes.push(`plan: ${cells} cell(s)`);
  $("#rb-note").textContent = notes.join("  ");
}

function buildPPCmd() {
  const parts = [pyPrefix(), "plot_progression.py"];
  const tags = checkedVals("#pp-tags");
  if (tags.length) parts.push("--tags", ...tags.map(t => t === "" ? '""' : t));
  const out = $("#pp-out").value.trim();
  if (out) parts.push("--out", out);
  setCmd("#pp-cmd", parts.join(" "));
}

const ONELINERS = [
  ["plot_resilience.py", "per-mod curves + leaderboard + breaking-point scatter"],
  ["plot_efficiency.py", "cost-vs-performance views"],
  ["verify_paper_numbers.py", "re-derive the numbers cited in the paper from results/"],
  ["run_swingup_ppo.py", "swing-up PPO training run"],
  ["calibrate_swingup.py", "swing-up reward calibration"],
  ["calibrate_s_max.py", "empirical s_max rule from a trained policy"],
  ["ladder_aurc.py", "ladder-normalized AURC aggregation"],
  ["reproduce_error_band.py", "critic-seed error band (--policy-dir --n-seeds --n-episodes --max-iters)"],
  ["gui.py", "regenerate this page"],
];
function buildOneliners() {
  const el = $("#oneliners");
  el.innerHTML = "";
  const rows = ONELINERS.map(([s, d]) => [pyPrefix() + " " + s, d]);
  rows.push(["PY=" + pyPrefix() + " ./reproduce_ppo_pendulum_progression.sh",
             "full PPO/Pendulum progression sweep (bash; .ps1 twin on Windows)"]);
  for (const [cmd, desc] of rows) {
    const div = document.createElement("div");
    div.innerHTML = `<div class="cmd"><span class="cmdtext">${esc(cmd)}</span></div>
                     <div class="desc">${esc(desc)}</div>`;
    attachCopy(div.querySelector(".cmd"), () => cmd);
    el.appendChild(div);
  }
}

function initRunTab() {
  const r = DATA.meta.roster;
  checksInto("#rb-agents", DATA.meta.agents, r.agents, buildRunCmd, "agents");
  checksInto("#rb-envs", DATA.meta.envs, r.envs, buildRunCmd, "envs");
  checksInto("#rb-exams", DATA.meta.exams, r.exams, buildRunCmd, "exams");
  ["#rb-seeds", "#rb-timesteps", "#rb-tag", "#rb-maxfail"].forEach(id =>
    $(id).addEventListener("input", buildRunCmd));
  ["#rb-skip", "#rb-force-train", "#rb-force-eval", "#rb-dry"].forEach(id =>
    $(id).addEventListener("change", buildRunCmd));
  checksInto("#pp-tags", DATA.tags, [], buildPPCmd, "pptags");
  $("#pp-out").addEventListener("input", buildPPCmd);
  $("#py-select").addEventListener("change", () => {
    buildRunCmd(); buildPPCmd(); buildOneliners(); buildWatchCmd();
  });
  buildRunCmd(); buildPPCmd(); buildOneliners();
}

/* ------------------------------ results ------------------------------- */
const raw = r => (r.exam && r.exam.raw) || {};
const ROWS = DATA.records.map((r, i) => ({
  i,
  tag: r._tag, agent: r.config?.agent_name ?? "?", env: r.config?.env_id ?? "?",
  seed: r.config?.train_seed, steps: r.config?.total_timesteps,
  mod: r.mod ? r.mod.mod_type : "NONE",
  aurc: r.success, absAurc: raw(r).absolute_aurc,
  clean: r.clean_return, sHalf: raw(r).s_half, cliff: raw(r).cliff_slope,
  params: r.param_count, lat: r.inf_lat_ms,
  ts: r.timestamp || "", id: r.experiment_id || "",
  rec: r,
}));

const COLS = [
  ["tag",     "tag",    v => esc(tagLabel(v)), false],
  ["agent",   "agent",  esc, false],
  ["env",     "env",    esc, false],
  ["seed",    "seed",   v => v ?? "—", true],
  ["steps",   "budget", compact, true],
  ["mod",     "mod",    v => `<span style="color:var(--ink)"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${modVar(v)};margin-right:5px"></span>${esc(v)}</span>`, false],
  ["aurc",    "AURC",   v => fmt(v, 3), true],
  ["absAurc", "abs AURC", v => fmt(v, 1), true],
  ["clean",   "clean",  v => fmt(v, 1), true],
  ["sHalf",   "s½",     v => fmt(v, 3), true],
  ["cliff",   "cliff",  v => fmt(v, 1), true],
  ["params",  "params", fmtInt, true],
  ["lat",     "lat ms", v => fmt(v, 3), true],
];
const ALL = Symbol("all");  // "no filter" sentinel -- cannot collide with any string value
const state = { tag: ALL, agent: ALL, env: ALL, mod: ALL, q: "", sortKey: "i", sortDir: 1 };

function fillSelect(elId, values, allLabel) {
  // option 0 is always "all"; the change handler detects it by selectedIndex,
  // not by value (values can be any string, including "" for the untagged tag)
  const el = $(elId);
  el.innerHTML = `<option>${esc(allLabel)}</option>` +
    values.map(v => `<option value="${esc(v)}">${esc(tagLabel(v))}</option>`).join("");
}
function filteredRows() {
  const q = state.q.toLowerCase();
  return ROWS.filter(r =>
    (state.tag === ALL || r.tag === state.tag) &&
    (state.agent === ALL || r.agent === state.agent) &&
    (state.env === ALL || r.env === state.env) &&
    (state.mod === ALL || r.mod === state.mod) &&
    (q === "" || [r.id, r.tag, r.agent, r.env, r.mod, r.rec.notes || "", r.ts]
      .join(" ").toLowerCase().includes(q)));
}
function renderHead() {
  $("#results-head").innerHTML = COLS.map(([k, label, , num]) => {
    const arrow = state.sortKey === k ? (state.sortDir > 0 ? " ▲" : " ▼") : "";
    return `<th data-k="${k}" class="${num ? "num" : ""}">${esc(label)}${arrow}</th>`;
  }).join("");
  $$("#results-head th").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.k;
    if (state.sortKey === k) state.sortDir *= -1;
    else { state.sortKey = k; state.sortDir = 1; }
    renderTable();
  }));
}
function renderTable() {
  renderHead();
  const rows = filteredRows();
  const k = state.sortKey, d = state.sortDir;
  rows.sort((a, b) => {
    const va = a[k], vb = b[k];
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === "number" && typeof vb === "number") return (va - vb) * d;
    return String(va).localeCompare(String(vb)) * d;
  });
  const body = $("#results-body");
  body.innerHTML = rows.map(r =>
    `<tr class="rec" data-i="${r.i}">` + COLS.map(([key, , render, num]) =>
      `<td class="${num ? "num" : ""}">${render(r[key])}</td>`).join("") + "</tr>"
  ).join("");
  $("#f-count").textContent = `${rows.length} / ${ROWS.length} records`;
  $$("tr.rec", body).forEach(tr => tr.addEventListener("click", () => toggleDetail(tr)));
}
function toggleDetail(tr) {
  const next = tr.nextElementSibling;
  if (next && next.classList.contains("detail")) { next.remove(); return; }
  const row = ROWS[+tr.dataset.i];
  const dtr = document.createElement("tr");
  dtr.className = "detail";
  dtr.innerHTML = `<td colspan="${COLS.length}">${detailHTML(row)}</td>`;
  tr.after(dtr);
  const wbtn = dtr.querySelector(".to-watch");
  if (wbtn) wbtn.addEventListener("click", () => watchFromRecord(row));
  const pre = dtr.querySelector("pre");
  if (pre) pre.textContent = JSON.stringify(row.rec, null, 2);
  bindTips(dtr);
}
function detailHTML(row) {
  const r = row.rec, rw = raw(r);
  const chart = rw.points && rw.points.length ? recordCurveSVG(row) :
    `<div class="empty">no probe points in this record</div>`;
  const em = r.env_metrics || {};
  const emStr = Object.keys(em).length
    ? Object.entries(em).map(([k, v]) => `${esc(k)}=${isNum(v) ? fmt(v, 3) : esc(v)}`).join(" · ")
    : "—";
  return `<div class="detail-grid">
    <div class="chartbox">${chart}
      <div class="caption">dots = probed mean return (whiskers ±1 std over
        n_episodes); line = the fitted logistic mapped back to raw reward units;
        hairline marks s½.</div>
    </div>
    <div class="facts">
      <dl style="margin:0">
        <dt>experiment</dt><dd>${esc(row.id)} · ${esc(row.ts)}</dd>
        <dt>code version</dt><dd>${esc(r.code_version ?? "—")}</dd>
        <dt>formula</dt><dd class="formula">${esc(r.exam?.formula ?? "—")}</dd>
        <dt>adapt_score</dt><dd>${fmt(r.adapt_score, 3)}</dd>
        <dt>s_max</dt><dd>${fmt(rw.s_max, 2)}</dd>
        <dt>clean ± std</dt><dd>${fmt(r.clean_return, 1)} ± ${fmt(r.clean_return_std, 1)}</dd>
        <dt>train</dt><dd>${fmtInt(r.train_env_steps)} env steps · ${fmt(r.train_time_sec, 0)} s</dd>
        <dt>inference</dt><dd>${fmt(r.inf_lat_ms, 3)} ms · ${fmtInt(r.inf_macs)} MACs · ${fmt(r.inf_mem_mb, 2)} MB</dd>
        <dt>hardware</dt><dd>${esc(r.hardware?.backend ?? "?")} · ${esc(r.hardware?.gpu || r.hardware?.cpu || "")}</dd>
        <dt>env metrics</dt><dd>${emStr}</dd>
        <dt>notes</dt><dd>${esc(r.notes || "—")}</dd>
      </dl>
      <div style="margin-top:10px"><button class="btn to-watch">build watch.py command for this record →</button></div>
      <details class="json"><summary class="hint" style="cursor:pointer">full record JSON</summary><pre></pre></details>
    </div>
  </div>`;
}

/* record detail chart: probe points + fitted logistic, raw reward units */
function recordCurveSVG(row) {
  const rw = raw(row.rec);
  const pts = [...rw.points].sort((a, b) => a.strength - b.strength);
  const sMax = (isNum(rw.s_max) && rw.s_max > 0 ? rw.s_max
                : Math.max(...pts.map(p => p.strength))) || 1;
  const rClean = isNum(rw.clean_return) ? rw.clean_return : pts[0].mean_return;
  let worstPt = pts.reduce((best, p) =>
    Math.abs(p.strength - sMax) < Math.abs(best.strength - sMax) ? p : best, pts[0]);
  const rWorst = worstPt.mean_return;
  const s0 = rw.s_half, k = rw.cliff_slope;
  const hasFit = isNum(s0) && isNum(k);
  const fitY = s => rWorst + (rClean - rWorst) / (1 + Math.exp(k * (s - s0)));

  const W = 600, H = 320, M = { l: 62, r: 16, t: 14, b: 42 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  let lo = Infinity, hi = -Infinity;
  for (const p of pts) {
    const sd = isNum(p.std_return) ? p.std_return : 0;
    lo = Math.min(lo, p.mean_return - sd); hi = Math.max(hi, p.mean_return + sd);
  }
  if (hasFit) { lo = Math.min(lo, rWorst, rClean); hi = Math.max(hi, rWorst, rClean); }
  const pad = (hi - lo || 1) * 0.06; lo -= pad; hi += pad;
  const X = s => M.l + (s / sMax) * iw;
  const Y = v => M.t + (1 - (v - lo) / (hi - lo)) * ih;
  const col = modVar(row.mod);

  let g = "";
  const yLbl = t => (hi - lo) >= 50 ? fmtInt(Math.round(t)) : String(+t.toFixed(2));
  for (const t of niceTicks(lo, hi, 5)) {
    g += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)" stroke-width="1"/>
          <text x="${M.l - 8}" y="${Y(t) + 4}" text-anchor="end" font-size="11" fill="var(--ink-mut)">${yLbl(t)}</text>`;
  }
  for (const t of niceTicks(0, sMax, 5)) {
    if (t > sMax + 1e-9) continue;
    g += `<text x="${X(t)}" y="${H - M.b + 18}" text-anchor="middle" font-size="11" fill="var(--ink-mut)">${+t.toFixed(3)}</text>
          <line x1="${X(t)}" x2="${X(t)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--axis)" stroke-width="1"/>`;
  }
  g += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--axis)" stroke-width="1"/>`;

  let fit = "";
  if (hasFit) {
    const d = [];
    for (let j = 0; j <= 100; j++) {
      const s = (j / 100) * sMax;
      d.push(`${j ? "L" : "M"}${X(s).toFixed(1)},${Y(fitY(s)).toFixed(1)}`);
    }
    fit = `<path d="${d.join(" ")}" fill="none" stroke="${col}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round"/>`;
    if (s0 <= sMax) {
      fit += `<line x1="${X(s0)}" x2="${X(s0)}" y1="${M.t}" y2="${H - M.b}" stroke="var(--grid)" stroke-width="1"/>
              <text x="${X(s0) + 4}" y="${M.t + 12}" font-size="11" fill="var(--ink-mut)">s½=${fmt(s0, 3)}</text>`;
    }
  }

  let dots = "";
  for (const p of pts) {
    const sd = isNum(p.std_return) ? p.std_return : 0;
    const cx = X(p.strength), cy = Y(p.mean_return);
    if (sd > 0) {
      dots += `<line x1="${cx}" x2="${cx}" y1="${Y(p.mean_return - sd)}" y2="${Y(p.mean_return + sd)}"
                 stroke="${col}" stroke-width="1.5" opacity="0.55"/>`;
    }
    const tipTxt = `<div class='t'>s = ${+p.strength.toFixed(4)}</div>` +
      `<div class='m'>mean ${fmt(p.mean_return, 1)} · std ${fmt(p.std_return, 1)} · n=${p.n_episodes}</div>`;
    dots += `<circle cx="${cx}" cy="${cy}" r="4.5" fill="${col}" stroke="var(--surface)" stroke-width="2"/>
             <circle cx="${cx}" cy="${cy}" r="13" fill="transparent" data-tip="${esc(tipTxt)}"/>`;
  }

  return `<svg viewBox="0 0 ${W} ${H}" role="img"
    aria-label="Resilience curve: mean return vs perturbation strength for ${esc(row.mod)}">
    <text x="${M.l}" y="${M.t - 2}" font-size="12" fill="var(--ink-2)" font-weight="600">
      ${esc(row.mod)}, return vs strength</text>
    ${g}${fit}${dots}
    <text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle" font-size="11"
      fill="var(--ink-mut)">perturbation strength s</text>
    <text transform="rotate(-90 13 ${(M.t + H - M.b) / 2})" x="13" y="${(M.t + H - M.b) / 2}"
      text-anchor="middle" font-size="11" fill="var(--ink-mut)">mean return (raw)</text>
  </svg>`;
}

function initResultsTab() {
  const uniq = k => [...new Set(ROWS.map(r => r[k]))];
  fillSelect("#f-tag", DATA.tags, "all tags");
  fillSelect("#f-agent", uniq("agent").sort(), "all agents");
  fillSelect("#f-env", uniq("env").sort(), "all envs");
  fillSelect("#f-mod", uniq("mod").sort(), "all mods");
  [["#f-tag", "tag"], ["#f-agent", "agent"], ["#f-env", "env"], ["#f-mod", "mod"]]
    .forEach(([id, key]) => $(id).addEventListener("change", e => {
      state[key] = e.target.selectedIndex === 0 ? ALL : e.target.value; renderTable(); }));
  $("#f-q").addEventListener("input", e => { state.q = e.target.value; renderTable(); });
  renderTable();
}

/* ------------------------------- plots -------------------------------- */
const hiddenMods = new Set();
// metrics that belong to the policy, not the perturbation: every mod row of a
// policy repeats the same value, so they draw as one neutral line, not 4 mods
const MOD_FREE = new Set(["clean_return", "inf_lat_ms", "train_time_sec"]);
function metricOf(row, metric) {
  if (metric === "success") return row.rec.success;
  if (metric === "clean_return") return row.rec.clean_return;
  if (metric === "inf_lat_ms") return row.rec.inf_lat_ms;
  if (metric === "train_time_sec") return row.rec.train_time_sec;
  return raw(row.rec)[metric];
}
function renderProgression() {
  const env = $("#p-env").value, agent = $("#p-agent").value, metric = $("#p-metric").value;
  const modFree = MOD_FREE.has(metric);
  const rows = ROWS.filter(r => r.env === env && r.agent === agent &&
    (modFree || r.mod !== "NONE") && isNum(r.steps) && isNum(metricOf(r, metric)));
  const box = $("#p-chart"), legend = $("#p-legend"), cap = $("#p-caption");
  const metricLabel = $("#p-metric").selectedOptions[0].textContent.split(" (")[0];

  const series = new Map();  // series key -> Map(budget -> [{seed, tag, v}])
  if (modFree) {
    // collapse the per-mod duplicates: one value per (budget, seed, tag)
    const byB = new Map(), seen = new Set();
    for (const r of rows) {
      const key = `${r.steps}|${r.seed}|${r.tag}`;
      if (seen.has(key)) continue;
      seen.add(key);
      if (!byB.has(r.steps)) byB.set(r.steps, []);
      byB.get(r.steps).push({ seed: r.seed, tag: r.tag, v: metricOf(r, metric) });
    }
    if (byB.size) series.set(metricLabel, byB);
  } else {
    for (const r of rows) {
      if (!series.has(r.mod)) series.set(r.mod, new Map());
      const byB = series.get(r.mod);
      if (!byB.has(r.steps)) byB.set(r.steps, []);
      byB.get(r.steps).push({ seed: r.seed, tag: r.tag, v: metricOf(r, metric) });
    }
  }
  const keys = modFree ? [...series.keys()]
                       : DATA.meta.mods.filter(m => series.has(m));
  const colorOf = k => modFree ? "var(--accent)" : modVar(k);

  legend.innerHTML = modFree ? "" : keys.map(m =>
    `<span class="key ${hiddenMods.has(m) ? "off" : ""}" data-mod="${esc(m)}">
       <span class="sw" style="background:${modVar(m)}"></span>${esc(m)}</span>`).join("");
  $$(".key", legend).forEach(k => k.addEventListener("click", () => {
    const m = k.dataset.mod;
    hiddenMods.has(m) ? hiddenMods.delete(m) : hiddenMods.add(m);
    renderProgression();
  }));

  const visible = modFree ? keys : keys.filter(m => !hiddenMods.has(m));
  if (!visible.length) {
    box.innerHTML = `<div class="empty">no records for ${esc(agent)} × ${esc(env)}, run a sweep or change the filters</div>`;
    cap.textContent = ""; return;
  }

  const budgets = [...new Set(rows.map(r => r.steps))].sort((a, b) => a - b);
  let lo = Infinity, hi = -Infinity;
  for (const m of visible) for (const [, vals] of series.get(m))
    for (const { v } of vals) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const padv = (hi - lo || 1) * 0.08; lo -= padv; hi += padv;

  const W = 860, H = 380, M = { l: 70, r: 20, t: 16, b: 46 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const lb = Math.log10(budgets[0]), ub = Math.log10(budgets[budgets.length - 1]);
  const span = (ub - lb) || 1;
  const X = b => M.l + ((Math.log10(b) - lb) / span) * iw;
  const Y = v => M.t + (1 - (v - lo) / (hi - lo)) * ih;

  let g = "";
  for (const t of niceTicks(lo, hi, 6)) {
    g += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)" stroke-width="1"/>
          <text x="${M.l - 8}" y="${Y(t) + 4}" text-anchor="end" font-size="11" fill="var(--ink-mut)">${Math.abs(t) >= 1000 ? compact(t) : +t.toFixed(3)}</text>`;
  }
  for (const b of budgets) {
    g += `<line x1="${X(b)}" x2="${X(b)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--axis)" stroke-width="1"/>
          <text x="${X(b)}" y="${H - M.b + 18}" text-anchor="middle" font-size="11" fill="var(--ink-mut)">${compact(b)}</text>`;
  }
  g += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--axis)" stroke-width="1"/>`;

  let marks = "";
  for (const m of visible) {
    const col = colorOf(m);
    const byB = series.get(m);
    const pts = budgets.filter(b => byB.has(b)).map(b => {
      const vals = byB.get(b);
      return { b, vals, mean: vals.reduce((s, x) => s + x.v, 0) / vals.length };
    });
    const d = pts.map((p, j) => `${j ? "L" : "M"}${X(p.b).toFixed(1)},${Y(p.mean).toFixed(1)}`);
    marks += `<path d="${d.join(" ")}" fill="none" stroke="${col}" stroke-width="2"
                stroke-linecap="round" stroke-linejoin="round"/>`;
    for (const p of pts) {
      for (const s of p.vals) {
        marks += `<circle cx="${X(p.b)}" cy="${Y(s.v)}" r="2.5" fill="${col}" opacity="0.45"/>`;
      }
      const tipTxt = `<div class='t'>${esc(m)} @ ${compact(p.b)}</div>` +
        `<div class='m'>mean ${fmt(p.mean, 3)}</div>` +
        p.vals.map(s => `<div class='m'>seed ${s.seed} (${esc(tagLabel(s.tag))}): ${fmt(s.v, 3)}</div>`).join("");
      marks += `<circle cx="${X(p.b)}" cy="${Y(p.mean)}" r="4.5" fill="${col}"
                  stroke="var(--surface)" stroke-width="2"/>
                <circle cx="${X(p.b)}" cy="${Y(p.mean)}" r="13" fill="transparent" data-tip="${esc(tipTxt)}"/>`;
    }
  }

  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img"
    aria-label="Progression of ${esc(metricLabel)} vs training budget per mod">
    ${g}${marks}
    <text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle" font-size="11"
      fill="var(--ink-mut)">training budget (env steps, log scale)</text>
    <text transform="rotate(-90 14 ${(M.t + H - M.b) / 2})" x="14" y="${(M.t + H - M.b) / 2}"
      text-anchor="middle" font-size="11" fill="var(--ink-mut)">${esc(metricLabel)}</text>
  </svg>`;
  bindTips(box);
  cap.textContent = (metric === "success")
    ? "⚠ AURC is intra-policy (normalized to each row's own clean-worst band)"
    : modFree
    ? `${metricLabel}, the per-mod rows repeat the same value, collapsed here to one line (mean across seeds; faint dots = individual seeds).`
    : "line = mean across seeds at each budget; faint dots = individual seeds. Hover a marker for the per-seed breakdown.";
}
function initPlotsTab() {
  const envs = [...new Set(ROWS.map(r => r.env))].sort();
  const agents = [...new Set(ROWS.map(r => r.agent))].sort();
  const count = (env, ag) => ROWS.filter(r => r.env === env && r.agent === ag && r.mod !== "NONE").length;
  let best = null;
  for (const e of envs) for (const a of agents)
    if (!best || count(e, a) > count(best[0], best[1])) best = [e, a];
  $("#p-env").innerHTML = envs.map(e => `<option ${best && e === best[0] ? "selected" : ""}>${esc(e)}</option>`).join("");
  $("#p-agent").innerHTML = agents.map(a => `<option ${best && a === best[1] ? "selected" : ""}>${esc(a)}</option>`).join("");
  ["#p-env", "#p-agent", "#p-metric"].forEach(id =>
    $(id).addEventListener("change", renderProgression));
}

/* ------------------------------- watch --------------------------------- */
function buildWatchCmd() {
  const mode = $("#w-mode").value;
  $("#w-single-row").style.display = mode === "single" ? "" : "none";
  $("#w-prog-row").style.display = mode === "progression" ? "" : "none";
  const parts = [pyPrefix(), "watch.py", "--env", $("#w-env").value];
  const notes = [];
  if (mode === "single") {
    const w = $("#w-weights-free").value.trim() || $("#w-weights").value;
    if (w) parts.push("--weights", w);
    else notes.push("⚠ incomplete: pick a weights dir (or paste a path), watch.py requires --weights.");
  } else {
    const tags = checkedVals("#w-tags");
    if (tags.length) parts.push("--progression", ...tags);
    else notes.push("⚠ incomplete: check at least one tag. --progression needs them.");
  }
  const mod = $("#w-mod").value, s = parseFloat($("#w-strength").value);
  if (mod && s > 0) parts.push("--mod", mod, "--strength", String(s));
  if (mod && !(s > 0)) notes.push("note: strength 0 renders clean. Pick a strength > 0 to apply the mod.");
  const seed = $("#w-seed").value.trim();
  if (seed && seed !== "0") parts.push("--seed", seed);
  const steps = $("#w-steps").value.trim();
  if (steps && steps !== "300") parts.push("--steps", steps);
  setCmd("#w-cmd", parts.join(" "));
  $("#w-note").textContent = notes.join(" ");
}
function watchFromRecord(row) {
  showTab("watch");
  $("#w-mode").value = "single";
  $("#w-env").value = row.env;
  const guess = (row.tag ? `weights/${row.tag}/` : "weights/") +
    `${row.agent}/${row.env}/seed${row.seed}`;
  const opt = [...$("#w-weights").options].find(o => o.value === guess);
  if (opt) { $("#w-weights").value = guess; $("#w-weights-free").value = ""; }
  else $("#w-weights-free").value = guess;
  const noteBits = [];
  if (!DATA.weights.includes(guess))
    noteBits.push(`⚠ ${guess} was not on disk at snapshot time, check the path.`);
  if (row.mod !== "NONE") {
    $("#w-mod").value = row.mod;
    if (isNum(row.sHalf)) {
      $("#w-strength").value = String(Math.round(row.sHalf * 100) / 100);
      noteBits.push(`strength prefilled at this record's s½ = ${fmt(row.sHalf, 3)} (its breaking point).`);
    }
  } else { $("#w-mod").value = ""; $("#w-strength").value = "0"; }
  $("#w-seed").value = row.seed ?? 0;
  buildWatchCmd();
  $("#w-note").textContent = noteBits.join(" ");
  if (row.agent !== "PPO")
    $("#w-note").textContent += " ⚠ watch.py currently loads PPO checkpoints only.";
}
function initWatchTab() {
  $("#w-env").innerHTML = DATA.meta.envs.map(e => `<option>${esc(e)}</option>`).join("");
  $("#w-mod").innerHTML = `<option value="">- clean -</option>` +
    DATA.meta.mods.map(m => `<option>${esc(m)}</option>`).join("");
  const groups = new Map();
  for (const w of DATA.weights) {
    const seg = w.split("/");
    const grp = seg.length >= 5 ? seg[1] : "(untagged)";
    if (!groups.has(grp)) groups.set(grp, []);
    groups.get(grp).push(w);
  }
  $("#w-weights").innerHTML = `<option value="">- pick weights -</option>` +
    [...groups.entries()].map(([grp, ws]) =>
      `<optgroup label="${esc(grp)}">` +
      ws.map(w => `<option value="${esc(w)}">${esc(w.replace("weights/", ""))}</option>`).join("") +
      `</optgroup>`).join("");
  checksInto("#w-tags", DATA.tags.filter(t => t !== ""), [], buildWatchCmd, "wtags");
  ["#w-mode", "#w-env", "#w-mod"].forEach(id =>
    $(id).addEventListener("change", buildWatchCmd));
  $("#w-weights").addEventListener("change", () => {
    // picking from the list must win over stale free text (free text overrides)
    if ($("#w-weights").value) $("#w-weights-free").value = "";
    buildWatchCmd();
  });
  ["#w-strength", "#w-seed", "#w-steps", "#w-weights-free"].forEach(id =>
    $(id).addEventListener("input", buildWatchCmd));
  buildWatchCmd();

  const gal = $("#w-gallery");
  if (!DATA.renders.length) {
    gal.innerHTML = `<div class="empty">no files in results/render/ yet, build a watch.py
      command above, run it, then regenerate with <code>python gui.py</code></div>`;
  } else {
    gal.innerHTML = `<div class="gallery">` + DATA.renders.map(n =>
      `<figure><img loading="lazy" src="render/${esc(n)}" alt="${esc(n)}">
       <figcaption>${esc(n)}</figcaption></figure>`).join("") + `</div>`;
  }
}

/* -------------------------------- boot --------------------------------- */
initRunTab();
initResultsTab();
initPlotsTab();
initWatchTab();
const hash = location.hash.replace("#", "");
if (["run", "results", "plots", "watch"].includes(hash)) showTab(hash);
</script>
</body>
</html>
"""


def generate(out_path: Path) -> Path:
    meta = collect_meta()
    records, tags = load_records()
    renders = scan_renders()
    weights = scan_weights()

    def js(obj) -> str:
        # embedded as a JS literal (NaN is valid JS); guard </script> breakout
        return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")

    html = (TEMPLATE
            .replace("%%GENERATED%%", time.strftime("%Y-%m-%d %H:%M"))
            .replace("%%N_RECORDS%%", str(len(records)))
            .replace("%%N_TAGS%%", str(len(tags)))
            .replace("%%N_RENDERS%%", str(len(renders)))
            .replace("%%ROSTER_SEEDS%%", " ".join(str(s) for s in meta["roster"]["seeds"]))
            .replace("%%TIMESTEPS_DEFAULT%%", str(meta["roster"]["timesteps_default"]))
            .replace("%%RECORDS%%", js(records))
            .replace("%%META%%", js(meta))
            .replace("%%TAGS%%", js(tags))
            .replace("%%RENDERS%%", js(renders))
            .replace("%%WEIGHTS%%", js(weights)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  {len(records)} records across {len(tags)} tags · "
          f"{len(renders)} renders · {len(weights)} weight dirs")
    print(f"open it in a browser: {out_path.as_uri()}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help=f"output path (default {DEFAULT_OUT})")
    args = ap.parse_args()
    generate(Path(args.out) if args.out else DEFAULT_OUT)


if __name__ == "__main__":
    main()

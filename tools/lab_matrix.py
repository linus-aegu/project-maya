"""tools/lab_matrix.py - tools/lab.py's `matrix`: knobs judged next to each other, not one at a time (docs/LAB.md).

A spec (JSON) names the knobs, each with its levels (a level: engine settings and arguments) and the incumbent's level
(what production runs today):

    {"name": "r1", "design": "screen", "reps": 2, "quality": "first",
     "run_args": ["--config", "maya-ds4-hip.json", "--expert-profile", "frozen.bin", "--text", "text.txt"],
     "factors": {
       "copydma": {"levels": [{"env": ["DS4_FLOW_COPY_DMA=0"]}, {"env": ["DS4_FLOW_COPY_DMA=1"]}], "incumbent": 1},
       "skip":    {"levels": [{"env": ["DS4_SKIP_UNPICKED=0"]}, {"env": ["DS4_SKIP_UNPICKED=1"]}], "incumbent": 0}},
     "final": {"copydma": 1, "skip": 1}}

- screen: the incumbent, then each knob alone at each of its other levels;
- factorial: every combination of the knobs' levels;
- loo (leave-one-out): `final` (default: the incumbent), then `final` with each knob at its level 0 (off) in turn.

`plan` prints the runs as a queue (tools/lab_unit.sh lines): the reps interleave (the incumbent - or final - first,
the others in an order that reverses each rep, so slow drift spreads over every configuration), each against the
first configuration, with --keep-going (a design needs every cell) and --quality on the reps the spec asks for.
`report` reads them back: each configuration's metrics against the first one's, and for a factorial each knob's main
effect and each pair's interaction (does one knob's gain depend on the other?).
"""
from __future__ import annotations

import itertools
import json
import math
import shlex
import statistics as st
from pathlib import Path

KEY_METRICS = ("turn", "read_prefix", "read16", "read64", "read256", "read1024", "read4096", "decode",
               "mixed_decode", "mixed_read", "mixed_gap")


def configs(spec: dict) -> list[dict[str, int]]:
    """The design's configurations (knob -> level index), the reference first."""
    f = spec["factors"]
    names = sorted(f)
    inc = {k: int(f[k].get("incumbent", 0)) for k in names}
    design = spec.get("design", "screen")
    if design == "screen":
        out = [inc]
        for k in names:
            for lv in range(len(f[k]["levels"])):
                if lv != inc[k]:
                    out.append({**inc, k: lv})
        return out
    if design == "factorial":
        combos = [dict(zip(names, c)) for c in itertools.product(*(range(len(f[k]["levels"])) for k in names))]
        combos.sort(key=lambda c: c != inc)   # (the incumbent first, if it is one of them)
        return combos
    if design == "loo":
        final = {**inc, **{k: int(v) for k, v in (spec.get("final") or {}).items()}}
        return [final] + [{**final, k: 0} for k in names if final[k] != 0]
    raise ValueError(f"design: screen, factorial or loo, not {design}")


def label(spec: dict, c: dict[str, int]) -> str:
    return f"m-{spec['name']}." + ".".join(f"{k}{c[k]}" for k in sorted(c))


def settings(spec: dict, c: dict[str, int]) -> list[str]:
    """lab.py run's --env / --arg for a configuration."""
    out = []
    for k in sorted(c):
        lv = spec["factors"][k]["levels"][c[k]]
        out += ["--exe", lv["exe"]] if lv.get("exe") else []
        out += [x for e in lv.get("env", []) for x in ("--env", e)]
        out += [f"--arg={x}" for x in lv.get("arg", [])]
    return out


def order(n: int, reps: int) -> list[tuple[int, int]]:
    """(rep, configuration) in run order: the first configuration first in every rep, the others forward in even reps
    and backward in odd ones (counterbalanced against drift)."""
    out = []
    for r in range(reps):
        rest = list(range(1, n))
        out += [(r, 0)] + [(r, i) for i in (rest if r % 2 == 0 else rest[::-1])]
    return out


def plan(spec: dict) -> str:
    cs = configs(spec)
    base = label(spec, cs[0])
    quality = spec.get("quality", "first")
    lines = ["#!/usr/bin/env bash", f"# lab matrix '{spec['name']}' ({spec.get('design', 'screen')}): "
             f"{len(cs)} configurations x {spec.get('reps', 2)} reps, written by tools/lab_matrix.py",
             "cd \"$(dirname \"$0\")\"/.. 2>/dev/null || true", "set -u"]
    if spec.get("warmup", True):
        # unmeasured: the first run after the server served a while decoded ~35% faster in its first answers (base-w,
        # base-c 2026-11-08), so it would pull whichever configuration came first up
        args = ["run"] + list(spec.get("run_args", [])) + ["--label", f"m-{spec['name']}-warmup", "--keep-going",
                                                           "--no-mixed"] + settings(spec, cs[0])
        lines.append("tools/lab_unit.sh --wait 7200 --max-sec 1800 --follow -- " + " ".join(shlex.quote(x) for x in args))
    for rep, i in order(len(cs), int(spec.get("reps", 2))):
        # (the reference's first two reps both: the two give lab_quality its noise floor)
        q = quality == "all" or (quality == "first" and (rep == 0 or (i == 0 and rep == 1)))
        args = (["run"] + list(spec.get("run_args", [])) + ["--label", label(spec, cs[i]), "--keep-going"] +
                settings(spec, cs[i]) + (["--against", base] if i > 0 or rep > 0 else []) + (["--quality"] if q else []))
        lines.append("tools/lab_unit.sh --wait 7200 --max-sec 1800 --follow -- " + " ".join(shlex.quote(x) for x in args))
    lines.append("echo \"QUEUE DONE\"")
    return "\n".join(lines) + "\n"


def run_means(runs: list[dict], metric: str) -> list[float]:
    """One value a run (its mean of the metric): the unit replicates are counted in."""
    return [st.fmean(r["metrics"][metric]) for r in runs if r.get("metrics", {}).get(metric)]


def effects(cells: dict[tuple, list[float]], names: list[str]) -> dict:
    """A two-level factorial's main effects and two-way interactions from its cells (levels tuple -> run means):
    each as the mean change from level 0 to 1, with a standard error from the replicates' pooled spread."""
    means = {c: st.fmean(v) for c, v in cells.items() if v}
    if len(means) < 2 ** len(names):
        return {}
    sds = [st.stdev(v) for v in cells.values() if len(v) > 1]
    n_runs = sum(len(v) for v in cells.values())
    se = (math.sqrt(st.fmean([s * s for s in sds])) * 2 / math.sqrt(n_runs)) if sds else float("nan")
    out = {}
    for i, k in enumerate(names):
        hi = st.fmean([m for c, m in means.items() if c[i] == 1])
        lo = st.fmean([m for c, m in means.items() if c[i] == 0])
        out[k] = (hi - lo, se)
    for i, j in itertools.combinations(range(len(names)), 2):
        d = st.fmean([m * (1 if (c[i] == c[j]) else -1) for c, m in means.items()])
        out[f"{names[i]} x {names[j]}"] = (d, se)
    return out


def report(spec: dict, runs_dir: Path, load_runs, judge, metric_kind) -> str:
    cs = configs(spec)
    base_runs = load_runs(runs_dir, label(spec, cs[0]))
    if not base_runs:
        return f"no runs of {label(spec, cs[0])} yet"
    sig = base_runs[0]["signature"]
    rows = []
    by_cfg = {}
    for c in cs:
        rs = [r for r in load_runs(runs_dir, label(spec, c)) if r.get("signature") == sig and not r.get("error")]
        by_cfg[tuple(c[k] for k in sorted(c))] = rs
    metrics = [m for m in KEY_METRICS if any(values(rs, m) for rs in by_cfg.values())]
    head = f"{'configuration':34}" + "".join(f"{m:>16}" for m in metrics) + f"{'quality':>22}"
    rows.append(head)
    base_key = tuple(cs[0][k] for k in sorted(cs[0]))
    for c in cs:
        key = tuple(c[k] for k in sorted(c))
        rs = by_cfg[key]
        cells = []
        for m in metrics:
            v, ref = run_means(rs, m), run_means(by_cfg[base_key], m)
            if not v:
                cells.append(f"{'-':>16}")
                continue
            higher, rel = metric_kind(m)
            txt = f"{st.fmean(v):.0f}" if not higher else f"{st.fmean(v):.1f}"
            if key != base_key and ref:
                verdict, delta = judge(v, ref, higher, rel)
                txt += f" {delta:+.0f}%" + {"better": "+", "worse": "!", "same": " "}[verdict]
            cells.append(f"{txt:>16}")
        q = [f"{md[:1]}:{r['quality'][md].get('verdict', '-')}" for r in rs[:1] for md in sorted(r.get("quality") or {})]
        rows.append(f"{label(spec, c)[-34:]:34}" + "".join(cells) + f"{' '.join(q) or '-':>22}   ({len(rs)} runs)")
    rows.append("  +: better, !: worse (more than twice the combined standard error of the runs); quality: p(rompt) / "
                "w(indow) against the reference's noise floor")
    names = sorted(spec["factors"])
    if spec.get("design") == "factorial" and all(len(spec["factors"][k]["levels"]) == 2 for k in names):
        rows.append("")
        rows.append("effects (level 1 minus level 0, mean over the other knobs; interactions: half the difference of a "
                    "knob's effect between the other's levels):")
        for m in metrics:
            higher, _ = metric_kind(m)
            cells = {k: run_means(v, m) for k, v in by_cfg.items()}
            eff = effects(cells, names)
            if not eff:
                continue
            grand = st.fmean([x for v in cells.values() for x in v])
            parts = []
            for k, (d, se) in eff.items():
                good = (d > 0) == higher
                sig_mark = "" if math.isnan(se) or abs(d) <= 2 * se else ("+" if good else "!")
                parts.append(f"{k} {d / grand * 100:+.1f}%{sig_mark}")
            rows.append(f"  {m:12} " + "  ".join(parts))
    if spec.get("design") == "loo":
        rows.append("")
        rows.append("leave-one-out: a knob whose removal is worse earns its place; 'same' - it adds nothing; better - "
                    "it hurts next to the others")
    return "\n".join(rows)


def load_spec(path: str) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    for k, f in spec["factors"].items():
        if len(f.get("levels", [])) < 2:
            raise ValueError(f"knob {k}: at least two levels")
    return spec

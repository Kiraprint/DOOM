#!/usr/bin/env python3
"""Aggregate the optimizer study from one or more train_dir roots.

Arms are discovered from directory names (`<arm>_s<seed>`), and each cell's
optimizer / lr / wd / muon_lr are read from its own config.json rather than from
a hard-coded table -- the grid's whole point is that a config value must be
traceable to what trained, so the report must not restate the intended flags.

Metric: `last20`, mean of the last 20 `Avg episode reward` lines, computed by
DOOM-Mamba2-agent's `agg_runs.parse_log` so numbers are comparable to the
published campaign tables.

    python3 agg_optim.py RUN_ROOT [RUN_ROOT ...] [--ref ARM] [--out FILE]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

AGENT_AGG = "/home/kir/Code/DOOM-Mamba2-agent/scripts/agg_runs.py"
NAME_RE = re.compile(r"^(?P<arm>.+)_s(?P<seed>\d+)$")
MIN_FRAMES_FRACTION = 0.90     # a cell that died early is not the arm it claims


def load_agg(path: str):
    spec = importlib.util.spec_from_file_location("agg_runs", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cell(root: Path, arm: str, seed: int, min_frames: int, agg):
    run = root / f"{arm}_s{seed}"
    log, cfgp = run / "sf_log.txt", run / "config.json"
    if not log.exists():
        return None, f"{run.name}: no sf_log.txt"
    try:
        cfg = json.loads(cfgp.read_text())
    except Exception:
        cfg = {}
    series, collected, total, _fps, wall_h, sessions = agg.parse_log(log)
    frames = collected or total
    if not series:
        return None, f"{run.name}: no reward series"
    if sessions and sessions > 1:
        return None, f"{run.name}: {sessions} sessions in one log"
    if not frames or frames < min_frames:
        return None, f"{run.name}: {frames} frames (< {min_frames})"
    budget = cfg.get("train_for_env_steps") or 0
    return {
        "run": run.name, "arm": arm, "seed": seed,
        "last20": statistics.fmean(v for _, v in series[-20:]),
        "frames": frames, "wall_h": wall_h, "budget": budget,
        "optimizer": cfg.get("optimizer"), "lr": cfg.get("learning_rate"),
        "muon_lr": cfg.get("muon_lr"), "wd": cfg.get("weight_decay"),
        "eps": cfg.get("adam_eps"),
    }, None


def fmt(v):
    if v is None:
        return "—"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--ref", default=None, help="arm to take paired deltas against")
    ap.add_argument("--out", default=None)
    ap.add_argument("--parser", default=AGENT_AGG)
    args = ap.parse_args()
    agg = load_agg(args.parser)

    found, excluded = [], []
    for r in args.roots:
        root = Path(r)
        for run in sorted(root.glob("*_s*")):
            m = NAME_RE.match(run.name)
            if not m or not run.is_dir():
                continue
            budget = 0
            try:
                budget = int(json.loads((run / "config.json").read_text()).get("train_for_env_steps") or 0)
            except Exception:
                pass
            min_frames = int(budget * MIN_FRAMES_FRACTION) if budget else 0
            row, why = cell(root, m.group("arm"), int(m.group("seed")), min_frames, agg)
            (found if row else excluded).append(row or why)

    if not found:
        print("no complete cells yet\n" + "\n".join(f"- {e}" for e in excluded))
        return 0

    arms = defaultdict(list)
    for row in found:
        key = (row["optimizer"], row["muon_lr"] if row["optimizer"] == "muon" else None,
               row["wd"], row["lr"])
        arms[key].append(row)

    ref_rows = [r for r in found if args.ref and r["arm"] == args.ref]
    ref_by_seed = {r["seed"]: r["last20"] for r in ref_rows}

    lines = [
        "# Optimizer study — measured cells",
        "",
        "Metric `last20` = mean of the last 20 `Avg episode reward` lines, parsed with the",
        "campaign's own `agg_runs.parse_log`. Optimizer/lr/wd come from each cell's",
        "`config.json`, not from the submitted command line.",
        "",
        "| arm | optimizer | muon lr | wd | lr | eps | seeds | last20 per seed | n | mean ± s.d. |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (optim, muon_lr, wd, lr), rows in sorted(arms.items(), key=lambda kv: statistics.fmean(r["last20"] for r in kv[1]), reverse=True):
        rows.sort(key=lambda r: r["seed"])
        vals = [r["last20"] for r in rows]
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        arm_names = "/".join(sorted({r["arm"] for r in rows}))
        stat = f"{mean:.2f} ± {sd:.2f}" if len(vals) > 1 else f"{mean:.2f} (n=1)"
        lines.append(
            f"| `{arm_names}` | {fmt(optim)} | {fmt(muon_lr)} | {fmt(wd)} | {fmt(lr)} | "
            f"{fmt(rows[0]['eps'])} | {','.join(str(r['seed']) for r in rows)} | "
            f"{' / '.join(f'{v:.2f}' for v in vals)} | {len(vals)} | {stat} |"
        )

    if ref_by_seed:
        lines += [
            "",
            f"Paired deltas vs `{args.ref}` (per seed, so seed-common variance cancels):",
            "",
            "| comparable arm | paired Δ |",
            "|---|---|",
        ]
        for (optim, muon_lr, wd, lr), rows in sorted(arms.items(), key=lambda kv: statistics.fmean(r["last20"] for r in kv[1]), reverse=True):
            deltas = [r["last20"] - ref_by_seed[r["seed"]] for r in rows if r["seed"] in ref_by_seed]
            if len(deltas) < 2 or not args.ref or rows[0]["arm"] == args.ref:
                continue
            d = statistics.fmean(deltas)
            s = statistics.stdev(deltas) if len(deltas) > 1 else float("nan")
            lines.append(f"| `{rows[0]['arm']}` | {d:+.2f} ± {s:.2f} (n={len(deltas)}) |")

    if excluded:
        lines += ["", "Not counted:", ""] + [f"- {e}" for e in excluded]

    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python
"""
Ablation studies for DOOM RNN architecture comparison.

Measures FPS, VRAM, and brief reward signal for controlled config variations.
Each experiment runs 5M env steps (enough to see FPS/VRAM trends).

Usage:
    python -m _vizdoom.ablation --quick          # All ablations, 1M steps each
    python -m _vizdoom.ablation --dry-run         # Show what would run
    python -m _vizdoom.ablation --list            # List all ablation experiments
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ablation")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# All ablation experiments
ABLATIONS = {
    # === Gradient Checkpointing ===
    "gc_mamba2_on": {
        "desc": "Mamba-2 gradient checkpointing ON (default)",
        "rnn_type": "mamba2",
        "params": {
            "--gradient_checkpointing": "True",
        },
        "steps": 5000000,
    },
    "gc_mamba2_off": {
        "desc": "Mamba-2 gradient checkpointing OFF",
        "rnn_type": "mamba2",
        "params": {
            "--gradient_checkpointing": "False",
        },
        "steps": 5000000,
    },
    # === Mamba-2 expand ===
    "expand_mamba2_e1": {
        "desc": "Mamba-2 expand=1 (current best)",
        "rnn_type": "mamba2",
        "params": {},
        "steps": 5000000,
    },
    "expand_mamba2_e2": {
        "desc": "Mamba-2 expand=2 (more params)",
        "rnn_type": "mamba2",
        "params": {
            "--mamba_expand": "2",
        },
        "steps": 5000000,
    },
    # === Mamba-2 d_state ===
    "ds_mamba2_ds64": {
        "desc": "Mamba-2 d_state=64 (current default)",
        "rnn_type": "mamba2",
        "params": {
            "--mamba_d_state": "64",
        },
        "steps": 5000000,
    },
    "ds_mamba2_ds128": {
        "desc": "Mamba-2 d_state=128 (larger state)",
        "rnn_type": "mamba2",
        "params": {
            "--mamba_d_state": "128",
        },
        "steps": 5000000,
    },
    # === GRU rnn_size ===
    "gru_rs256": {
        "desc": "GRU rnn_size=256 (smaller)",
        "rnn_type": "gru",
        "params": {},
        "steps": 5000000,
    },
    "gru_rs512": {
        "desc": "GRU rnn_size=512 (default)",
        "rnn_type": "gru",
        "params": {},
        "steps": 5000000,
    },
    "gru_rs1024": {
        "desc": "GRU rnn_size=1024 (bigger)",
        "rnn_type": "gru",
        "params": {},
        "steps": 5000000,
    },
}


def run_experiment(exp_name: str, cfg: dict, dry_run: bool = False) -> Optional[Dict]:
    """Run one ablation experiment."""
    desc = cfg["desc"]
    rnn_type = cfg["rnn_type"]
    params = cfg.get("params", {})
    steps = cfg.get("steps", 5000000)

    ts = time.strftime('%Y%m%d_%H%M%S')
    full_name = f"ablation_{exp_name}_{ts}"
    exp_dir = PROJECT_ROOT / "train_dir" / full_name

    argv = [
        "uv", "run", "python", str(PROJECT_ROOT / "models" / "train.py"),
        "--rnn_type", rnn_type,
    ]
    for k, v in params.items():
        argv.extend([k, v])
    argv.extend(["--train_for_env_steps", str(steps)])
    argv.extend(["--experiment", full_name])

    logger.info("=" * 60)
    logger.info("Ablation: %s", desc)
    logger.info("  experiment: %s", full_name)
    logger.info("  rnn_type: %s", rnn_type)
    logger.info("  steps: %d", steps)
    logger.info("  params: %s", json.dumps(params))
    logger.info("  cmd: %s", " ".join(argv))

    if dry_run:
        return {"experiment": exp_name, "cmd": " ".join(argv), "dry_run": True}

    start = time.time()
    result = subprocess.run(argv, cwd=PROJECT_ROOT, capture_output=True, text=True)
    elapsed = time.time() - start

    # Parse results
    best_reward = 0.0
    final_reward = 0.0
    fps = 0.0
    text = ""
    log_file = exp_dir / "sf_log.txt"
    if log_file.exists():
        with open(log_file) as f:
            text = f.read()
        rewards = [float(r) for r in re.findall(r"Avg episode reward: \[\(0, '([\d.]+)'\)\]", text)]
        fps_list = [float(f) for f in re.findall(r'300 sec: ([\d.]+)\)', text)]
        if rewards:
            best_reward = max(rewards)
            final_reward = rewards[-1]
        if fps_list:
            fps = fps_list[-1]

    peak_vram = 0.0
    gpu_match = re.search(r"torch.cuda.max_memory_allocated.*?([\d.]+) GB", text if log_file.exists() else "")
    # Try nvidia-smi peak from log
    smi_match = re.findall(r"GPU Memory: ([\d.]+)MB allocated", text if log_file.exists() else "")
    if smi_match:
        peak_vram = max(float(m) for m in smi_match)

    status = "completed" if result.returncode == 0 else f"failed (rc={result.returncode})"

    result_data = {
        "experiment": exp_name,
        "desc": desc,
        "rnn_type": rnn_type,
        "status": status,
        "best_reward": round(best_reward, 2),
        "final_reward": round(final_reward, 2),
        "fps": round(fps, 0),
        "peak_vram_mb": round(peak_vram, 0),
        "elapsed_seconds": round(elapsed, 0),
    }

    logger.info("Result: %s", json.dumps(result_data))
    return result_data


def main():
    parser = argparse.ArgumentParser(description="Ablation studies")
    parser.add_argument("--quick", action="store_true", help="1M steps per ablation")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--experiment", type=str, default=None, help="Run single experiment")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'Experiment':25s} {'Description':50s} {'Steps':>10s}")
        print("-" * 85)
        for name, cfg in ABLATIONS.items():
            steps = cfg.get("steps", 5000000)
            if args.quick:
                steps = 1000000
            print(f"{name:25s} {cfg['desc']:50s} {steps:>10d}")
        return

    if args.dry_run:
        for name, cfg in ABLATIONS.items():
            steps = cfg.get("steps", 1000000 if args.quick else 5000000)
            print(f"[DRY RUN] {name}: {cfg['desc']} ({steps} steps)")
        return

    if args.experiment:
        if args.experiment not in ABLATIONS:
            logger.error("Unknown experiment: %s. Use --list to see available.", args.experiment)
            sys.exit(1)
        exp_cfg = dict(ABLATIONS[args.experiment])
        if args.quick:
            exp_cfg["steps"] = 1000000
        results = [run_experiment(args.experiment, exp_cfg, dry_run=False)]
    else:
        results = []
        for name, cfg in ABLATIONS.items():
            exp_cfg = dict(cfg)
            if args.quick:
                exp_cfg["steps"] = 1000000
            result = run_experiment(name, exp_cfg, dry_run=False)
            results.append(result)

    # Save summary
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    report_path = PROJECT_ROOT / "train_dir" / f"ablation_results_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", report_path)

    # Print summary table
    print(f"\n{'Experiment':25s} {'Best':>6s} {'Final':>6s} {'FPS':>7s}")
    print("-" * 45)
    for r in results:
        if r:
            print(f"{r['experiment']:25s} {r['best_reward']:6.2f} {r['final_reward']:6.2f} {r['fps']:7.0f}")


if __name__ == "__main__":
    main()

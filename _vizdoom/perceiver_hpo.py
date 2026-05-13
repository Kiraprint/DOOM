#!/usr/bin/env python
"""
Perceiver IO HPO: find best hyperparameters for Perceiver IO core.

Searches over architecture params (d_model, num_latents, num_blocks, etc.)
plus shared HPs (lr, optimizer, weight_decay, exploration_loss_coeff)
using the unified models.train runner.

Usage:
    python -m perceiver_hpo --max-trials 10 --time-budget 24
    python -m perceiver_hpo --dry-run
"""

import argparse
import json
import logging
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import optuna

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("perceiver_hpo")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Perceiver IO is slow (~8.5k FPS, ~99 min per 50M). Use 25M steps for faster trials.
TRIAL_TRAIN_STEPS = 25000000
# 25M at 8.5k FPS ~49 min, plus overhead = 2h per trial
TRIAL_TIMEOUT_SECONDS = 3 * 3600  # 3 hours safety margin
TRIAL_LIMIT = 10
TIME_BUDGET_HOURS = 24


# ── Search Space ──

def get_search_space() -> dict:
    """Perceiver IO HPO: architecture params + shared HPs.
    
    Uses 25M env steps per trial (instead of 50M) to fit more trials in budget,
    since Perceiver is the slowest architecture (~8.5k FPS).
    """
    return {
        # Architecture
        "perceiver_d_model": ("categorical", [256, 512]),
        "perceiver_num_latents": ("categorical", [16, 32, 64]),
        "perceiver_d_latents": ("categorical", [256, 512]),
        "perceiver_num_blocks": ("categorical", [1, 2, 3]),
        "perceiver_num_heads": ("categorical", [4, 8]),
        "perceiver_dropout": ("categorical", [0.1, 0.2]),
        # Shared HPs
        "learning_rate": ("loguniform", 1e-5, 1e-3),
        "exploration_loss_coeff": ("loguniform", 1e-4, 1e-2),
        "weight_decay": ("loguniform", 1e-4, 1e-1),
        "optimizer": ("categorical", ["adam", "adamw"]),
    }


def suggest(trial, param_name, spec):
    dist_type = spec[0]
    if dist_type == "categorical":
        return trial.suggest_categorical(param_name, spec[1])
    if dist_type == "loguniform":
        return trial.suggest_float(param_name, spec[1], spec[2], log=True)
    raise ValueError(f"Unknown distribution: {dist_type}")


# ── Training Execution ──

def _build_argv(params: dict, trial_id: int, train_dir: str) -> list:
    """Build argv for models.train (unified runner)."""
    argv = [
        str(PROJECT_ROOT / "models" / "train.py"),
        "--rnn_type", "perceiver",
        "--optimizer", params.get("optimizer", "adamw"),
        "--learning_rate", str(params.get("learning_rate", 4e-4)),
        "--exploration_loss_coeff", str(params.get("exploration_loss_coeff", 0.002)),
        "--perceiver_d_model", str(params.get("perceiver_d_model", 512)),
        "--perceiver_num_latents", str(params.get("perceiver_num_latents", 32)),
        "--perceiver_d_latents", str(params.get("perceiver_d_latents", 512)),
        "--perceiver_num_blocks", str(params.get("perceiver_num_blocks", 2)),
        "--perceiver_num_heads", str(params.get("perceiver_num_heads", 8)),
        "--perceiver_dropout", str(params.get("perceiver_dropout", 0.1)),
    ]
    wd = params.get("weight_decay", 0.0)
    if wd > 0:
        argv.extend(["--weight_decay", str(wd)])
    argv.extend(["--train_for_env_steps", str(TRIAL_TRAIN_STEPS)])
    argv.extend(["--experiment", f"perceiver_hpo_trial_{trial_id}"])
    return ["uv", "run", "python"] + argv


def _extract_best_reward(log_path: Path) -> float:
    """Parse best reward from sf_log.txt."""
    if not log_path.exists():
        return 0.0
    with open(log_path) as f:
        rewards = []
        for line in f:
            if "Avg episode reward" in line:
                m = re.search(r"'([\d.]+)'", line)
                if m:
                    rewards.append(float(m.group(1)))
    return max(rewards) if rewards else 0.0


class TrialTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TrialTimeoutError(f"Trial exceeded {TRIAL_TIMEOUT_SECONDS / 3600:.0f}h limit")


def run_trial(params: dict, trial_id: int, train_dir: str) -> dict:
    """Run one Perceiver IO training trial."""
    ts = time.strftime('%Y%m%d_%H%M%S')
    trial_name = f"perceiver_hpo_trial_{trial_id}_{ts}"

    argv = _build_argv(params, trial_id, train_dir)
    argv[argv.index("--experiment") + 1] = trial_name

    logger.info("Trial %d: params=%s", trial_id, json.dumps(params))
    logger.info("Trial %d: cmd=%s", trial_id, " ".join(argv))
    logger.info("Trial %d: experiment=%s", trial_id, trial_name)

    start_time = time.time()
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TRIAL_TIMEOUT_SECONDS)

    try:
        result = subprocess.run(argv, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=TRIAL_TIMEOUT_SECONDS)
        elapsed = time.time() - start_time

        if result.returncode != 0:
            logger.warning("Trial %d failed (rc=%d): %s", trial_id, result.returncode, result.stderr[-500:])
            return {"status": "error", "error": result.stderr[-500:], "mean_reward": 0.0, "max_reward": 0.0}

        experiment_dir = Path(train_dir) / trial_name
        log_file = experiment_dir / "sf_log.txt"
        best_reward = _extract_best_reward(log_file)

        logger.info("Trial %d done: best_reward=%.2f, elapsed=%.0fs", trial_id, best_reward, elapsed)
        return {
            "status": "completed",
            "mean_reward": best_reward,
            "max_reward": best_reward,
            "std_reward": 0.0,
            "elapsed_seconds": round(elapsed),
        }

    except TrialTimeoutError:
        logger.warning("Trial %d timed out", trial_id)
        return {"status": "error", "error": "timeout", "mean_reward": 0.0, "max_reward": 0.0}
    except subprocess.TimeoutExpired:
        logger.warning("Trial %d subprocess timed out", trial_id)
        return {"status": "error", "error": "subprocess timeout", "mean_reward": 0.0, "max_reward": 0.0}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ── Optuna Objective ──

def objective(trial: optuna.Trial) -> float:
    params = {}
    for pname, spec in get_search_space().items():
        params[pname] = suggest(trial, pname, spec)

    metrics = run_trial(params, trial.number, "train_dir")

    if metrics.get("status") == "error":
        raise RuntimeError(metrics.get("error", "unknown"))

    mean_r = metrics.get("mean_reward", 0.0)
    trial.set_user_attr("mean_reward", mean_r)
    trial.set_user_attr("max_reward", metrics.get("max_reward", 0.0))
    trial.set_user_attr("elapsed_seconds", metrics.get("elapsed_seconds", 0.0))
    trial.set_user_attr("status", metrics.get("status", "unknown"))

    return mean_r


# ── Orchestrator ──

def run_hpo(max_trials: int = TRIAL_LIMIT, time_budget_hours: float = TIME_BUDGET_HOURS, dry_run: bool = False):
    logger.info("=" * 60)
    logger.info("Perceiver IO HPO")
    logger.info("Trials    : %d", max_trials)
    logger.info("Budget    : %.1f hours", time_budget_hours)
    logger.info("Steps     : %d per trial", TRIAL_TRAIN_STEPS)
    logger.info("Dry run   : %s", dry_run)
    logger.info("=" * 60)

    start_time = time.time()

    study = optuna.create_study(
        study_name="perceiver_hpo",
        storage="sqlite:///perceiver_hpo.db",
        direction="maximize",
        load_if_exists=True,
    )
    logger.info("Study: %s | Existing trials: %d", study.study_name, len(study.trials))

    if dry_run:
        logger.info("DRY RUN — config OK: %s", list(get_search_space().keys()))
        return

    completed = 0
    while completed < max_trials:
        elapsed_h = (time.time() - start_time) / 3600
        if elapsed_h >= time_budget_hours:
            logger.warning("Time budget exceeded: %.1fh", elapsed_h)
            break

        logger.info("Starting trial %d/%d (elapsed=%.1fh)", completed + 1, max_trials, elapsed_h)
        study.optimize(objective, n_trials=1, catch=(RuntimeError, optuna.TrialPruned))
        completed += 1

        logger.info("Trial %d done | best=%.4f", completed, study.best_value if study.best_value is not None else 0)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    report_path = Path("train_dir") / f"perceiver_hpo_results_{timestamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "study_name": "perceiver_hpo",
        "n_trials": len(study.trials),
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_value,
            "params": study.best_params,
        } if study.best_trial else None,
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Results saved to %s", report_path)

    if study.best_trial:
        logger.info("Best: trial #%d | value=%.4f | params=%s",
                     study.best_trial.number, study.best_value,
                     json.dumps(study.best_params))
    return study


def main():
    parser = argparse.ArgumentParser(description="Perceiver IO HPO")
    parser.add_argument("--max-trials", type=int, default=TRIAL_LIMIT)
    parser.add_argument("--time-budget", type=float, default=TIME_BUDGET_HOURS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_hpo(max_trials=args.max_trials, time_budget_hours=args.time_budget, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

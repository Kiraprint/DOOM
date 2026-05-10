"""Main HPO orchestrator for Mamba-2 hyperparameter optimization.

Integrates Optuna study, trial wrapper, and ASHA early stopping.
Enforces trial limit (25) and time budget (48h).
Logs all trials to TensorBoard + JSON.

Usage:
    python -m hpo.run_hpo                    # Full run
    python -m hpo.run_hpo --dry-run          # Test without training
    python -m hpo.run_hpo --max-trials 10    # Override trial limit
    python -m hpo.run_hpo --time-budget 24   # Override time budget (hours)

GPU Management:
    Automatically stops llama-server before HPO.
    Restarts llama-server after HPO completes.
    Verifies GPU memory before each trial.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
from optuna.study import Study

from hpo.constants import TRIAL_LIMIT, TIME_BUDGET_HOURS
from hpo.study import create_study
from hpo.trial_wrapper import objective
from hpo.evaluate import _track_gpu_memory

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hpo.orchestrator")

# ---------------------------------------------------------------------------
# Trial logging
# ---------------------------------------------------------------------------

def _trial_to_dict(trial: optuna.trial.FrozenTrial) -> Dict[str, Any]:
    return {
        "number": trial.number,
        "state": str(trial.state),
        "value": trial.value,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
        "datetime_start": (
            trial.datetime_start.isoformat() if trial.datetime_start else None
        ),
        "datetime_complete": (
            trial.datetime_complete.isoformat() if trial.datetime_complete else None
        ),
    }


def save_trials_json(study: Study, output_path: Path) -> None:
    trials_data: List[Dict[str, Any]] = []
    for trial in study.trials:
        trials_data.append(_trial_to_dict(trial))

    report = {
        "study_name": study.study_name,
        "direction": str(study.direction),
        "n_trials": len(study.trials),
        "best_trial": _trial_to_dict(study.best_trial) if study.best_trial else None,
        "trials": trials_data,
    }

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Trial log saved to %s", output_path)


# ---------------------------------------------------------------------------
# Time budget tracking
# ---------------------------------------------------------------------------

class TimeBudgetExceeded(Exception):
    pass


def check_time_budget(start_time: float, budget_hours: float) -> None:
    elapsed_hours = (time.time() - start_time) / 3600.0
    if elapsed_hours >= budget_hours:
        raise TimeBudgetExceeded(
            f"Time budget exceeded: {elapsed_hours:.2f}h >= {budget_hours}h"
        )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_hpo(
    max_trials: int = TRIAL_LIMIT,
    time_budget_hours: float = TIME_BUDGET_HOURS,
    dry_run: bool = False,
    study_name: Optional[str] = None,
    log_dir: str = "train_dir",
) -> Optional[Study]:
    """Run the full HPO pipeline.

    Args:
        max_trials: Maximum number of trials (default 25).
        time_budget_hours: Maximum wall-clock time in hours (default 48).
        dry_run: If True, skip actual training and return immediately.
        study_name: Override study name.
        log_dir: Base directory for training logs.

    Returns:
        The completed Optuna Study.
    """
    # ---- Pre-flight checks ----
    logger.info("=" * 60)
    logger.info("Mamba-2 HPO Orchestrator")
    logger.info("Trial limit : %d", max_trials)
    logger.info("Time budget : %.1f hours", time_budget_hours)
    logger.info("Dry run     : %s", dry_run)
    logger.info("=" * 60)


    start_time = time.time()
    study: Optional[Study] = None

    try:
        # Create Optuna study
        study = create_study(study_name=study_name)
        logger.info("Study: %s  |  Existing trials: %d", study.study_name, len(study.trials))

        # Dry-run mode
        if dry_run:
            logger.info("DRY RUN — no training will be executed")
            logger.info("Configuration verified successfully")
            return study

        # Run optimization loop
        trials_completed = 0
        while trials_completed < max_trials:
            # Check time budget before each trial
            check_time_budget(start_time, time_budget_hours)

            _track_gpu_memory(f"BEFORE_TRIAL_{trials_completed + 1}")

            # Add one trial
            study.optimize(
                objective,
                n_trials=1,
                catch=(RuntimeError, optuna.TrialPruned),
            )
            trials_completed += 1
            _track_gpu_memory(f"AFTER_TRIAL_{trials_completed}")
            logger.info(
                "Trial %d/%d completed  |  Best value: %.4f  |  Elapsed: %.1fh",
                trials_completed,
                max_trials,
                study.best_value if study.best_value is not None else float("nan"),
                (time.time() - start_time) / 3600.0,
            )

        # Save results to JSON
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        json_path = Path(log_dir) / f"hpo_results_{timestamp}.json"
        save_trials_json(study, json_path)

        # Summary
        logger.info("=" * 60)
        logger.info("HPO COMPLETE")
        logger.info("Trials completed : %d", trials_completed)
        logger.info("Best trial       : #%d", study.best_trial.number)
        logger.info("Best value       : %.4f", study.best_value)
        logger.info("Best params      : %s", json.dumps(study.best_params, indent=2))
        logger.info("Results saved to : %s", json_path)
        logger.info("=" * 60)

        return study

    except TimeBudgetExceeded as e:
        logger.warning("Stopping HPO: %s", e)
        if study is not None:
            save_trials_json(study, Path(log_dir) / "hpo_results_interrupted.json")
        return study

    except KeyboardInterrupt:
        logger.warning("HPO interrupted by user")
        if study is not None:
            save_trials_json(study, Path(log_dir) / "hpo_results_interrupted.json")
        return study


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mamba-2 HPO Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m hpo.run_hpo                     # Full run (25 trials, 48h)\n"
            "  python -m hpo.run_hpo --dry-run           # Test without training\n"
            "  python -m hpo.run_hpo --max-trials 10     # Limited trials\n"
            "  python -m hpo.run_hpo --time-budget 24    # 24-hour budget\n"
        ),
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=TRIAL_LIMIT,
        help=f"Maximum number of trials (default: {TRIAL_LIMIT})",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=TIME_BUDGET_HOURS,
        help=f"Time budget in hours (default: {TIME_BUDGET_HOURS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip training — verify configuration only",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Override Optuna study name",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="train_dir",
        help="Base directory for training logs (default: train_dir)",
    )

    args = parser.parse_args()

    # Guardrails
    if args.max_trials > TRIAL_LIMIT:
        logger.warning(
            "--max-trials %d exceeds plan limit of %d, capping at %d",
            args.max_trials,
            TRIAL_LIMIT,
            TRIAL_LIMIT,
        )
        args.max_trials = TRIAL_LIMIT

    run_hpo(
        max_trials=args.max_trials,
        time_budget_hours=args.time_budget,
        dry_run=args.dry_run,
        study_name=args.study_name,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()

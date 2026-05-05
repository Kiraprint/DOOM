"""HPO result aggregation and reporting module.

Reads trial log JSON files produced by run_hpo.py, aggregates metrics
across all trials, computes stability indicators (oscillation amplitude),
and generates a structured report with top-3 configurations and
verification commands.

Usage:
    from hpo.report import analyze_results

    report = analyze_results(log_dir="train_dir")
    print(json.dumps(report, indent=2))

    # Or as a CLI:
    python -m hpo.report --log-dir train_dir

Output format (JSON):
    {
        "summary": { ... },
        "config_aggregates": [ { "config_key": ..., "mean_reward": ..., "std_reward": ..., "oscillation_amplitude": ..., "n_trials": ... }, ... ],
        "top_3_configs": [ { "config": ..., "mean_reward": ..., "verification_command": ... }, ... ],
        "all_trials": [ ... ]
    }
"""

import argparse
import json
import logging
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hpo.report")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    """Parsed trial result from a JSON log."""
    number: int
    state: str
    value: float
    params: Dict[str, Any]
    user_attrs: Dict[str, Any]
    datetime_start: Optional[str]
    datetime_complete: Optional[str]


@dataclass
class ConfigAggregate:
    """Aggregated metrics for a unique hyperparameter configuration."""
    config_key: str
    params: Dict[str, Any]
    mean_reward: float
    std_reward: float
    max_reward: float
    min_reward: float
    oscillation_amplitude: float
    n_trials: int
    trial_numbers: List[int]


@dataclass
class Report:
    """Full analysis report."""
    summary: Dict[str, Any]
    config_aggregates: List[Dict[str, Any]]
    top_3_configs: List[Dict[str, Any]]
    all_trials: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _load_trial_logs(log_dir: str) -> List[Dict[str, Any]]:
    """Load all hpo_results_*.json files from log_dir."""
    log_path = Path(log_dir)
    if not log_path.exists():
        logger.warning("Log directory %s does not exist", log_path)
        return []

    json_files = sorted(log_path.glob("hpo_results_*.json"))
    if not json_files:
        logger.warning("No hpo_results_*.json files found in %s", log_path)
        return []

    all_data: List[Dict[str, Any]] = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_source_file"] = str(jf)
            all_data.append(data)
            logger.info("Loaded %d trials from %s", data.get("n_trials", 0), jf)
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Failed to load %s: %s", jf, e)

    return all_data


def _parse_trials(data: List[Dict[str, Any]]) -> List[TrialResult]:
    """Convert raw JSON to TrialResult objects."""
    trials: List[TrialResult] = []
    for study_data in data:
        for t in study_data.get("trials", []):
            if t.get("state") != "COMPLETE":
                continue
            trials.append(TrialResult(
                number=t["number"],
                state=t["state"],
                value=t.get("value", 0.0) or 0.0,
                params=t.get("params", {}),
                user_attrs=t.get("user_attrs", {}),
                datetime_start=t.get("datetime_start"),
                datetime_complete=t.get("datetime_complete"),
            ))
    return trials


# ---------------------------------------------------------------------------
# Config key generation
# ---------------------------------------------------------------------------

def _make_config_key(params: Dict[str, Any]) -> str:
    """Generate a deterministic string key for a parameter set."""
    parts = []
    for k in sorted(params.keys()):
        v = params[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.6g}")
        else:
            parts.append(f"{k}={v}")
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------------

def _compute_oscillation_amplitude(rewards: List[float]) -> float:
    """Compute oscillation amplitude as max - min of reward series.

    A high amplitude indicates unstable training across runs with the
    same configuration, suggesting the config is fragile.
    """
    if len(rewards) < 2:
        return 0.0
    return max(rewards) - min(rewards)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_by_config(trials: List[TrialResult]) -> List[ConfigAggregate]:
    """Group trials by unique config and compute aggregate metrics."""
    buckets: Dict[str, List[TrialResult]] = {}
    for t in trials:
        key = _make_config_key(t.params)
        buckets.setdefault(key, []).append(t)

    aggregates: List[ConfigAggregate] = []
    for key, trial_list in buckets.items():
        rewards = [t.user_attrs.get("mean_reward", t.value) for t in trial_list]
        std_rewards = [t.user_attrs.get("std_reward", 0.0) for t in trial_list]
        max_rewards = [t.user_attrs.get("max_reward", 0.0) for t in trial_list]

        amp = _compute_oscillation_amplitude(rewards)

        aggregates.append(ConfigAggregate(
            config_key=key,
            params=trial_list[0].params,
            mean_reward=statistics.mean(rewards),
            std_reward=statistics.mean(std_rewards),
            max_reward=max(max_rewards),
            min_reward=min(rewards),
            oscillation_amplitude=amp,
            n_trials=len(trial_list),
            trial_numbers=[t.number for t in trial_list],
        ))

    # Sort by mean_reward descending
    aggregates.sort(key=lambda a: a.mean_reward, reverse=True)
    return aggregates


# ---------------------------------------------------------------------------
# Verification command builder
# ---------------------------------------------------------------------------

def _build_verify_command(params: Dict[str, Any], trial_id: int) -> str:
    cmd = "python -m hpo.run_hpo --dry-run --log-dir train_dir"
    return cmd


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def analyze_results(log_dir: str = "train_dir") -> Report:
    """Main entry point: analyze all HPO trial logs and return a report."""
    data = _load_trial_logs(log_dir)
    if not data:
        logger.warning("No data to analyze")
        return Report(
            summary={"n_log_files": 0, "n_trials": 0, "error": "no log files found"},
            config_aggregates=[],
            top_3_configs=[],
            all_trials=[],
        )

    trials = _parse_trials(data)
    aggregates = _aggregate_by_config(trials)

    # Summary
    all_rewards = [t.user_attrs.get("mean_reward", t.value) for t in trials]
    summary = {
        "n_log_files": len(data),
        "n_trials": len(trials),
        "n_unique_configs": len(aggregates),
        "best_mean_reward": aggregates[0].mean_reward if aggregates else 0.0,
        "worst_mean_reward": aggregates[-1].mean_reward if aggregates else 0.0,
        "avg_oscillation_amplitude": statistics.mean(
            [a.oscillation_amplitude for a in aggregates]
        ) if aggregates else 0.0,
    }

    # Top 3 configs
    top_3: List[Dict[str, Any]] = []
    for agg in aggregates[:3]:
        top_3.append({
            "config_key": agg.config_key,
            "params": agg.params,
            "mean_reward": round(agg.mean_reward, 4),
            "std_reward": round(agg.std_reward, 4),
            "oscillation_amplitude": round(agg.oscillation_amplitude, 4),
            "n_trials": agg.n_trials,
            "trial_numbers": agg.trial_numbers,
            "verification_command": _build_verify_command(
                agg.params, agg.trial_numbers[0]
            ),
        })

    # Serialize aggregates
    agg_serialized = []
    for a in aggregates:
        d = asdict(a)
        for k in ("mean_reward", "std_reward", "max_reward", "min_reward", "oscillation_amplitude"):
            d[k] = round(d[k], 4)
        agg_serialized.append(d)

    return Report(
        summary=summary,
        config_aggregates=agg_serialized,
        top_3_configs=top_3,
        all_trials=[asdict(t) for t in trials],
    )


def report_to_json(report: Report, *, indent: int = 2) -> str:
    """Serialize a Report to a JSON string."""
    return json.dumps({
        "summary": report.summary,
        "config_aggregates": report.config_aggregates,
        "top_3_configs": report.top_3_configs,
        "all_trials": report.all_trials,
    }, indent=indent, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HPO Result Aggregation and Reporting",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="train_dir",
        help="Directory containing hpo_results_*.json files (default: train_dir)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top configs to highlight (default: 3)",
    )
    args = parser.parse_args()

    report = analyze_results(args.log_dir)
    output = report_to_json(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report written to {out_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()

"""Restart failed HPO trials with their exact original parameters.

Usage:
    python -m hpo.restart_failed
"""
import json
import sqlite3
import subprocess
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "hpo_study.db"
TRAIN_DIR = Path(__file__).resolve().parent.parent / "train_dir"


def decode_params(trial_id: int) -> dict:
    """Read params from DB and decode categorical indices."""
    conn = sqlite3.connect(str(DB))
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT param_name, param_value, distribution_json "
            "FROM trial_params WHERE trial_id = ?",
            (trial_id,),
        ).fetchall()

        params = {}
        for row in rows:
            name = row["param_name"]
            pval = row["param_value"]
            dist = json.loads(row["distribution_json"])

            if "CategoricalDistribution" in str(dist):
                choices = dist["attributes"]["choices"]
                params[name] = choices[int(pval)]
            else:
                params[name] = pval
        return params
    finally:
        conn.close()


def main():
    # Find failed trials
    conn = sqlite3.connect(str(DB))
    try:
        failed = conn.execute(
            "SELECT number, trial_id FROM trials WHERE state = 'FAIL' ORDER BY number"
        ).fetchall()
    finally:
        conn.close()

    # Only retry the specific trials that failed due to dev/shm issues
    retry_numbers = {49, 50, 51}

    # Filter failed trials to only those we want to retry
    failed = [(num, tid) for num, tid in failed if num in retry_numbers]

    if not failed:
        print("No failed trials to retry.")
        return

    print(f"Found {len(failed)} failed trial(s) to retry: {[t[0] for t in failed]}")

    # Clean dev/shm to prevent the same issue
    r = subprocess.run(["ipcrm", "-M", "0xce320210"], capture_output=True, text=True, timeout=30)
    if r.stdout:
        print("Cleaned dev/shm: %s %s", r.stdout, r.stderr)

    # Now run each trial
    from hpo.evaluate import run_trial, _cleanup_shm
    from hpo.trial_wrapper import _stability_weighted_reward

    # Pre-clean dev/shm
    _cleanup_shm()

    for trial_num, trial_id in failed:
        params = decode_params(trial_id)
        print(f"\n{'=' * 60}")
        print(f"Restarting trial #{trial_num} (DB trial_id={trial_id})")
        print(f"Params: {json.dumps(params, indent=2)}")
        print(f"{'=' * 60}")

        # Call run_trial directly — it handles everything
        result = run_trial(
            params=params,
            trial_id=trial_num,
            train_dir=str(TRAIN_DIR),
            timeout=7200,
        )

        # Compute stability-weighted reward from extracted metrics
        metrics = result
        reward = _stability_weighted_reward(metrics)

        print(f"\nResult: {result.get('status', 'unknown')}")
        print(f"  max_reward={metrics.get('max_reward', 0):.4f}")
        print(f"  mean_reward={metrics.get('mean_reward', 0):.4f}")
        print(f"  std_reward={metrics.get('std_reward', 0):.4f}")
        print(f"  stability_weighted_reward={reward:.4f}")
        print(f"  elapsed={result.get('elapsed_seconds', 0):.0f}s")

    print(f"\n{'=' * 60}")
    print("All failed trials restarted.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

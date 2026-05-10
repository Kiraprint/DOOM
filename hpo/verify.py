"""Best-config verification with multiple seeds.

Runs top-3 HPO configurations with 3 seeds each, computes mean +/- std
reward, and compares against the GRU baseline (15.32).

Usage:
    python -m hpo.verify
    python -m hpo.verify --log-dir train_dir --seeds 1,2,3
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hpo.report import analyze_results
from hpo.evaluate import run_trial

GRU_BASELINE_REWARD = 15.32
DEFAULT_SEEDS = (0, 1, 2)


def _get_top_configs(log_dir: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Load top-N configs from HPO report."""
    report = analyze_results(log_dir)
    configs = report.top_3_configs[:top_n]
    if not configs:
        raise RuntimeError(
            f"No HPO results found in {log_dir}. "
            "Run HPO first (python -m hpo.run_hpo)."
        )
    return configs


def _run_single_seed(params: Dict[str, Any], seed: int, train_dir: str) -> Dict[str, Any]:
    """Run one training trial with a fixed seed."""
    params_with_seed = {**params, "seed": seed}
    result = run_trial(
        params=params_with_seed,
        trial_id=seed,
        train_dir=train_dir,
    )
    result["seed"] = seed
    return result


def _run_verification(config: Dict[str, Any], seeds: List[int], train_dir: str) -> Dict[str, Any]:
    """Run all seeds for a single config, return aggregated metrics."""
    params = config["params"]
    config_key = config["config_key"]

    print(f"\n{'=' * 60}")
    print(f"Verifying: {config_key}")
    print(f"{'=' * 60}")

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        t0 = time.time()
        print(f"  Seed {seed}...")
        result = _run_single_seed(params, seed, train_dir)
        elapsed = time.time() - t0
        mean_r = result.get("mean_reward", 0.0)
        print(f"  Seed {seed}: mean_reward={mean_r:.2f}  ({elapsed:.0f}s)")
        results.append(result)

    rewards = [r.get("mean_reward", 0.0) for r in results]
    mean_reward = statistics.mean(rewards)
    std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0.0

    return {
        "config_key": config_key,
        "params": params,
        "seeds": seeds,
        "per_seed_rewards": rewards,
        "mean_reward": round(mean_reward, 4),
        "std_reward": round(std_reward, 4),
        "n_seeds": len(seeds),
        "beats_gru_baseline": mean_reward > GRU_BASELINE_REWARD,
        "delta_vs_gru": round(mean_reward - GRU_BASELINE_REWARD, 4),
    }


def _format_table(results: List[Dict[str, Any]]) -> str:
    """Format verification results as a human-readable table."""
    header = (
        f"{'Config Key':<30} {'Mean':>8} {'Std':>8} "
        f"{'vs GRU':>8} {'Beats GRU':>10}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        tag = "+" if r["beats_gru_baseline"] else "-"
        line = (
            f"{r['config_key'][:29]:<30} "
            f"{r['mean_reward']:>8.2f} "
            f"{r['std_reward']:>8.2f} "
            f"{r['delta_vs_gru']:+>7.2f}{tag} "
            f"{'YES' if r['beats_gru_baseline'] else 'NO':>10}"
        )
        lines.append(line)

    return "\n".join(lines)


def verify_best_configs(
    log_dir: str = "train_dir",
    train_dir: str = "train_dir",
    top_n: int = 3,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Main verification entry point."""
    if seeds is None:
        seeds = list(DEFAULT_SEEDS)

    configs = _get_top_configs(log_dir, top_n)
    print(f"Loaded {len(configs)} top configs from {log_dir}")
    print(f"Seeds: {seeds}")
    print(f"GRU baseline reward: {GRU_BASELINE_REWARD}")

    verification_results: List[Dict[str, Any]] = []
    t_start = time.time()

    for config in configs:
        result = _run_verification(config, seeds, train_dir)
        verification_results.append(result)

    total_elapsed = time.time() - t_start

    # Build summary
    summary = {
        "gru_baseline_reward": GRU_BASELINE_REWARD,
        "seeds_used": seeds,
        "n_configs_verified": len(verification_results),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "configs_beating_baseline": sum(
            1 for r in verification_results if r["beats_gru_baseline"]
        ),
    }

    output = {
        "summary": summary,
        "results": verification_results,
        "table": _format_table(verification_results),
    }

    print(f"\n\n{'#':^{62}}")
    print(_format_table(verification_results))
    print(f"\nTotal time: {total_elapsed:.0f}s")
    print(
        f"Configs beating GRU ({GRU_BASELINE_REWARD}): "
        f"{summary['configs_beating_baseline']}/{summary['n_configs_verified']}"
    )

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify best HPO configs with multiple seeds",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="train_dir",
        help="Directory with HPO results (default: train_dir)",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="train_dir",
        help="Directory for verification training logs (default: train_dir)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top configs to verify (default: 3)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated seed list (default: 0,1,2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    result = verify_best_configs(
        log_dir=args.log_dir,
        train_dir=args.train_dir,
        top_n=args.top_n,
        seeds=seeds,
    )

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print("Results saved to %s", str(out_path))


if __name__ == "__main__":
    main()

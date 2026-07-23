"""Optuna trial wrapper for Mamba-2 HPO.

Usage:
    import optuna
    from hpo.trial_wrapper import objective

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)
"""

import time
import signal
import optuna
from typing import Dict, Any

from hpo.search_space import get_search_space, suggest
from hpo.evaluate import run_trial, _get_gpu_memory, _track_gpu_memory


TRIAL_TIMEOUT_SECONDS = 2 * 3600  # 2 hours


class TrialTimeoutError(Exception):
    """Raised when a trial exceeds the time budget."""
    pass


def _timeout_handler(signum, frame):
    raise TrialTimeoutError(
        f"Trial exceeded {TRIAL_TIMEOUT_SECONDS / 3600:.0f}-hour time limit"
    )


def _suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Convert Optuna trial suggestions to a parameter dictionary."""
    search_space = get_search_space()
    params = {}
    for param_name, spec in search_space.items():
        params[param_name] = suggest(trial, param_name, spec)
    return params


def _stability_weighted_reward(metrics: Dict[str, Any]) -> float:
    """
    Compute stability-weighted reward.

    Penalises high-variance runs to encourage robust hyperparameters.
    reward = mean_reward / (1 + std_reward / max(1, mean_reward))
    """
    mean_reward = metrics.get('mean_reward', 0.0)
    std_reward = metrics.get('std_reward', 0.0)

    if mean_reward <= 0:
        return 0.0

    weight = 1.0 / (1.0 + std_reward / max(1.0, mean_reward))
    return mean_reward * weight


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.

    Suggests parameters from search space, runs training, returns
    stability-weighted reward metric.

    Returns:
        Stability-weighted reward (higher is better).
    """
    trial_num = trial.number
    start_time = time.time()

    # Set alarm-based timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TRIAL_TIMEOUT_SECONDS)

    try:
        # Suggest parameters
        params = _suggest_params(trial)

        # Run training trial
        metrics = run_trial(
            params=params,
            trial_id=trial_num,
            timeout=TRIAL_TIMEOUT_SECONDS,
        )

        # Check for error status
        if metrics.get('status') == 'error':
            print(f"Trial {trial_num} reported error: {metrics.get('error', 'unknown')}")
            raise RuntimeError(metrics.get('error', 'unknown'))

        # Compute stability-weighted reward
        reward = _stability_weighted_reward(metrics)

        # Report intermediate values to Optuna
        trial.set_user_attr('max_reward', metrics.get('max_reward', 0.0))
        trial.set_user_attr('mean_reward', metrics.get('mean_reward', 0.0))
        trial.set_user_attr('std_reward', metrics.get('std_reward', 0.0))
        trial.set_user_attr('elapsed_seconds', metrics.get('elapsed_seconds', 0.0))
        trial.set_user_attr('status', metrics.get('status', 'unknown'))
        trial.set_user_attr('gpu_memory_allocated_mb', metrics.get('gpu_memory_allocated_mb', 0.0))
        trial.set_user_attr('gpu_memory_peak_mb', metrics.get('gpu_memory_peak_mb', 0.0))
        trial.set_user_attr('gpu_memory_change_mb', metrics.get('gpu_memory_change_mb', 0.0))

        return reward

    except TrialTimeoutError as e:
        print(f"Trial {trial_num} timed out: {e}")
        raise optuna.TrialPruned() from e

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Trial {trial_num} failed after {elapsed:.0f}s: {e}")
        trial.set_user_attr('status', 'error')
        trial.set_user_attr('error', str(e))
        trial.set_user_attr('elapsed_seconds', elapsed)
        raise

    finally:
        signal.alarm(0)  # Cancel alarm
        signal.signal(signal.SIGALRM, old_handler)
        elapsed = time.time() - start_time
        print(f"Trial {trial_num} wrapper done in {elapsed:.0f}s")

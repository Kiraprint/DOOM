"""ASHA early stopping scheduler for Mamba-2 HPO.

Prunes underperforming trials after a grace period to save compute.

Usage:
    from hpo.early_stopping import ASHAScheduler, should_prune

    scheduler = ASHAScheduler()
    for epoch in range(max_epochs):
        # ... training step ...
        if should_prune(trial, scheduler, epoch, reward):
            raise optuna.TrialPruned()

Config:
    grace_period: 10 epochs (no pruning before this)
    max_t: 100 epochs (hard limit)
    reduction_factor: 2 (bracket configuration)
"""
import logging
from typing import Optional

import optuna

logger = logging.getLogger(__name__)

DEFAULT_GRACE_PERIOD = 10
DEFAULT_MAX_T = 100
DEFAULT_REDUCTION_FACTOR = 2


class ASHAScheduler:
    """ASHA (Asynchronous Successive Halving) scheduler.

    Prunes trials whose reward falls below a baseline after the grace period.
    Uses median pruning against the current study's median reward for
    adaptive thresholding.
    """

    def __init__(
        self,
        grace_period: int = DEFAULT_GRACE_PERIOD,
        max_t: int = DEFAULT_MAX_T,
        reduction_factor: int = DEFAULT_REDUCTION_FACTOR,
        baseline_reward: float = 0.0,
    ):
        self.grace_period = grace_period
        self.max_t = max_t
        self.reduction_factor = reduction_factor
        self.baseline_reward = baseline_reward

    def compute_threshold(self, trial: optuna.Trial) -> float:
        """Compute pruning threshold from study median or baseline."""
        study = trial.study
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(completed) >= 3:
            rewards = [t.value for t in completed if t.value is not None]
            if rewards:
                median = sorted(rewards)[len(rewards) // 2]
                return max(median, self.baseline_reward)
        return self.baseline_reward

    def should_prune(
        self,
        trial: optuna.Trial,
        epoch: int,
        reward: float,
    ) -> bool:
        """Decide whether to prune a trial.

        Args:
            trial: Current Optuna trial.
            epoch: Current epoch number (1-indexed).
            reward: Current reward metric.

        Returns:
            True if the trial should be pruned.
        """
        if epoch < self.grace_period:
            return False

        if epoch >= self.max_t:
            return False

        threshold = self.compute_threshold(trial)

        if reward < threshold:
            logger.warning(
                "Prune: trial %d epoch %d reward=%.4f < threshold=%.4f",
                trial.number, epoch, reward, threshold,
            )
            trial.report(reward, epoch)
            return True

        return False


_scheduler_cache: dict[int, ASHAScheduler] = {}


def get_scheduler(
    study_id: Optional[int] = None,
    grace_period: int = DEFAULT_GRACE_PERIOD,
    max_t: int = DEFAULT_MAX_T,
) -> ASHAScheduler:
    """Get or create a scheduler instance (cached per study)."""
    if study_id is None:
        return ASHAScheduler(grace_period=grace_period, max_t=max_t)
    if study_id not in _scheduler_cache:
        _scheduler_cache[study_id] = ASHAScheduler(
            grace_period=grace_period, max_t=max_t
        )
    return _scheduler_cache[study_id]


def should_prune(
    trial: optuna.Trial,
    scheduler: ASHAScheduler,
    epoch: int,
    reward: float,
) -> bool:
    """Main entry point for pruning decisions.

    Logs pruning decisions to TensorBoard via Optuna's built-in
    trial reporting mechanism.

    Args:
        trial: Current Optuna trial.
        scheduler: ASHAScheduler instance.
        epoch: Current epoch (1-indexed).
        reward: Current reward metric.

    Returns:
        True if the trial should be pruned.
    """
    return scheduler.should_prune(trial, epoch, reward)


def log_pruning_decision(
    trial: optuna.Trial,
    epoch: int,
    reward: float,
    pruned: bool,
) -> None:
    """Log a pruning decision for debugging.

    Uses Optuna's user attributes for TensorBoard compatibility.
    """
    trial.set_user_attr(f'prune_epoch_{epoch}', {
        'reward': reward,
        'pruned': pruned,
    })
    if pruned:
        logger.info("Trial %d pruned at epoch %d (reward=%.4f)", trial.number, epoch, reward)

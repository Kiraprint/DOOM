"""Optuna Study Configuration for Mamba-2 HPO.

Provides study setup with TPE sampler, stability-weighted objective,
and persistent storage for resumption.
"""

import optuna
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage, JournalStorage
from pathlib import Path


STORAGE_URL = "sqlite:///hpo_study.db"
STUDY_NAME = "mamba2_hpo"
STABILITY_BONUS_WEIGHT = 0.1


def compute_stability_bonus(trial):
    """Compute stability bonus from trial attributes.
    
    Stability bonus rewards low variance in reward across evaluation episodes.
    Formula: bonus = weight * (1 - normalized_std)
    
    Args:
        trial: Optuna trial object with user_attrs containing reward_std
        
    Returns:
        Stability bonus value between 0 and weight
    """
    reward_std = trial.user_attrs.get("reward_std", None)
    if reward_std is None:
        return 0.0
    
    # Normalize std to [0, 1] range (assuming max reasonable std is 5.0)
    normalized_std = min(reward_std / 5.0, 1.0)
    return STABILITY_BONUS_WEIGHT * (1.0 - normalized_std)


def compute_objective_value(mean_reward, trial):
    """Compute final objective value with stability bonus.
    
    Combines mean reward with stability bonus to encourage
    not just high performance but also consistent performance.
    
    Args:
        mean_reward: Mean reward from training trial
        trial: Optuna trial object
        
    Returns:
        Combined objective value to maximize
    """
    stability = compute_stability_bonus(trial)
    return mean_reward + stability


def create_study(storage_url=None, study_name=None):
    """Create Optuna study with TPE sampler configuration.
    
    Configures:
    - TPESampler with n_startup_trials=10 for sample-efficient search
    - Maximization direction (higher reward + stability is better)
    - Persistent storage for resumption after interruptions
    
    Args:
        storage_url: Override storage URL (default: SQLite in hpo_study.db)
        study_name: Override study name (default: "mamba2_hpo")
        
    Returns:
        Configured Optuna study instance
    """
    storage_url = storage_url or STORAGE_URL
    study_name = study_name or STUDY_NAME
    
    try:
        storage = RDBStorage(storage_url, engine_kwargs={"pool_pre_ping": True})
    except Exception:
        # Fallback to journal storage if RDB fails
        storage_path = Path("hpo_study_storage.json")
        storage = JournalStorage(storage_path)
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(
            n_startup_trials=10,
            seed=42,
            consider_prior=True,
            consider_magic_clip=True,
            consider_endpoints=True,
            n_ei_candidates=24,
        ),
        storage=storage,
        load_if_exists=True,
    )
    
    return study


def load_study(storage_url=None, study_name=None):
    """Load existing study from storage.
    
    Args:
        storage_url: Storage URL (default: SQLite in hpo_study.db)
        study_name: Study name (default: "mamba2_hpo")
        
    Returns:
        Existing Optuna study instance
        
    Raises:
        KeyError: If study doesn't exist
    """
    storage_url = storage_url or STORAGE_URL
    study_name = study_name or STUDY_NAME
    
    return optuna.load_study(
        study_name=study_name,
        storage=storage_url,
    )


if __name__ == "__main__":
    study = create_study()
    print(f"Study created: {study.study_name}")
    print(f"Direction: {study.direction}")
    print(f"Sampler: {study.sampler}")
    print(f"Trials: {len(study.trials)}")

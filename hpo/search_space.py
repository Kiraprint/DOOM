"""Mamba-2 HPO Search Space. Optuna-compatible."""


def get_search_space() -> dict:
    """HPO searches model params only. Env/worker/recurrence/batch_size fixed.
    
    Fixed baseline: 8 workers × 8 envs × rollout 64 = batch 4096, recurrence=32
    """
    return {
        "mamba_d_model": ("categorical", [256, 512, 1024]),
        "mamba_d_state": ("categorical", [64, 128]),
        "mamba_headdim": ("categorical", [64, 128]),
        "mamba_expand": ("categorical", [1, 2]),
        "learning_rate": ("loguniform", 1e-5, 1e-3),
        "exploration_loss_coeff": ("loguniform", 1e-4, 1e-2),
        "weight_decay": ("loguniform", 1e-4, 1e-1),
        "optimizer": ("categorical", ["adam", "lamb", "adamw"]),
        "rnn_num_layers": ("categorical", [1, 2, 3]),  # Memory blocks (1 = current default)
        # batch_size fixed at 4096 (8 workers × 8 envs × rollout 64)
        # recurrence fixed at 32 (divides rollout=64 evenly)
    }


def suggest(trial, param_name, spec):
    dist_type = spec[0]
    if dist_type == "categorical":
        return trial.suggest_categorical(param_name, spec[1])
    if dist_type == "loguniform":
        return trial.suggest_float(param_name, spec[1], spec[2], log=True)
    raise ValueError(f"Unknown distribution: {dist_type}")

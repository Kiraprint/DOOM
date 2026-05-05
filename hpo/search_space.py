"""Mamba-2 HPO Search Space. Optuna-compatible. No KL adaptive scheduling."""


def get_search_space() -> dict:
    """Return search space. Values are (dist_type, *args) tuples."""
    return {
        "d_model": ("categorical", [256, 512, 1024]),
        "d_state": ("categorical", [64, 128]),
        "headdim": ("categorical", [64, 128]),
        "expand": ("categorical", [1, 2]),
        "learning_rate": ("loguniform", 1e-5, 1e-3),
        "exploration_loss_coeff": ("loguniform", 1e-4, 1e-2),
        "batch_size": ("categorical", [2048, 4096]),
    }


def suggest(trial, param_name, spec):
    dist_type = spec[0]
    if dist_type == "categorical":
        return trial.suggest_categorical(param_name, spec[1])
    if dist_type == "loguniform":
        return trial.suggest_float(param_name, spec[1], spec[2], log=True)
    raise ValueError(f"Unknown distribution: {dist_type}")

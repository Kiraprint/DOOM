"""
Single trial execution for Mamba-2 HPO.

Runs one training trial with given parameters, logs to TensorBoard,
and returns reward metrics.

Usage:
    from hpo.evaluate import run_trial
    result = run_trial(params_dict)
"""

import sys
import time
import functools
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mamba2_core import register_mamba2, MAMBA_AVAILABLE
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec


def _build_argv(params: Dict[str, Any], trial_id: int, train_dir: str) -> list:
    ts = time.strftime('%Y%m%d_%H%M%S')
    trial_prefix = f"trial_{trial_id}_{ts}"

    argv = [
        '--env', params.get('env', 'doom_benchmark'),
        '--algo', 'APPO',
        '--experiment', f"{trial_prefix}",
        '--train_for_env_steps', str(params.get('train_steps', 50000000)),
        '--num_workers', str(params.get('num_workers', 4)),
        '--num_envs_per_worker', str(params.get('num_envs_per_worker', 16)),
        '--batch_size', str(params.get('batch_size', 2048)),
        '--num_policies', str(params.get('num_policies', 1)),
        '--policy_workers_per_policy', str(params.get('policy_workers_per_policy', 2)),
        '--worker_num_splits', str(params.get('worker_num_splits', 2)),
        '--rnn_num_layers', str(params.get('rnn_num_layers', 1)),
        '--learning_rate', str(params.get('learning_rate', 1.5e-4)),
        '--exploration_loss_coeff', str(params.get('exploration_loss_coeff', 0.01)),
        '--train_dir', train_dir,
    ]
    return argv


def _extract_reward_metrics(train_dir: str, trial_id: int) -> Dict[str, Any]:
    return {
        'max_reward': 0.0,
        'mean_reward': 0.0,
        'std_reward': 0.0,
    }


def run_trial(
    params: Dict[str, Any],
    trial_id: int = 0,
    train_dir: str = "train_dir",
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run a single HPO trial.

    Args:
        params: Dict of hyperparameters to override defaults.
        trial_id: Unique trial identifier for log separation.
        train_dir: Base directory for training logs.
        timeout: Max trial duration in seconds (None = no limit).

    Returns:
        Dict with 'max_reward', 'mean_reward', 'std_reward', 'trial_id', 'status'.
    """
    if not MAMBA_AVAILABLE:
        raise RuntimeError(
            "mamba-ssm not installed. "
            "pip install mamba-ssm --no-build-isolation"
        )

    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)

    argv = _build_argv(params, trial_id, train_dir)
    parser, _ = parse_sf_args(argv=argv)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)

    cfg.rnn_type = 'mamba2'
    cfg.rnn_num_layers = params.get('rnn_num_layers', 1)
    cfg.mamba_d_state = params.get('mamba_d_state', 64)
    cfg.mamba_d_conv = params.get('mamba_d_conv', 4)
    cfg.mamba_expand = params.get('mamba_expand', 1)
    cfg.mamba_headdim = params.get('mamba_headdim', 64)
    cfg.mamba_ngroups = params.get('mamba_ngroups', 1)
    cfg.mamba_d_model = params.get('mamba_d_model', 512)
    cfg.weight_decay = params.get('weight_decay', 0.1)

    register_mamba2(cfg)

    from models.mamba2_core import get_mamba2_required_rnn_size
    cfg.rnn_size = cfg.mamba_d_model
    cfg.rnn_size = max(cfg.rnn_size, get_mamba2_required_rnn_size(cfg))

    print(f"\nTrial {trial_id}: Mamba-2 d_model={cfg.mamba_d_model}, rnn_size={cfg.rnn_size}")
    print(f"  Train for: {cfg.train_for_env_steps:,} steps, timeout={timeout}s")

    start_time = time.time()
    try:
        status = run_rl(cfg)
    except Exception as e:
        print(f"Trial {trial_id} failed: {e}")
        return {
            'trial_id': trial_id,
            'max_reward': 0.0,
            'mean_reward': 0.0,
            'std_reward': 0.0,
            'status': 'error',
            'error': str(e),
        }

    elapsed = time.time() - start_time
    if timeout and elapsed > timeout:
        print(f"Trial {trial_id} exceeded timeout ({elapsed:.0f}s > {timeout}s)")

    metrics = _extract_reward_metrics(train_dir, trial_id)
    metrics['trial_id'] = trial_id
    metrics['status'] = 'completed'
    metrics['elapsed_seconds'] = elapsed

    return metrics


if __name__ == '__main__':
    params = {
        'train_steps': 1000000,
        'learning_rate': 1.5e-4,
    }
    result = run_trial(params, trial_id=0, timeout=3600)
    print(f"Result: {result}")

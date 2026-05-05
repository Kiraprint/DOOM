"""
Mamba-2 training script for VizDoom with sample-factory.

Trains APPO with Mamba-2 sequence model on Doom Benchmark for 50M steps.

Usage:
    cd /home/kir/Code/DOOM && python -m models.train_mamba2
"""

import sys
import json
import functools
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
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


def parse_hpo_args():
    """Parse HPO-specific CLI arguments (separate from sample-factory parser)."""
    parser = argparse.ArgumentParser(description='Mamba-2 HPO Training', add_help=False)
    parser.add_argument('--hpo_trial_id', type=str, default=None,
                        help='HPO trial identifier for trial-specific logging')
    parser.add_argument('--hpo_params', type=str, default=None,
                        help='JSON string of Mamba-2 parameters to override (e.g. \'{"mamba_d_state": 128}\')')
    return parser.parse_known_args()


def apply_hpo_params(cfg, hpo_params):
    """Apply HPO parameter overrides to config."""
    if not hpo_params:
        return
    print(f"  HPO params: {hpo_params}")
    mamba_keys = {
        'mamba_d_state', 'mamba_d_conv', 'mamba_expand',
        'mamba_headdim', 'mamba_ngroups', 'mamba_d_model',
        'rnn_num_layers', 'learning_rate', 'weight_decay',
        'exploration_loss_coeff', 'batch_size',
    }
    for key, value in hpo_params.items():
        if key in mamba_keys and hasattr(cfg, key):
            old = getattr(cfg, key)
            setattr(cfg, key, value)
            if old != value:
                print(f"    {key}: {old} -> {value}")


def report_trial_metrics(hpo_trial_id, status, cfg):
    """Report trial completion metrics for HPO."""
    if not hpo_trial_id:
        return
    log_dir = Path(cfg.save_dir) if hasattr(cfg, 'save_dir') else Path('./train_dir')
    metrics_file = log_dir / f'hpo_trial_{hpo_trial_id}_metrics.json'
    metrics = {
        'trial_id': hpo_trial_id,
        'status': str(status),
        'success': status == 0,
        'params': {
            'mamba_d_state': cfg.mamba_d_state,
            'mamba_d_conv': cfg.mamba_d_conv,
            'mamba_expand': cfg.mamba_expand,
            'mamba_headdim': cfg.mamba_headdim,
            'mamba_ngroups': cfg.mamba_ngroups,
            'mamba_d_model': cfg.mamba_d_model,
            'rnn_num_layers': cfg.rnn_num_layers,
            'learning_rate': cfg.learning_rate,
            'weight_decay': cfg.weight_decay,
        },
    }
    try:
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Trial metrics written to: {metrics_file}")
    except Exception as e:
        print(f"  WARNING: Failed to write trial metrics: {e}")


def main():
    # Parse HPO args first (separate from sample-factory parser)
    hpo_args, remaining_argv = parse_hpo_args()
    hpo_trial_id = hpo_args.hpo_trial_id
    hpo_params = None
    if hpo_args.hpo_params:
        try:
            hpo_params = json.loads(hpo_args.hpo_params)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid --hpo_params JSON: {e}")
            return

    if not MAMBA_AVAILABLE:
        print("ERROR: mamba-ssm not installed.")
        print("Install with: pip install mamba-ssm --no-build-isolation")
        return

    # Register Doom environments
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)

    # Register Doom encoder
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)

    # Parse config
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f'doom_battle_mamba2_50m_{ts}'
    if hpo_trial_id:
        experiment_name = f'hpo_{hpo_trial_id}_{ts}'

    argv = [
        '--env', 'doom_benchmark',
        '--algo', 'APPO',
        '--experiment', experiment_name,
        '--train_for_env_steps', '50000000',
        '--num_workers', '4',
        '--num_envs_per_worker', '16',
        '--batch_size', '2048',
        '--num_policies', '1',
        '--policy_workers_per_policy', '2',
        '--worker_num_splits', '2',
        '--rnn_num_layers', '1',
        '--learning_rate', '1.5e-4',
        '--exploration_loss_coeff', '0.01',
    ]
    parser, _ = parse_sf_args(argv=argv)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)

    # Mamba-2 specific settings (MUST be before register_mamba2 so cfg is frozen)
    cfg.rnn_type = 'mamba2'
    cfg.rnn_num_layers = 1
    cfg.mamba_d_state = 64
    cfg.mamba_d_conv = 4
    cfg.mamba_expand = 1
    cfg.mamba_headdim = 64
    cfg.mamba_ngroups = 1
    cfg.mamba_d_model = 512

    apply_hpo_params(cfg, hpo_params)

    # Register Mamba-2 core — passes cfg so rnn_num_layers is frozen
    register_mamba2(cfg)

    # d_model (computation dimension) is separate from rnn_size (state storage)
    # core_output_size stays at d_model for decoder compatibility
    cfg.rnn_size = cfg.mamba_d_model  # Set first for calculation

    from models.mamba2_core import get_mamba2_required_rnn_size
    required_size = get_mamba2_required_rnn_size(cfg)

    # get_mamba2_required_rnn_size returns per-layer size.
    # sample-factory's get_rnn_size() multiplies rnn_size * rnn_num_layers
    # when allocating buffers. So rnn_size must be per-layer, not total.
    cfg.rnn_size = max(cfg.rnn_size, required_size)
    print(f"  Mamba-2 d_model: {cfg.mamba_d_model}, rnn_size (per-layer): {cfg.rnn_size} (total: {required_size * cfg.rnn_num_layers})")

    # CRITICAL: Weight decay prevents B/C norm divergence in Mamba-2
    cfg.weight_decay = 0.1

    print(f"\nStarting Mamba-2 training on {cfg.env}")
    print(f"  Algorithm: {cfg.algo}")
    print(f"  RNN type: {cfg.rnn_type} (d_model={cfg.mamba_d_model}, rnn_size={cfg.rnn_size})")
    print(f"  Train for: {cfg.train_for_env_steps:,} env steps")
    print(f"  Workers: {cfg.num_workers} x {cfg.num_envs_per_worker} envs")
    print()

    status = run_rl(cfg)
    print(f"\nTraining completed with status: {status}")

    report_trial_metrics(hpo_trial_id, status, cfg)


if __name__ == '__main__':
    main()

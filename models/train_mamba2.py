"""
Mamba-2 training script for VizDoom with sample-factory.

Trains APPO with Mamba-2 sequence model on Doom Benchmark for 50M steps.

Usage:
    cd /home/kir/Code/DOOM && python -m models.train_mamba2
"""

import sys
import functools
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Timestamp for log naming
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

from models.mamba2_core import register_mamba2, MAMBA_AVAILABLE
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec


def main():
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

    # Register Mamba-2 core
    register_mamba2()

    # Mamba-2 specific settings (MUST be set before parsing to ensure correct rnn_size)
    cfg_rnn_num_layers = 2  # Increased from 1 for better temporal reasoning
    cfg_mamba_d_state = 16
    cfg_mamba_d_conv = 4
    cfg_mamba_expand = 2
    cfg_mamba_headdim = 32
    cfg_mamba_ngroups = 1
    cfg_mamba_d_model = 512  # Increased from 256 for better representation

    # Calculate required rnn_size before parsing
    from models.mamba2_core import get_mamba2_required_rnn_size
    temp_cfg = SimpleNamespace(
        rnn_size=cfg_mamba_d_model,
        rnn_num_layers=cfg_rnn_num_layers,
        mamba_d_model=cfg_mamba_d_model,
        mamba_d_state=cfg_mamba_d_state,
        mamba_d_conv=cfg_mamba_d_conv,
        mamba_expand=cfg_mamba_expand,
        mamba_headdim=cfg_mamba_headdim,
        mamba_ngroups=cfg_mamba_ngroups,
    )
    required_rnn_size = get_mamba2_required_rnn_size(temp_cfg)

    # Parse config with correct rnn_size
    argv = [
        '--env', 'doom_benchmark',
        '--algo', 'APPO',
        '--experiment', f'doom_battle_mamba2_v2_250m_{TIMESTAMP}',
        '--train_for_env_steps', '250000000',
        '--num_workers', '8',
        '--num_envs_per_worker', '16',
        '--batch_size', '4096',
        '--num_policies', '1',
        '--policy_workers_per_policy', '2',
        '--worker_num_splits', '2',
        '--rnn_size', str(required_rnn_size),  # Set here so sample-factory allocates correct buffer
    ]
    parser, _ = parse_sf_args(argv=argv)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)

    # Apply Mamba-2 settings
    cfg.rnn_type = 'mamba2'
    cfg.mamba_d_state = cfg_mamba_d_state
    cfg.mamba_d_conv = cfg_mamba_d_conv
    cfg.mamba_expand = cfg_mamba_expand
    cfg.mamba_headdim = cfg_mamba_headdim
    cfg.mamba_ngroups = cfg_mamba_ngroups
    cfg.mamba_d_model = cfg_mamba_d_model

    # CRITICAL: Store actual Mamba layer count before overriding rnn_num_layers.
    # Sample Factory's get_rnn_size() multiplies rnn_size * rnn_num_layers for buffer
    # allocation. Mamba2Core already encodes all layers into rnn_size, so we must set
    # rnn_num_layers=1 to prevent double-counting. Mamba2Core reads mamba_num_layers
    # for its internal layer count.
    cfg.mamba_num_layers = cfg_rnn_num_layers
    cfg.rnn_num_layers = 1

    print(f"  Mamba-2 d_model: {cfg.mamba_d_model}, rnn_size: {cfg.rnn_size} (required: {required_rnn_size})")

    # Entropy coefficient skipped - requires environment-specific tuning
    # cfg.entropy_coef = 0.01  # Uncomment if needed after testing

    # Verify rnn_size matches what sample-factory will allocate
    print(f"  Final rnn_size: {cfg.rnn_size} (required: {required_rnn_size})")
    assert cfg.rnn_size >= required_rnn_size, f"rnn_size too small: {cfg.rnn_size} < {required_rnn_size}"

    # Mamba-2 is more sensitive to learning rate than GRU.
    # Lower LR prevents gradient explosion and improves stability.
    cfg.learning_rate = 5e-5

    # Increase max_grad_norm — Mamba-2 can have larger gradients
    cfg.max_grad_norm = 10.0

    # Use constant LR schedule — Mamba-2 benefits from stable LR
    cfg.lr_schedule = "constant"

    print(f"\nStarting Mamba-2 training on {cfg.env}")
    print(f"  Algorithm: {cfg.algo}")
    print(f"  RNN type: {cfg.rnn_type} (d_model={cfg.mamba_d_model}, rnn_size={cfg.rnn_size})")
    print(f"  Train for: {cfg.train_for_env_steps:,} env steps")
    print(f"  Workers: {cfg.num_workers} x {cfg.num_envs_per_worker} envs")
    print()

    status = run_rl(cfg)
    print(f"\nTraining completed with status: {status}")


if __name__ == '__main__':
    main()

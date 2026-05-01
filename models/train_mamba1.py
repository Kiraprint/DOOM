"""
Mamba-1 training script for VizDoom with sample-factory.

Trains APPO with Mamba-1 sequence model on Doom Benchmark for 50M steps.

Usage:
    cd /home/kir/Code/DOOM && python -m models.train_mamba1
"""

import sys
import functools
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mamba1_core import register_mamba1, MAMBA_AVAILABLE
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

    # Register Mamba-1 core
    register_mamba1()

    # Calculate required rnn_size BEFORE parsing (needed for CLI args)
    from models.mamba1_core import get_mamba1_required_rnn_size
    from types import SimpleNamespace

    pre_cfg = SimpleNamespace()
    pre_cfg.rnn_size = 256
    pre_cfg.mamba_d_model = 256
    pre_cfg.rnn_num_layers = 1
    pre_cfg.mamba_d_state = 16
    pre_cfg.mamba_d_conv = 4
    pre_cfg.mamba_expand = 2
    required_size = get_mamba1_required_rnn_size(pre_cfg)

    # Parse config matching original command params
    # rnn_size must be passed via CLI to survive checkpoint loading
    argv = [
        '--env', 'doom_benchmark',
        '--algo', 'APPO',
        '--experiment', 'doom_battle_mamba1_v2_50m',
        '--train_for_env_steps', '50000000',
        '--num_workers', '10',
        '--num_envs_per_worker', '32',
        '--batch_size', '4096',
        '--num_policies', '1',
        '--policy_workers_per_policy', '2',
        '--worker_num_splits', '2',
        '--rnn_size', str(required_size),
    ]
    parser, _ = parse_sf_args(argv=argv)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)

    # Mamba-1 specific settings
    cfg.rnn_type = 'mamba1'
    cfg.mamba_num_layers = 1
    cfg.rnn_num_layers = 1  # Must be 1 to prevent SF double-counting
    cfg.mamba_d_state = 16
    cfg.mamba_d_conv = 4
    cfg.mamba_expand = 2
    cfg.mamba_ngroups = 1

    # d_model (computation dimension) is separate from rnn_size (state storage)
    # core_output_size stays at d_model for decoder compatibility
    cfg.mamba_d_model = 256  # Computation dimension

    print(f"\nStarting Mamba-1 training on {cfg.env}")
    print(f"  Algorithm: {cfg.algo}")
    print(f"  RNN type: {cfg.rnn_type} (d_model={cfg.mamba_d_model}, rnn_size={cfg.rnn_size})")
    print(f"  Train for: {cfg.train_for_env_steps:,} env steps")
    print(f"  Workers: {cfg.num_workers} x {cfg.num_envs_per_worker} envs")
    print()

    status = run_rl(cfg)
    print(f"\nTraining completed with status: {status}")


if __name__ == '__main__':
    main()

"""Quick integration test for Mamba-2 with sample-factory."""

import functools
import torch
from torch.nn.utils.rnn import PackedSequence

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.model.core import ModelCore

from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec

from models.mamba2_core import Mamba2Core, MAMBA_AVAILABLE


def main():
    if not MAMBA_AVAILABLE:
        print("ERROR: mamba-ssm not installed")
        return

    # Register doom envs
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)

    # Register doom encoder
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)

    # Parse config
    argv = [
        '--env', 'doom_benchmark',
        '--algo', 'APPO',
        '--experiment', 'test_mamba2_integration',
        '--train_for_env_steps', '1000',  # Very short for testing
        '--num_workers', '2',
        '--num_envs_per_worker', '2',
        '--batch_size', '256',
        '--rollout', '16',
        '--recurrence', '16',
    ]
    parser, _ = parse_sf_args(argv=argv)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)

    # Register mamba-2 core
    orig_make_core = global_model_factory().make_model_core_func

    def make_core_with_mamba(cfg, core_input_size):
        if cfg.use_rnn and getattr(cfg, 'rnn_type', 'gru') == 'mamba2':
            return Mamba2Core(cfg, core_input_size)
        return orig_make_core(cfg, core_input_size)

    global_model_factory().register_model_core_factory(make_core_with_mamba)

    # Override rnn_type to mamba2
    cfg.rnn_type = 'mamba2'
    cfg.rnn_size = 128  # Smaller for faster testing

    # Add mamba config
    cfg.mamba_d_state = 16
    cfg.mamba_d_conv = 4
    cfg.mamba_expand = 2
    cfg.mamba_headdim = 32
    cfg.mamba_ngroups = 1

    # Run training
    print("Starting training with Mamba-2...")
    print(f"Config: env={cfg.env}, algo={cfg.algo}, rnn_type={cfg.rnn_type}")
    print(f"Training for {cfg.train_for_env_steps} env steps")

    status = run_rl(cfg)
    print(f"Training completed with status: {status}")


if __name__ == '__main__':
    main()

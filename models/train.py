"""Unified experiment runner: GRU, Mamba-2, Mamba-1, Transformer, or Perceiver IO.
Full HP override support for fair comparison.

Usage:
    # Cell 1: GRU + defaults (extra seeds)
    python -m models.train --rnn_type gru --experiment gru_baseline_50m_seed2

    # Cell 2: GRU + optimized HPs
    python -m models.train --rnn_type gru --optimizer adamw \
        --learning_rate 4.05e-4 --exploration_loss_coeff 0.00202 \
        --weight_decay 0.00196 --experiment gru_optimized_50m

    # Cell 3: Mamba-2 best config
    python -m models.train --rnn_type mamba2 \
        --mamba_d_model 512 --mamba_d_state 128 --mamba_headdim 128 \
        --mamba_expand 1 --optimizer adamw --learning_rate 4.05e-4 \
        --exploration_loss_coeff 0.00202 --weight_decay 0.00196 \
        --experiment mamba2_hpo_best_50m

    # Mamba-1
    python -m models.train --rnn_type mamba1 \
        --mamba1_d_model 512 --mamba1_d_state 16 --mamba1_d_conv 4 \
        --mamba1_expand 2 --optimizer adamw --learning_rate 4.05e-4 \
        --experiment mamba1_50m

    # Transformer
    python -m models.train --rnn_type transformer \
        --transformer_d_model 512 --transformer_nhead 8 \
        --transformer_window_size 64 --optimizer adamw \
        --learning_rate 4.05e-4 --experiment transformer_50m

    # Perceiver IO
    python -m models.train --rnn_type perceiver \
        --perceiver_d_model 512 --perceiver_num_latents 32 \
        --perceiver_d_latents 512 --perceiver_num_blocks 2 \
        --optimizer adamw --learning_rate 4.05e-4 \
        --experiment perceiver_50m
"""

import sys
import json
import argparse
import functools
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rnn_type", default="gru",
                        choices=["gru", "mamba2", "mamba1", "transformer", "perceiver"])
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--gradient_checkpointing", type=str, default=None,
                        choices=["True", "False"], help="Override gradient checkpointing")
    parser.add_argument("--train_for_env_steps", type=int, default=None,
                        help="Override training steps")
    # Mamba-2 params
    parser.add_argument("--mamba_d_model", type=int, default=512)
    parser.add_argument("--mamba_d_state", type=int, default=64)
    parser.add_argument("--mamba_d_conv", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=1)
    parser.add_argument("--mamba_headdim", type=int, default=64)
    parser.add_argument("--mamba_ngroups", type=int, default=1)
    # Mamba-1 params
    parser.add_argument("--mamba1_d_model", type=int, default=512)
    parser.add_argument("--mamba1_d_state", type=int, default=16)
    parser.add_argument("--mamba1_d_conv", type=int, default=4)
    parser.add_argument("--mamba1_expand", type=int, default=2)
    # Transformer params
    parser.add_argument("--transformer_d_model", type=int, default=512)
    parser.add_argument("--transformer_nhead", type=int, default=8)
    parser.add_argument("--transformer_num_layers", type=int, default=1)
    parser.add_argument("--transformer_window_size", type=int, default=64)
    parser.add_argument("--transformer_dim_feedforward", type=int, default=2048)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    # Perceiver IO params
    parser.add_argument("--perceiver_d_model", type=int, default=512)
    parser.add_argument("--perceiver_num_latents", type=int, default=32)
    parser.add_argument("--perceiver_d_latents", type=int, default=512)
    parser.add_argument("--perceiver_num_blocks", type=int, default=2)
    parser.add_argument("--perceiver_num_heads", type=int, default=8)
    parser.add_argument("--perceiver_dropout", type=float, default=0.1)
    extra_args, remaining = parser.parse_known_args()

    # Register Doom environments and encoder
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)

    # SF parser only accepts gru/lstm for --rnn_type.
    # For custom rnn_types, pass gru to satisfy validation, then override after.
    sf_rnn_type = extra_args.rnn_type
    if sf_rnn_type in ("mamba2", "mamba1", "transformer", "perceiver"):
        sf_rnn_type = "gru"

    # Build SF args with fixed env/worker config
    argv = remaining + [
        "--env", "doom_benchmark",
        "--algo", "APPO",
        "--train_for_env_steps", "50000000",
        "--num_workers", "8",
        "--num_envs_per_worker", "8",
        "--batch_size", "4096",
        "--num_policies", "1",
        "--policy_workers_per_policy", "2",
        "--worker_num_splits", "2",
        "--rollout", "64",
        "--rnn_num_layers", "1",
        "--recurrence", "32",
        "--use_rnn", "True",
        "--rnn_type", sf_rnn_type,
    ]
    parser_sf, _ = parse_sf_args(argv=argv)
    add_doom_env_args(parser_sf)
    doom_override_defaults(parser_sf)
    cfg = parse_full_cfg(parser_sf, argv)

    # Apply extra params
    if extra_args.weight_decay is not None:
        cfg.weight_decay = extra_args.weight_decay
    if extra_args.gradient_checkpointing is not None:
        cfg.gradient_checkpointing = extra_args.gradient_checkpointing == "True"
    if extra_args.train_for_env_steps is not None:
        cfg.train_for_env_steps = extra_args.train_for_env_steps

    cfg.rnn_type = extra_args.rnn_type
    cfg.rnn_num_layers = 1
    cfg.rnn_size = 512

    if extra_args.rnn_type == "mamba2":
        from models.mamba2_core import register_mamba2, get_mamba2_required_rnn_size
        cfg.mamba_d_model = extra_args.mamba_d_model
        cfg.mamba_d_state = extra_args.mamba_d_state
        cfg.mamba_d_conv = extra_args.mamba_d_conv
        cfg.mamba_expand = extra_args.mamba_expand
        cfg.mamba_headdim = extra_args.mamba_headdim
        cfg.mamba_ngroups = extra_args.mamba_ngroups
        register_mamba2(cfg)
        cfg.rnn_size = max(cfg.rnn_size, get_mamba2_required_rnn_size(cfg))

    elif extra_args.rnn_type == "mamba1":
        from models.mamba1_core import register_mamba1, get_mamba1_required_rnn_size
        cfg.mamba1_d_model = extra_args.mamba1_d_model
        cfg.mamba1_d_state = extra_args.mamba1_d_state
        cfg.mamba1_d_conv = extra_args.mamba1_d_conv
        cfg.mamba1_expand = extra_args.mamba1_expand
        register_mamba1(cfg)
        cfg.rnn_size = max(cfg.rnn_size, get_mamba1_required_rnn_size(cfg))

    elif extra_args.rnn_type == "transformer":
        from models.transformer_core import register_transformer, get_transformer_required_rnn_size
        cfg.transformer_d_model = extra_args.transformer_d_model
        cfg.transformer_nhead = extra_args.transformer_nhead
        cfg.transformer_num_layers = extra_args.transformer_num_layers
        cfg.transformer_window_size = extra_args.transformer_window_size
        cfg.transformer_dim_feedforward = extra_args.transformer_dim_feedforward
        cfg.transformer_dropout = extra_args.transformer_dropout
        register_transformer(cfg)
        cfg.rnn_size = max(cfg.rnn_size, get_transformer_required_rnn_size(cfg))

    elif extra_args.rnn_type == "perceiver":
        from models.perceiver_core import register_perceiver, get_perceiver_required_rnn_size
        cfg.perceiver_d_model = extra_args.perceiver_d_model
        cfg.perceiver_num_latents = extra_args.perceiver_num_latents
        cfg.perceiver_d_latents = extra_args.perceiver_d_latents
        cfg.perceiver_num_blocks = extra_args.perceiver_num_blocks
        cfg.perceiver_num_heads = extra_args.perceiver_num_heads
        cfg.perceiver_dropout = extra_args.perceiver_dropout
        register_perceiver(cfg)
        cfg.rnn_size = max(cfg.rnn_size, get_perceiver_required_rnn_size(cfg))

    print(f"\nStarting experiment: {cfg.experiment}")
    print(f"  rnn_type={cfg.rnn_type}, optimizer={cfg.optimizer}, lr={cfg.learning_rate}")
    print(f"  rnn_size={cfg.rnn_size}, batch={cfg.batch_size}, workers={cfg.num_workers}x{cfg.num_envs_per_worker}")
    if hasattr(cfg, 'weight_decay'):
        print(f"  weight_decay={cfg.weight_decay}")
    print(f"  exploration_loss_coeff={cfg.exploration_loss_coeff}")
    print()

    status = run_rl(cfg)
    print(f"\nCompleted with status: {status}")


if __name__ == "__main__":
    main()

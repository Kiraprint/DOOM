"""Restart a single failed trial with exact params."""
import json
import sys
import functools
import time
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "hpo_study.db"
TRAIN_DIR = Path(__file__).resolve().parent.parent / "train_dir"


def decode_params(trial_id: int) -> dict:
    import sqlite3
    conn = sqlite3.connect(str(DB))
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT param_name, param_value, distribution_json "
            "FROM trial_params WHERE trial_id = ?",
            (trial_id,),
        ).fetchall()
        params = {}
        for row in rows:
            name = row["param_name"]
            pval = row["param_value"]
            dist = json.loads(row["distribution_json"])

            if "CategoricalDistribution" in str(dist):
                choices = dist["attributes"]["choices"]
                params[name] = choices[int(pval)]
            else:
                params[name] = pval
        return params
    finally:
        conn.close()


def main(trial_num: int):
    # Find trial_id for this trial number
    import sqlite3
    conn = sqlite3.connect(str(DB))
    try:
        trial_id = conn.execute(
            "SELECT trial_id FROM trials WHERE number = ?",
            (trial_num,),
        ).fetchone()[0]
    finally:
        conn.close()
    params = decode_params(trial_id)
    print(f"Trial #{trial_num} params: {json.dumps(params, indent=2)}")

    # Clean dev/shm
    from hpo.evaluate import _cleanup_shm
    _cleanup_shm()

    # Build argv
    ts = time.strftime('%Y%m%d_%H%M%S')
    experiment = f"trial_{trial_num}_{ts}"

    argv = [
        '--env', params.get('env', 'doom_benchmark'),
        '--algo', 'APPO',
        '--experiment', experiment,
        '--train_for_env_steps', str(params.get('train_steps', 50000000)),
        '--num_workers', '8',
        '--num_envs_per_worker', '8',
        '--batch_size', '4096',
        '--num_policies', str(params.get('num_policies', 1)),
        '--policy_workers_per_policy', str(params.get('policy_workers_per_policy', 2)),
        '--worker_num_splits', str(params.get('worker_num_splits', 2)),
        '--rollout', '64', '--recurrence', '32',
        '--rnn_num_layers', str(params.get('rnn_num_layers', 1)),
        '--learning_rate', str(params.get('learning_rate', 1.5e-4)),
        '--exploration_loss_coeff', str(params.get('exploration_loss_coeff', 0.01)),
        '--train_dir', str(TRAIN_DIR),
        '--restart_behavior', 'overwrite',
        '--async_rl', 'False',
    ]

    from sample_factory.algo.utils.context import global_model_factory
    from sample_factory.envs.env_utils import register_env
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
    from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec
    from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
    from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
    from models.mamba2_core import register_mamba2, MAMBA_AVAILABLE
    from models.mamba2_core import get_mamba2_required_rnn_size
    from sample_factory.train import run_rl

    if not MAMBA_AVAILABLE:
        raise RuntimeError("mamba-ssm not installed")

    # Register envs the same way as run_trial (with partial, not lambda)
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)

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
    cfg.rnn_size = cfg.mamba_d_model
    cfg.rnn_size = max(cfg.rnn_size, get_mamba2_required_rnn_size(cfg))

    print(f"\nTrial {trial_num}: Mamba-2 d_model={cfg.mamba_d_model}, rnn_size={cfg.rnn_size}")
    print(f"  Train for: {cfg.train_for_env_steps:,} steps")
    print(f"  Async RL: {cfg.async_rl}")

    start_time = time.time()
    try:
        status = run_rl(cfg)
    except (RuntimeError, Exception) as e:
        elapsed = time.time() - start_time
        print(f"Trial {trial_num} FAILED after {elapsed:.0f}s: {e}")
        return {'status': 'error', 'error': str(e)}

    elapsed = time.time() - start_time
    print(f"\nTrial {trial_num} COMPLETED in {elapsed:.0f}s")
    print(f"  Status: {status}")
    return {'status': 'completed', 'elapsed_seconds': elapsed}


if __name__ == "__main__":
    trial_num = int(sys.argv[1]) if len(sys.argv) > 1 else 51
    main(trial_num)

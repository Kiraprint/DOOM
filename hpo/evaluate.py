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


def _get_gpu_memory():
    """Get current GPU memory usage in MB.
    Uses nvidia-smi as primary (works across process boundaries),
    falls back to torch.cuda for same-process tracking."""
    import torch

    # Try nvidia-smi first (works across processes)
    nvidia_used, _ = _get_nvidia_smi_memory()
    if nvidia_used > 0:
        return nvidia_used, nvidia_used, nvidia_used

    # Fall back to torch (same-process only)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**2)
        return allocated, reserved, max_allocated

    return 0, 0, 0


def _get_nvidia_smi_memory():
    """Get GPU memory usage from nvidia-smi in MB."""
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            used = float(parts[0].replace('MiB', ''))
            free = float(parts[1].replace('MiB', ''))
            return used, free
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        pass
    return 0, 0


def _track_gpu_memory(tag: str):
    """Print GPU memory usage with a tag."""
    allocated, reserved, max_allocated = _get_gpu_memory()
    nvidia_used, nvidia_free = _get_nvidia_smi_memory()
    print(f"[{tag}] GPU Memory: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved, {nvidia_used:.0f}MB total")


def _cleanup_gpu():
    """Clean up GPU memory between trials."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _build_argv(params: Dict[str, Any], trial_id: int, train_dir: str) -> list:
    """Build argv with fixed env/worker/recurrence baseline + HPO model params."""
    ts = time.strftime('%Y%m%d_%H%M%S')
    trial_prefix = f"trial_{trial_id}_{ts}"

    # Fixed baseline: 8×8 envs, rollout=64, recurrence=32, batch=4096
    argv = [
        '--env', params.get('env', 'doom_benchmark'),
        '--algo', 'APPO',
        '--experiment', f"{trial_prefix}",
        '--train_for_env_steps', str(params.get('train_steps', 50000000)),
        '--num_workers', '8',
        '--num_envs_per_worker', '8',
        '--batch_size', '4096',
        '--num_policies', '1',
        '--policy_workers_per_policy', '2',
        '--worker_num_splits', '2',
        '--rollout', '64',
        '--rnn_num_layers', str(params.get('rnn_num_layers', 1)),
        '--recurrence', '32',
        '--default_niceness', '10',
        '--optimizer', params.get('optimizer', 'adam'),
        '--learning_rate', str(params.get('learning_rate', 1.5e-4)),
        '--exploration_loss_coeff', str(params.get('exploration_loss_coeff', 0.01)),
        '--train_dir', train_dir,
    ]
    return argv


def _extract_reward_metrics(train_dir: str, trial_id: int, timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Extract reward metrics from TensorBoard events for a completed trial.

    Args:
        train_dir: Base training directory (e.g., "train_dir")
        trial_id: Trial identifier
        timeout: Max trial duration (used to find correct trial subdir)

    Returns:
        Dict with 'max_reward', 'mean_reward', 'std_reward'
    """
    import statistics
    from tensorboard.backend.event_processing import event_accumulator

    # Find trial directory - try multiple patterns
    trial_dir = None
    patterns = [
        f"trial_{trial_id}_*",
        f"trial_{trial_id}.*",
    ]

    base_dir = Path(train_dir)
    for pattern in patterns:
        matches = sorted(base_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            for m in matches:
                # Check for .summary subdirectory
                summary_dir = m / ".summary" / "0"
                if summary_dir.exists():
                    trial_dir = summary_dir
                    break
                # Check directly in trial dir
                events = list(m.glob("events.out.tfevents.*"))
                if events:
                    trial_dir = m
                    break

    if trial_dir is None:
        # Last resort: scan all trial_* directories
        all_trials = sorted(base_dir.glob("trial_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for td in all_trials:
            summary_dir = td / ".summary" / "0"
            if summary_dir.exists():
                trial_dir = summary_dir
                break

    if trial_dir is None:
        return {'max_reward': 0.0, 'mean_reward': 0.0, 'std_reward': 0.0,
            'gpu_memory_allocated_mb': 0.0, 'gpu_memory_peak_mb': 0.0}

    try:
        # Find events file
        events_dir = trial_dir / ".summary" / "0" if (trial_dir / ".summary" / "0").exists() else trial_dir
        events_files = list(events_dir.glob("events.out.tfevents.*"))
        if not events_files:
            return {'max_reward': 0.0, 'mean_reward': 0.0, 'std_reward': 0.0,
                'gpu_memory_allocated_mb': 0.0, 'gpu_memory_peak_mb': 0.0}

        ea = event_accumulator.EventAccumulator(str(events_files[0]))
        ea.Reload()

        # Get reward scalars
        reward_tag = 'reward/reward'
        if reward_tag not in ea.Tags().get('scalars', []):
            return {'max_reward': 0.0, 'mean_reward': 0.0, 'std_reward': 0.0}

        reward_events = ea.Scalars(reward_tag)
        if not reward_events:
            return {'max_reward': 0.0, 'mean_reward': 0.0, 'std_reward': 0.0}

        rewards = [e.value for e in reward_events]

        return {
            'max_reward': max(rewards),
            'mean_reward': statistics.mean(rewards),
            'std_reward': statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            'gpu_memory_allocated_mb': 0.0, 'gpu_memory_peak_mb': 0.0,
        }
    except (IOError, OSError) as e:
        print(f"Warning: TensorBoard read failed: {e}")
        return {'max_reward': 0.0, 'mean_reward': 0.0, 'std_reward': 0.0}


def _cleanup_stale_processes() -> None:
    """Kill stale vizdoom and sample-factory processes from crashed trials."""
    import subprocess as _sp
    for proc_name in ("vizdoom", "doom"):
        try:
            pids = _sp.run(
                ["pgrep", "-f", proc_name], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            if pids:
                print(f"[CLEANUP] Killing stale {proc_name} processes: {pids}")
                _sp.run(["pkill", "-9", "-f", proc_name], timeout=10)
                time.sleep(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass


def _cleanup_shm() -> None:
    """Clean up stale shared memory from crashed sample-factory / PyTorch processes.

    Previously, crashed trials left behind cuda.shm.*, sem.mp-*, torch_* and
    torch_shm_ files in /dev/shm which prevented new workers from allocating
    shared memory, causing silent hangs (Fps=0.0, components not starting).
    """
    shm = Path("/dev/shm")
    # Expanded patterns — crashed trials leave many more files than originally tracked
    patterns = (
        "torch_*", "torch_shm_*", "cuda.shm.*", "sem.mp-*",
        "mp-*", ".torch_shm_", "faster_fifo_*", "sample_factory_*",
    )
    for pattern in patterns:
        for f in shm.glob(pattern):
            try:
                f.unlink()
            except (PermissionError, OSError):
                pass


def run_trial(
    params: Dict[str, Any],
    trial_id: int = 0,
    train_dir: str = "train_dir",
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    _cleanup_stale_processes()
    _cleanup_shm()

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
    cfg.mamba_expand = params.get('mamba_expand', 2)
    cfg.mamba_headdim = params.get('mamba_headdim', 64)
    cfg.mamba_ngroups = params.get('mamba_ngroups', 1)
    cfg.mamba_d_model = params.get('mamba_d_model', 512)
    cfg.weight_decay = params.get('weight_decay', 0.1)

    register_mamba2(cfg)

    from models.mamba2_core import get_mamba2_required_rnn_size
    cfg.rnn_size = cfg.mamba_d_model
    cfg.rnn_size = max(cfg.rnn_size, get_mamba2_required_rnn_size(cfg))

    print(f"Trial {trial_id}: d_model={cfg.mamba_d_model}, rnn_size={cfg.rnn_size}")

    # Cap per-trial VRAM to avoid OOM when running concurrently.
    # Triton backward pass needs workspace ~2x batch*rnn_size, so leave 30% headroom.
    import torch
    total_mem = torch.cuda.get_device_properties(0).total_memory  # bytes
    per_trial_limit = int(0.75 * total_mem)  # 75% of GPU per trial
    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    print(f"Trial {trial_id}: VRAM limit ~{per_trial_limit // (1024**2)}MB (75% of {torch.cuda.get_device_properties(0).total_memory // (1024**2)}MB)")

    # Track GPU memory before training
    _track_gpu_memory("BEFORE_TRAIN")
    start_memory = _get_gpu_memory()[0]  # allocated MB

    start_time = time.time()
    try:
        status = run_rl(cfg)
    except Exception as e:
        # Catch ALL exceptions, not just RuntimeError.
        # Shared memory errors, attribute errors, and VizDoom crashes
        # raise different exception types.
        print(f"Trial {trial_id} failed: {type(e).__name__}: {e}")
        allocated, reserved, peak = _get_gpu_memory()
        # Clean up aggressively on failure to prevent cascading failures
        _cleanup_stale_processes()
        _cleanup_shm()
        return {
            'trial_id': trial_id,
            'max_reward': 0.0,
            'mean_reward': 0.0,
            'std_reward': 0.0,
            'status': 'error',
            'error': f"{type(e).__name__}: {e}",
            'gpu_memory_allocated_mb': allocated,
            'gpu_memory_peak_mb': peak,
        }

    elapsed = time.time() - start_time
    if timeout and elapsed > timeout:
        print(f"Trial {trial_id} exceeded timeout: {elapsed:.0f}s > {timeout}s")

    # Track GPU memory after training
    _track_gpu_memory("AFTER_TRAIN")
    end_memory = _get_gpu_memory()[0]
    print(f"Trial {trial_id} GPU change: {end_memory - start_memory:+.0f}MB")

    # Clean up GPU memory and stale resources before extracting metrics
    _cleanup_gpu()
    _cleanup_stale_processes()
    _cleanup_shm()

    metrics = _extract_reward_metrics(train_dir, trial_id, timeout)
    metrics['trial_id'] = trial_id
    metrics['status'] = 'completed'
    metrics['elapsed_seconds'] = elapsed

    # Add GPU memory metrics
    allocated, reserved, peak = _get_gpu_memory()
    metrics['gpu_memory_allocated_mb'] = allocated
    metrics['gpu_memory_reserved_mb'] = reserved
    metrics['gpu_memory_peak_mb'] = peak
    metrics['gpu_memory_change_mb'] = end_memory - start_memory

    return metrics


if __name__ == '__main__':
    params = {
        'train_steps': 1000000,
        'learning_rate': 1.5e-4,
    }
    result = run_trial(params, trial_id=0, timeout=3600)
    print(f"Result: {result}")

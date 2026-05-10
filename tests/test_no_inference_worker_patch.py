"""
Tests to verify InferenceWorker is NOT monkey-patched by hpo/evaluate.py.

The previous _patch_inference_worker() caused two critical bugs:
1. Pickling failure: '_patched_prepare_policy_outputs_non_batched' attribute error
2. signal_slot crash: 'event_loop' attribute error on worker termination

These tests ensure the patches are removed and stay removed.

Run with: python -m pytest tests/test_no_inference_worker_patch.py -v
"""

import pickle
import subprocess
import sys
import pytest
from pathlib import Path


def test_no_inference_worker_patch_after_import():
    """Verify importing hpo.evaluate does NOT patch InferenceWorker."""
    # Import hpo.evaluate (which used to call _patch_inference_worker())
    from hpo import evaluate
    
    # Import InferenceWorker after the import
    from sample_factory.algo.sampling.inference_worker import InferenceWorker
    
    # The patched function was named _patched_prepare_policy_outputs_non_batched
    # If patching occurred, the function would have a different code object
    # than the original from the sample_factory source
    
    # Check that _resize_buffers_if_needed does NOT exist (it was added by patch)
    assert not hasattr(InferenceWorker, '_resize_buffers_if_needed'), \
        "InferenceWorker should NOT have _resize_buffers_if_needed (added by monkey-patch)"
    
    # Check that _initial_rnn_size is not set as a class attribute (added by patch)
    # Note: it may exist as instance attribute, which is fine
    assert not hasattr(InferenceWorker, '_initial_rnn_size'), \
        "InferenceWorker should NOT have _initial_rnn_size class attribute (added by monkey-patch)"


def _check_inference_worker_picklable(queue):
    """Module-level target for spawn test (local funcs can't be pickled)."""
    try:
        from sample_factory.algo.sampling.inference_worker import InferenceWorker
        data = pickle.dumps(InferenceWorker)
        restored = pickle.loads(data)
        queue.put(('ok', None))
    except Exception as e:
        queue.put(('error', str(e)))


def test_inference_worker_picklable():
    """Verify InferenceWorker class can be pickled without attribute errors.
    
    This was failing with:
    AttributeError: 'InferenceWorker' object has no attribute '_patched_prepare_policy_outputs_non_batched'
    """
    import multiprocessing as mp
    
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=_check_inference_worker_picklable, args=(q,))
    p.start()
    p.join(timeout=30)
    
    assert p.exitcode == 0, f"Process crashed with exitcode {p.exitcode}"
    
    status, msg = q.get()
    assert status == 'ok', f"Pickling failed: {msg}"


def test_no_patch_function_in_module():
    """Verify _patch_inference_worker does not exist in hpo.evaluate module."""
    from hpo import evaluate
    
    assert not hasattr(evaluate, '_patch_inference_worker'), \
        "hpo.evaluate should NOT have _patch_inference_worker function (removed)"


def test_inference_worker_original_methods_intact():
    """Verify InferenceWorker methods are the originals from sample_factory."""
    from hpo import evaluate  # Import after patching would have occurred
    from sample_factory.algo.sampling.inference_worker import InferenceWorker
    
    # The original _run method should be intact (not replaced with _patched_run)
    # We check by verifying the method's qualified name matches the original
    import inspect
    source_file = inspect.getfile(InferenceWorker._run)
    assert 'sample_factory' in source_file, \
        f"InferenceWorker._run should come from sample_factory, got {source_file}"


def test_hpo_evaluate_imports_cleanly():
    """Verify hpo.evaluate imports without side effects."""
    import importlib
    
    # Force reimport
    if 'hpo.evaluate' in sys.modules:
        del sys.modules['hpo.evaluate']
    
    # Should not raise any exceptions
    import hpo.evaluate
    
    # Verify no InferenceWorker patching occurred
    from sample_factory.algo.sampling.inference_worker import InferenceWorker
    assert not hasattr(InferenceWorker, '_resize_buffers_if_needed')


def test_build_argv_correct():
    """Verify _build_argv produces correct argument lists."""
    from hpo.evaluate import _build_argv
    
    params = {
        'train_steps': 1000000,
        'learning_rate': 1.5e-4,
        'rnn_num_layers': 2,
    }
    argv = _build_argv(params, trial_id=42, train_dir='test_dir')
    
    assert '--env' in argv
    assert 'doom_benchmark' in argv
    assert '--algo' in argv
    assert 'APPO' in argv
    assert '--train_for_env_steps' in argv
    assert '1000000' in argv
    assert '--learning_rate' in argv


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Tests for Mamba-2 core integration with sample-factory.

Run with: python -m pytest tests/test_mamba2_core.py -v
"""

import pytest
import torch
from torch.nn.utils.rnn import PackedSequence
from unittest.mock import MagicMock

from models.mamba2_core import Mamba2Core, MAMBA_AVAILABLE


@pytest.fixture
def mamba_cfg():
    """Create a mock config for Mamba-2."""
    cfg = MagicMock()
    cfg.rnn_size = 128
    cfg.rnn_num_layers = 1
    cfg.mamba_d_state = 16
    cfg.mamba_d_conv = 4
    cfg.mamba_expand = 2
    cfg.mamba_headdim = 32
    cfg.mamba_ngroups = 1
    return cfg


@pytest.fixture
def mamba_core(mamba_cfg):
    """Create a Mamba2Core instance."""
    return Mamba2Core(mamba_cfg, input_size=128).to('cuda')


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
def test_core_creation(mamba_core):
    """Test that Mamba2Core can be instantiated."""
    assert mamba_core.core_output_size == 128


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
def test_forward_with_packed_sequence(mamba_core):
    """Test forward pass with PackedSequence (training mode)."""
    batch_size = 4
    seq_len = 10

    x = torch.randn(seq_len, batch_size, 128, device='cuda')
    batch_sizes = torch.tensor([batch_size] * seq_len)
    pseq = PackedSequence(x, batch_sizes, None, None)
    rnn_states = torch.randn(batch_size, 256, device='cuda')

    out, new_states = mamba_core.forward(pseq, rnn_states)

    assert isinstance(out, PackedSequence)
    assert out.data.shape == (seq_len, batch_size, 128)


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
def test_forward_without_sequence(mamba_core):
    """Test forward pass without sequence dimension."""
    batch_size = 4

    head_output = torch.randn(batch_size, 128, device='cuda')
    rnn_states = torch.randn(batch_size, 256, device='cuda')

    x, new_states = mamba_core.forward(head_output, rnn_states)

    assert x.shape == (batch_size, 128)


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
def test_states_unchanged(mamba_core):
    """Verify rnn_states are passed through unchanged."""
    batch_size = 4

    head_output = torch.randn(batch_size, 128, device='cuda')
    rnn_states = torch.randn(batch_size, 256, device='cuda')

    _, new_states = mamba_core.forward(head_output, rnn_states)

    assert torch.equal(rnn_states, new_states)


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
def test_multi_layer():
    """Test with multiple Mamba-2 layers."""
    cfg = MagicMock()
    cfg.rnn_size = 128
    cfg.rnn_num_layers = 2
    cfg.mamba_d_state = 16
    cfg.mamba_d_conv = 4
    cfg.mamba_expand = 2
    cfg.mamba_headdim = 32
    cfg.mamba_ngroups = 1

    core = Mamba2Core(cfg, input_size=128).to('cuda')

    batch_size = 4
    seq_len = 10
    x = torch.randn(seq_len, batch_size, 128, device='cuda')
    batch_sizes = torch.tensor([batch_size] * seq_len)
    pseq = PackedSequence(x, batch_sizes, None, None)
    rnn_states = torch.randn(batch_size, 256, device='cuda')

    out, new_states = core.forward(pseq, rnn_states)
    assert out.data.shape == (seq_len, batch_size, 128)


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
def test_parameter_count():
    """Verify parameter count is reasonable."""
    cfg = MagicMock()
    cfg.rnn_size = 256  # Match training config
    cfg.rnn_num_layers = 1
    cfg.mamba_d_state = 16
    cfg.mamba_d_conv = 4
    cfg.mamba_expand = 2
    cfg.mamba_headdim = 32
    cfg.mamba_ngroups = 1

    core = Mamba2Core(cfg, input_size=256)
    total_params = sum(p.numel() for p in core.parameters())

    # Should be in the range of a few hundred thousand
    assert 100_000 < total_params < 5_000_000, f"Unexpected param count: {total_params}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

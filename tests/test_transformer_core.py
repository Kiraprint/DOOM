"""Tests for Transformer core."""

import pytest
import torch
from torch.nn.utils.rnn import PackedSequence
from unittest.mock import MagicMock

from models.transformer_core import TransformerCore


@pytest.fixture
def transformer_cfg():
    cfg = MagicMock()
    cfg.rnn_size = 2048  # must >= d_model * window = 64*16 = 1024
    cfg.rnn_num_layers = 1
    cfg.transformer_d_model = 64
    cfg.transformer_nhead = 4
    cfg.transformer_num_layers = 1
    cfg.transformer_window_size = 16
    cfg.transformer_dim_feedforward = 256
    cfg.transformer_dropout = 0.0
    return cfg


@pytest.fixture
def transformer_core(transformer_cfg):
    return TransformerCore(transformer_cfg, input_size=64)


def _make_pseq(batch_size, seq_len, dim):
    """Create PackedSequence with 2D data matching sample-factory format (total, dim)."""
    total = seq_len * batch_size
    x = torch.randn(total, dim)
    batch_sizes = torch.tensor([batch_size] * seq_len)
    return PackedSequence(x, batch_sizes, None, None)


def test_core_creation(transformer_core):
    assert transformer_core.core_output_size == 64


def test_forward_with_packed_sequence(transformer_core):
    batch_size = 2
    seq_len = 8
    head_output = _make_pseq(batch_size, seq_len, 64)
    rnn_states = torch.randn(batch_size, 1024)
    out, new_states = transformer_core.forward(head_output, rnn_states)
    assert isinstance(out, PackedSequence)
    assert out.data.shape == (batch_size * seq_len, 64)


def test_forward_without_sequence(transformer_core):
    batch_size = 2
    head_output = torch.randn(batch_size, 64)
    rnn_states = torch.randn(batch_size, 1024)
    x, new_states = transformer_core.forward(head_output, rnn_states)
    assert x.shape == (batch_size, 64)


def test_states_unchanged(transformer_core):
    batch_size = 2
    head_output = torch.randn(batch_size, 64)
    rnn_states = torch.randn(batch_size, 1024)
    _, new_states = transformer_core.forward(head_output, rnn_states)
    assert not torch.equal(rnn_states, new_states)


def test_multi_layer():
    cfg = MagicMock()
    cfg.rnn_size = 2048  # must >= d_model * window = 64*16 = 1024
    cfg.rnn_num_layers = 2
    cfg.transformer_d_model = 64
    cfg.transformer_nhead = 4
    cfg.transformer_num_layers = 2
    cfg.transformer_window_size = 16
    cfg.transformer_dim_feedforward = 256
    cfg.transformer_dropout = 0.0
    core = TransformerCore(cfg, input_size=64)
    batch_size = 2
    seq_len = 8
    head_output = _make_pseq(batch_size, seq_len, 64)
    rnn_states = torch.randn(batch_size, 1024)
    out, _ = core.forward(head_output, rnn_states)
    assert out.data.shape == (batch_size * seq_len, 64)


def test_parameter_count():
    cfg = MagicMock()
    cfg.rnn_size = 2048  # must >= d_model * window = 64*16 = 1024
    cfg.rnn_num_layers = 1
    cfg.transformer_d_model = 64
    cfg.transformer_nhead = 4
    cfg.transformer_num_layers = 1
    cfg.transformer_window_size = 16
    cfg.transformer_dim_feedforward = 256
    cfg.transformer_dropout = 0.0
    core = TransformerCore(cfg, input_size=64)
    total = sum(p.numel() for p in core.parameters())
    assert 10_000 < total < 5_000_000, f"Unexpected param count: {total}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

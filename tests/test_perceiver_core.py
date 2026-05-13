"""Tests for Perceiver IO core."""

import pytest
import torch
from torch.nn.utils.rnn import PackedSequence
from unittest.mock import MagicMock

from models.perceiver_core import PerceiverCore


@pytest.fixture
def perceiver_cfg():
    cfg = MagicMock()
    cfg.rnn_size = 2048
    cfg.rnn_num_layers = 1
    cfg.perceiver_d_model = 64
    cfg.perceiver_num_latents = 8
    cfg.perceiver_d_latents = 64
    cfg.perceiver_num_blocks = 1
    cfg.perceiver_num_heads = 4
    cfg.perceiver_dropout = 0.0
    cfg.perceiver_num_layers = 1
    return cfg


@pytest.fixture
def perceiver_core(perceiver_cfg):
    return PerceiverCore(perceiver_cfg, input_size=64)


def test_core_creation(perceiver_core):
    assert perceiver_core.core_output_size == 64


def test_forward_with_packed_sequence(perceiver_core):
    batch_size = 2
    seq_len = 8
    x = torch.randn(seq_len, batch_size, 64)
    batch_sizes = torch.tensor([batch_size] * seq_len)
    pseq = PackedSequence(x, batch_sizes, None, None)
    rnn_states = torch.randn(batch_size, 512)
    out, new_states = perceiver_core.forward(pseq, rnn_states)
    assert isinstance(out, PackedSequence)
    assert out.data.shape == (seq_len, batch_size, 64)


def test_forward_without_sequence(perceiver_core):
    batch_size = 2
    head_output = torch.randn(batch_size, 64)
    rnn_states = torch.randn(batch_size, 512)
    x, new_states = perceiver_core.forward(head_output, rnn_states)
    assert x.shape == (batch_size, 64)


def test_states_unchanged(perceiver_core):
    batch_size = 2
    head_output = torch.randn(batch_size, 64)
    rnn_states = torch.randn(batch_size, 512)
    _, new_states = perceiver_core.forward(head_output, rnn_states)
    assert not torch.equal(rnn_states, new_states)


def test_multi_block():
    cfg = MagicMock()
    cfg.rnn_size = 4096
    cfg.rnn_num_layers = 1
    cfg.perceiver_d_model = 64
    cfg.perceiver_num_latents = 8
    cfg.perceiver_d_latents = 64
    cfg.perceiver_num_blocks = 2
    cfg.perceiver_num_heads = 4
    cfg.perceiver_dropout = 0.0
    cfg.perceiver_num_layers = 1
    core = PerceiverCore(cfg, input_size=64)
    batch_size = 2
    seq_len = 8
    x = torch.randn(seq_len, batch_size, 64)
    batch_sizes = torch.tensor([batch_size] * seq_len)
    pseq = PackedSequence(x, batch_sizes, None, None)
    rnn_states = torch.randn(batch_size, 512)
    out, _ = core.forward(pseq, rnn_states)
    assert out.data.shape == (seq_len, batch_size, 64)


def test_parameter_count():
    cfg = MagicMock()
    cfg.rnn_size = 2048
    cfg.rnn_num_layers = 1
    cfg.perceiver_d_model = 64
    cfg.perceiver_num_latents = 8
    cfg.perceiver_d_latents = 64
    cfg.perceiver_num_blocks = 1
    cfg.perceiver_num_heads = 4
    cfg.perceiver_dropout = 0.0
    cfg.perceiver_num_layers = 1
    core = PerceiverCore(cfg, input_size=64)
    total = sum(p.numel() for p in core.parameters())
    assert 10_000 < total < 10_000_000, f"Unexpected param count: {total}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

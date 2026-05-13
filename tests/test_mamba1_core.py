"""Tests for Mamba-1 core."""

import pytest
import torch
from torch.nn.utils.rnn import PackedSequence
from unittest.mock import MagicMock

from models.mamba1_core import Mamba1Core, MAMBA1_AVAILABLE


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def mamba1_cfg():
    cfg = MagicMock()
    # rnn_size must be >= conv_state_size + ssm_state_size
    # d_inner=d_model*expand=512*2=1024, conv=1024*4=4096, ssm=1024*16=16384 → total=20480
    cfg.rnn_size = 24576
    cfg.rnn_num_layers = 1
    cfg.mamba1_d_model = 512
    cfg.mamba1_d_state = 16
    cfg.mamba1_d_conv = 4
    cfg.mamba1_expand = 2
    cfg.device = DEVICE
    return cfg


@pytest.fixture
def mamba1_core(mamba1_cfg):
    core = Mamba1Core(mamba1_cfg, input_size=128)
    return core.to(DEVICE)


ENC_DIM = 128  # input_size matches encoder feature dim


@pytest.mark.skipif(not MAMBA1_AVAILABLE, reason="mamba-ssm not installed")
def test_core_creation(mamba1_core):
    assert mamba1_core.core_output_size == 512  # d_model after input_proj


@pytest.mark.skipif(not MAMBA1_AVAILABLE, reason="mamba-ssm not installed")
def test_forward_with_packed_sequence(mamba1_core):
    batch_size = 4
    seq_len = 10
    # sample-factory PackedSequence uses 2D data: (sum(batch_sizes), dim)
    total = seq_len * batch_size
    x = torch.randn(total, ENC_DIM, device=DEVICE)
    batch_sizes = torch.tensor([batch_size] * seq_len)
    pseq = PackedSequence(x, batch_sizes, None, None)
    rnn_states = torch.randn(batch_size, 24576, device=DEVICE)
    out, new_states = mamba1_core.forward(pseq, rnn_states)
    assert isinstance(out, PackedSequence)
    assert out.data.shape == (total, 512)


@pytest.mark.skipif(not MAMBA1_AVAILABLE, reason="mamba-ssm not installed")
def test_forward_without_sequence(mamba1_core):
    batch_size = 4
    head_output = torch.randn(batch_size, ENC_DIM, device=DEVICE)
    rnn_states = torch.randn(batch_size, 24576, device=DEVICE)
    x, new_states = mamba1_core.forward(head_output, rnn_states)
    assert x.shape == (batch_size, 512)


@pytest.mark.skipif(not MAMBA1_AVAILABLE, reason="mamba-ssm not installed")
def test_states_unchanged(mamba1_core):
    batch_size = 4
    head_output = torch.randn(batch_size, ENC_DIM, device=DEVICE)
    rnn_states = torch.randn(batch_size, 24576, device=DEVICE)
    _, new_states = mamba1_core.forward(head_output, rnn_states)
    assert not torch.equal(rnn_states, new_states)


@pytest.mark.skipif(not MAMBA1_AVAILABLE, reason="mamba-ssm not installed")
def test_multi_layer():
    cfg = MagicMock()
    # d_inner=128*2=256, conv=256*4=1024, ssm=256*16=4096 → total=5120
    cfg.rnn_size = 8192
    cfg.rnn_num_layers = 2
    cfg.mamba1_d_model = 128
    cfg.mamba1_d_state = 16
    cfg.mamba1_d_conv = 4
    cfg.mamba1_expand = 2
    cfg.device = DEVICE
    core = Mamba1Core(cfg, input_size=128).to(DEVICE)
    batch_size = 4
    seq_len = 10
    total = seq_len * batch_size
    x = torch.randn(total, 128, device=DEVICE)
    batch_sizes = torch.tensor([batch_size] * seq_len)
    pseq = PackedSequence(x, batch_sizes, None, None)
    rnn_states = torch.randn(batch_size, 8192, device=DEVICE)
    out, _ = core.forward(pseq, rnn_states)
    assert out.data.shape == (total, 128)


@pytest.mark.skipif(not MAMBA1_AVAILABLE, reason="mamba-ssm not installed")
def test_parameter_count():
    cfg = MagicMock()
    # d_inner=256*2=512, conv=512*4=2048, ssm=512*16=8192 → total=10240
    cfg.rnn_size = 16384
    cfg.rnn_num_layers = 1
    cfg.mamba1_d_model = 256
    cfg.mamba1_d_state = 16
    cfg.mamba1_d_conv = 4
    cfg.mamba1_expand = 2
    core = Mamba1Core(cfg, input_size=256)
    total = sum(p.numel() for p in core.parameters())
    assert 50_000 < total < 50_000_000, f"Unexpected param count: {total}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

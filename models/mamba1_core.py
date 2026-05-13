"""Mamba-1 core for sample-factory. Drop-in GRU replacement.

Uses original Mamba (selective scan) from mamba-ssm package.
Follows same pattern as mamba2_core.py but for Mamba v1 API.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from sample_factory.model.core import ModelCore
from sample_factory.utils.typing import Config

from models.mamba2_core import _unpack_packed_sequence_2d, _pack_to_2d_sequence

try:
    from mamba_ssm.modules.mamba_simple import Mamba as Mamba1Block
    from mamba_ssm.utils.generation import InferenceParams
    MAMBA1_AVAILABLE = True
except ImportError:
    MAMBA1_AVAILABLE = False
    Mamba1Block = None
    InferenceParams = None


class Mamba1StateEncoder:
    """Encodes/decodes Mamba-1 internal state to/from flat rnn_states.

    Mamba-1's allocate_inference_cache returns:
        conv_state: (batch, d_inner, d_conv)
        ssm_state:  (batch, d_inner, d_state)
    """

    def __init__(self, d_inner: int, d_conv: int, d_state: int):
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.d_state = d_state
        self.conv_state_size = d_inner * d_conv
        self.ssm_state_size = d_inner * d_state
        self.total_size = self.conv_state_size + self.ssm_state_size

    def encode(self, conv_state, ssm_state):
        batch = conv_state.shape[0]
        conv_flat = conv_state.reshape(batch, -1)
        ssm_flat = ssm_state.reshape(batch, -1)
        return torch.cat([conv_flat, ssm_flat], dim=-1)

    def decode(self, flat_state, device=None, dtype=None):
        if device is None:
            device = flat_state.device
        if dtype is None:
            dtype = flat_state.dtype
        batch = flat_state.shape[0]
        conv_flat = flat_state[:, :self.conv_state_size]
        ssm_flat = flat_state[:, self.conv_state_size:]
        conv_state = conv_flat.reshape(batch, self.d_inner, self.d_conv)
        ssm_state = ssm_flat.reshape(batch, self.d_inner, self.d_state)
        return conv_state, ssm_state

    def get_required_rnn_size(self):
        return self.total_size


class Mamba1Core(ModelCore):
    """Mamba-1 core replacing GRU in sample-factory.

    Follows exact pattern of Mamba2Core but uses Mamba v1 API.
    """

    def __init__(self, cfg: Config, input_size: int):
        super().__init__(cfg)

        if not MAMBA1_AVAILABLE:
            raise ImportError("mamba-ssm not installed. pip install mamba-ssm --no-build-isolation")

        self.cfg = cfg
        self.d_model = getattr(cfg, 'mamba1_d_model', cfg.rnn_size)
        self.num_layers = cfg.rnn_num_layers
        self.d_state = getattr(cfg, 'mamba1_d_state', 16)
        self.d_conv = getattr(cfg, 'mamba1_d_conv', 4)
        self.expand = getattr(cfg, 'mamba1_expand', 2)
        self.d_inner = int(self.expand * self.d_model)

        self.state_encoder = Mamba1StateEncoder(self.d_inner, self.d_conv, self.d_state)

        required_per_layer = self.state_encoder.total_size
        if cfg.rnn_size < required_per_layer:
            raise ValueError(f"rnn_size ({cfg.rnn_size}) too small for Mamba-1. Need {required_per_layer} per layer")

        self.mamba_wrapped = nn.ModuleList()
        self.mamba_norms = nn.ModuleList()

        for i in range(self.num_layers):
            norm = nn.LayerNorm(self.d_model)
            block = Mamba1Block(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                layer_idx=i,
            )
            self.mamba_wrapped.append(block)
            self.mamba_norms.append(norm)

        if input_size != self.d_model:
            self.input_proj = nn.Linear(input_size, self.d_model)
        else:
            self.input_proj = nn.Identity()

        self.core_output_size = self.d_model

    def _create_inference_params(self, batch_size, max_seqlen=256):
        params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)
        params.key_value_memory_dict = {}
        for block in self.mamba_wrapped:
            conv_state, ssm_state = block.allocate_inference_cache(batch_size, max_seqlen)
            params.key_value_memory_dict[block.layer_idx] = (conv_state, ssm_state)
        return params

    def _load_states_from_rnn(self, inference_params, rnn_states):
        state_norm = rnn_states.norm().item()
        is_episode_start = state_norm < 1e-6

        for block in self.mamba_wrapped:
            idx = block.layer_idx
            conv_state, ssm_state = inference_params.key_value_memory_dict[idx]

            if is_episode_start:
                conv_state.zero_()
                ssm_state.zero_()
            else:
                layer_size = self.state_encoder.total_size
                offset = idx * layer_size
                layer_flat = rnn_states[:, offset:offset + layer_size]

                if layer_flat.shape[1] >= self.state_encoder.total_size:
                    conv_decoded, ssm_decoded = self.state_encoder.decode(layer_flat)
                    conv_state.copy_(conv_decoded.to(conv_state.dtype))
                    ssm_state.copy_(ssm_decoded.to(ssm_state.dtype))

    def _get_inference_states(self, inference_params):
        device = list(inference_params.key_value_memory_dict.values())[0][0].device
        all_states = []
        for block in self.mamba_wrapped:
            idx = block.layer_idx
            conv_state, ssm_state = inference_params.key_value_memory_dict[idx]
            encoded = self.state_encoder.encode(conv_state, ssm_state)
            all_states.append(encoded)
        return torch.cat(all_states, dim=-1).to(device)

    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output)

        if is_seq:
            x_data = _unpack_packed_sequence_2d(head_output)
        else:
            x_data = head_output.unsqueeze(0)

        x_data = x_data.permute(1, 0, 2)
        x_data = self.input_proj(x_data)

        if is_seq:
            x_data = self.mamba_norms[0](x_data)
            x_data = self.mamba_wrapped[0](x_data)
            for norm, block in zip(self.mamba_norms[1:], self.mamba_wrapped[1:]):
                x_data = block(norm(x_data))
        else:
            batch_size = x_data.shape[0]
            max_seqlen = getattr(self.cfg, 'train_horizon', 256)
            inference_params = self._create_inference_params(batch_size, max_seqlen)
            self._load_states_from_rnn(inference_params, rnn_states)
            inference_params.seqlen_offset = 1
            for norm, block in zip(self.mamba_norms, self.mamba_wrapped):
                x_data = block(norm(x_data), inference_params=inference_params)
            new_rnn_states = self._get_inference_states(inference_params)

        x_data = x_data.permute(1, 0, 2)

        if is_seq:
            x = _pack_to_2d_sequence(x_data, head_output)
            new_rnn_states = rnn_states
        else:
            x = x_data.squeeze(0)

        return x, new_rnn_states


def make_mamba1_core(cfg, input_size):
    return Mamba1Core(cfg, input_size)


_MAMBA1_REGISTERED = False


class Mamba1Factory:
    """Picklable factory that always returns Mamba1Core.

    Only registered when rnn_type='mamba1', so no fallback needed.
    """
    __slots__ = ('_num_layers',)
    def __init__(self, num_layers):
        self._num_layers = num_layers
    def __call__(self, cfg, core_input_size):
        cfg.rnn_num_layers = self._num_layers
        return Mamba1Core(cfg, core_input_size)


def register_mamba1(cfg=None):
    global _MAMBA1_REGISTERED
    if _MAMBA1_REGISTERED:
        return
    num_layers = cfg.rnn_num_layers if cfg is not None else 1
    factory = Mamba1Factory(num_layers)
    from sample_factory.algo.utils.context import global_model_factory
    global_model_factory().register_model_core_factory(factory)
    _MAMBA1_REGISTERED = True
    print(f"[Mamba1] Registered as rnn_type='mamba1' (num_layers={num_layers})")


def get_mamba1_required_rnn_size(cfg):
    d_model = getattr(cfg, 'mamba1_d_model', cfg.rnn_size)
    d_state = getattr(cfg, 'mamba1_d_state', 16)
    d_conv = getattr(cfg, 'mamba1_d_conv', 4)
    expand = getattr(cfg, 'mamba1_expand', 2)
    d_inner = int(expand * d_model)
    encoder = Mamba1StateEncoder(d_inner, d_conv, d_state)
    return encoder.total_size

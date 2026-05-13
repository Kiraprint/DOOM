"""Transformer encoder core for sample-factory. Drop-in GRU replacement.

GPT-style decoder (encoder-only, no cross-attention) with causal attention.
During training: processes full sequence via PackedSequence.
During inference: maintains sliding window of past tokens as state.
"""

import math
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from sample_factory.model.core import ModelCore
from sample_factory.utils.typing import Config

from models.mamba2_core import _unpack_packed_sequence_2d, _pack_to_2d_sequence


def _causal_mask(sz, device):
    """Causal attention mask. Upper triangle = -inf."""
    mask = torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
    return mask


class TransformerStateEncoder:
    """Encodes/decodes sliding window buffer to/from flat rnn_states."""

    def __init__(self, d_model: int, window_size: int, num_layers: int):
        self.d_model = d_model
        self.window_size = window_size
        self.num_layers = num_layers
        # Per layer: buffer of past tokens (window_size * d_model)
        # For num_layers, we store one buffer per layer
        self.per_layer_size = window_size * d_model
        self.total_size = self.per_layer_size * num_layers

    def encode(self, buffer):
        """buffer: (batch, window_size, d_model)"""
        batch = buffer.shape[0]
        return buffer.reshape(batch, -1)

    def decode(self, flat_state, batch, device=None, dtype=None):
        if device is None:
            device = flat_state.device
        if dtype is None:
            dtype = flat_state.dtype
        buffer = flat_state[:, :self.per_layer_size].reshape(batch, self.window_size, self.d_model)
        return buffer.to(device=device, dtype=dtype)

    def get_required_rnn_size(self):
        return self.per_layer_size


class TransformerCore(ModelCore):
    """GPT-style encoder core for sample-factory.

    Uses nn.TransformerEncoderLayer with causal masking (no cross-attention).
    Inference state = sliding window of past tokens.
    """

    def __init__(self, cfg: Config, input_size: int):
        super().__init__(cfg)

        self.cfg = cfg
        self.d_model = getattr(cfg, 'transformer_d_model', cfg.rnn_size)
        self.nhead = getattr(cfg, 'transformer_nhead', 8)
        self.num_layers = getattr(cfg, 'transformer_num_layers', cfg.rnn_num_layers)
        self.window_size = getattr(cfg, 'transformer_window_size', 64)
        self.dim_feedforward = getattr(cfg, 'transformer_dim_feedforward', self.d_model * 4)
        self.dropout = getattr(cfg, 'transformer_dropout', 0.1)

        self.state_encoder = TransformerStateEncoder(self.d_model, self.window_size, self.num_layers)

        required_per_layer = self.state_encoder.per_layer_size
        if cfg.rnn_size < required_per_layer:
            raise ValueError(
                f"rnn_size ({cfg.rnn_size}) too small for Transformer. "
                f"Need {required_per_layer} per layer (d_model={self.d_model} * window={self.window_size})"
            )

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.layers.append(layer)
            self.norms.append(nn.LayerNorm(self.d_model))

        if input_size != self.d_model:
            self.input_proj = nn.Linear(input_size, self.d_model)
        else:
            self.input_proj = nn.Identity()

        # Learned positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.window_size, self.d_model) * 0.02)

        self.core_output_size = self.d_model

    def _forward_sequence(self, x, mask=None):
        """Run transformer layers on (B, T, D) with optional causal mask."""
        for norm, layer in zip(self.norms, self.layers):
            x = layer(x, src_mask=mask)
        return x

    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output)

        if is_seq:
            x_data = _unpack_packed_sequence_2d(head_output)  # (T, B, D)
        else:
            x_data = head_output.unsqueeze(0)  # (1, B, D)

        x_data = x_data.permute(1, 0, 2)  # -> (B, T, D)
        x_data = self.input_proj(x_data)

        B, T, D = x_data.shape

        if is_seq:
            # Training: add position encoding, causal mask
            x_data = x_data + self.pos_embedding[:, :T, :]
            mask = _causal_mask(T, x_data.device)
            x_data = self._forward_sequence(x_data, mask)
            # Take last token output as core output
            out = x_data[:, -1:, :]
        else:
            # Inference: maintain sliding window as state
            batch_size = B

            # Decode buffer from rnn_states
            state_norm = rnn_states.norm().item()
            is_episode_start = state_norm < 1e-6

            if is_episode_start:
                buffer = torch.zeros(batch_size, self.window_size, D, device=x_data.device, dtype=x_data.dtype)
            else:
                buffer = self.state_encoder.decode(rnn_states, batch_size, device=x_data.device, dtype=x_data.dtype)

            # Shift buffer left, append new input at end
            buffer = torch.roll(buffer, shifts=-1, dims=1)
            buffer[:, -1:, :] = x_data + self.pos_embedding[:, -1:, :]

            # Run transformer on full buffer
            mask = _causal_mask(self.window_size, x_data.device)
            buffer_out = self._forward_sequence(buffer, mask)

            # Output is last token
            out = buffer_out[:, -1:, :]

            # Encode buffer back to rnn_states
            new_rnn_states = self.state_encoder.encode(buffer_out.detach())

        x_data = out.permute(1, 0, 2)  # -> (1, B, D) or (1, B, D)

        if is_seq:
            # For training: output shape (T, B, D) matching input
            # But we only need last token per sequence
            x = _pack_to_2d_sequence(x_data.expand(T, -1, -1).contiguous(), head_output)
            new_rnn_states = rnn_states
        else:
            x = x_data.squeeze(0)  # (B, D)

        return x, new_rnn_states


def make_transformer_core(cfg, input_size):
    return TransformerCore(cfg, input_size)


_TRANSFORMER_REGISTERED = False


class TransformerFactory:
    """Picklable factory that always returns TransformerCore.

    Only registered when rnn_type='transformer', so no fallback needed.
    """
    __slots__ = ('_num_layers',)
    def __init__(self, num_layers):
        self._num_layers = num_layers
    def __call__(self, cfg, core_input_size):
        cfg.rnn_num_layers = self._num_layers
        return TransformerCore(cfg, core_input_size)


def register_transformer(cfg=None):
    global _TRANSFORMER_REGISTERED
    if _TRANSFORMER_REGISTERED:
        return
    num_layers = cfg.rnn_num_layers if cfg is not None else 1
    factory = TransformerFactory(num_layers)
    from sample_factory.algo.utils.context import global_model_factory
    global_model_factory().register_model_core_factory(factory)
    _TRANSFORMER_REGISTERED = True
    print(f"[Transformer] Registered as rnn_type='transformer' (num_layers={num_layers})")


def get_transformer_required_rnn_size(cfg):
    d_model = getattr(cfg, 'transformer_d_model', cfg.rnn_size)
    window_size = getattr(cfg, 'transformer_window_size', 64)
    num_layers = getattr(cfg, 'transformer_num_layers', cfg.rnn_num_layers)
    encoder = TransformerStateEncoder(d_model, window_size, num_layers)
    return encoder.per_layer_size

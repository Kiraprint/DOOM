"""Perceiver IO core for sample-factory. Drop-in GRU replacement.

Uses cross-attention to compress inputs into a smaller latent space.
The latent array serves as the RNN state during inference.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
import math

from sample_factory.model.core import ModelCore
from sample_factory.utils.typing import Config

from models.mamba2_core import _unpack_packed_sequence_2d, _pack_to_2d_sequence


class PreNormResidual(nn.Module):
    """Pre-norm residual block wrapping any module."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(self.norm(x), **kwargs)


class CrossAttentionBlock(nn.Module):
    """Cross-attention: latent (Q) attends to input (K,V).

    Pre-norm applied to latent before Q projection to prevent unbounded growth.
    Input x is assumed already normalized (from PerceiverCore.input_norm).
    """
    def __init__(self, d_latent, d_input, num_heads, dropout=0.1):
        super().__init__()
        self.d_latent = d_latent
        self.d_input = d_input
        self.num_heads = num_heads

        self.norm_latent = nn.LayerNorm(d_latent)

        # Project latent (Q) and input (K,V) to same dimension for attention
        self.q_proj = nn.Linear(d_latent, d_latent)
        self.kv_proj = nn.Linear(d_input, d_latent)

        self.attn = nn.MultiheadAttention(d_latent, num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_latent)
        self.ffn = nn.Sequential(
            nn.Linear(d_latent, d_latent * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent * 4, d_latent),
        )

    def forward(self, latent, x):
        q = self.q_proj(self.norm_latent(latent))
        kv = self.kv_proj(x)
        attn_out, _ = self.attn(q, kv, kv)
        latent = latent + attn_out
        latent = latent + self.ffn(self.norm_ffn(latent))
        return latent


class SelfAttentionBlock(nn.Module):
    """Self-attention on latent array with pre-norm."""
    def __init__(self, d_latent, num_heads, dropout=0.1):
        super().__init__()
        self.d_latent = d_latent
        self.norm_attn = nn.LayerNorm(d_latent)
        self.attn = nn.MultiheadAttention(d_latent, num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_latent)
        self.ffn = nn.Sequential(
            nn.Linear(d_latent, d_latent * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent * 4, d_latent),
        )

    def forward(self, latent):
        normed = self.norm_attn(latent)
        attn_out, _ = self.attn(normed, normed, normed)
        latent = latent + attn_out
        latent = latent + self.ffn(self.norm_ffn(latent))
        return latent


class PerceiverStateEncoder:
    """Encodes/decodes latent array to/from flat rnn_states."""

    def __init__(self, num_latents: int, d_latents: int):
        self.num_latents = num_latents
        self.d_latents = d_latents
        self.total_size = num_latents * d_latents

    def encode(self, latent):
        batch = latent.shape[0]
        return latent.reshape(batch, -1)

    def decode(self, flat_state, batch, device=None, dtype=None):
        if device is None:
            device = flat_state.device
        if dtype is None:
            dtype = flat_state.dtype
        latent = flat_state[:, :self.total_size].reshape(batch, self.num_latents, self.d_latents)
        return latent.to(device=device, dtype=dtype)

    def get_required_rnn_size(self):
        return self.total_size


class PerceiverCore(ModelCore):
    """Perceiver IO core for sample-factory.

    Uses cross-attention to compress observations into a latent space.
    The latent array IS the RNN state during inference.
    """

    def __init__(self, cfg: Config, input_size: int):
        super().__init__(cfg)

        self.cfg = cfg
        self.d_model = getattr(cfg, 'perceiver_d_model', 512)
        self.num_latents = getattr(cfg, 'perceiver_num_latents', 32)
        self.d_latents = getattr(cfg, 'perceiver_d_latents', 512)
        self.num_blocks = getattr(cfg, 'perceiver_num_blocks', 2)
        self.num_heads = getattr(cfg, 'perceiver_num_heads', 8)
        self.dropout = getattr(cfg, 'perceiver_dropout', 0.1)
        self.num_layers = getattr(cfg, 'perceiver_num_layers', cfg.rnn_num_layers)

        self.state_encoder = PerceiverStateEncoder(self.num_latents, self.d_latents)

        required_rnn = self.state_encoder.total_size
        if cfg.rnn_size < required_rnn:
            raise ValueError(
                f"rnn_size ({cfg.rnn_size}) too small for Perceiver. "
                f"Need {required_rnn} (num_latents={self.num_latents} * d_latents={self.d_latents})"
            )

        # Input projection
        self.input_proj = nn.Linear(input_size, self.d_model) if input_size != self.d_model else nn.Identity()
        self.input_norm = nn.LayerNorm(self.d_model)

        # Learnable latent array (shared across batch)
        self.latent = nn.Parameter(torch.randn(1, self.num_latents, self.d_latents) * 0.02)

        # Learned position encoding for input
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, self.d_model) * 0.02)

        # Cross-attention blocks: input -> latent
        self.cross_blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.cross_blocks.append(CrossAttentionBlock(self.d_latents, self.d_model, self.num_heads, self.dropout))

        # Self-attention blocks: latent -> latent
        self.self_blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.self_blocks.append(SelfAttentionBlock(self.d_latents, self.num_heads, self.dropout))

        # Output projection: latent -> output
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.d_latents),
            nn.Linear(self.d_latents, self.d_model),
        )

        self.core_output_size = self.d_model

    def _process(self, x, latent):
        """Process input x with latent array.

        Args:
            x: (B, T, D) input sequence
            latent: (B, num_latents, d_latents) initial latent

        Returns:
            latent: (B, num_latents, d_latents) updated latent
            output: (B, D) pooled output
        """
        for cross, self_attn in zip(self.cross_blocks, self.self_blocks):
            latent = cross(latent, x)
            latent = self_attn(latent)

        # Pool latent (mean over latents)
        pooled = latent.mean(dim=1)
        output = self.output_proj(pooled)
        return latent, output

    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output)

        if is_seq:
            x_data = _unpack_packed_sequence_2d(head_output)  # (T, B, D)
        else:
            x_data = head_output.unsqueeze(0)  # (1, B, D)

        x_data = x_data.permute(1, 0, 2)  # -> (B, T, D)
        x_data = self.input_proj(x_data)
        x_data = self.input_norm(x_data)

        B, T, D = x_data.shape

        if is_seq:
            # Training: process full sequence
            x_data = x_data + self.pos_encoding[:, :T, :]
            latent = self.latent.expand(B, -1, -1)
            latent, output = self._process(x_data, latent)
            out = output.unsqueeze(1)  # (B, 1, D)
        else:
            # Inference: use latent from rnn_states
            state_norm = rnn_states.norm().item()
            is_episode_start = state_norm < 1e-6

            if is_episode_start:
                latent = self.latent.expand(B, -1, -1).detach()
            else:
                latent = self.state_encoder.decode(rnn_states, B, device=x_data.device, dtype=x_data.dtype)

            x_data = x_data + self.pos_encoding[:, :1, :]
            latent, output = self._process(x_data, latent)
            out = output.unsqueeze(1)  # (B, 1, D)

            new_rnn_states = self.state_encoder.encode(latent.detach())

        out = out.permute(1, 0, 2)  # -> (1, B, D)

        if is_seq:
            T_full = x_data.shape[1]
            x = _pack_to_2d_sequence(out.expand(T_full, -1, -1).contiguous(), head_output)
            new_rnn_states = rnn_states
        else:
            x = out.squeeze(0)  # (B, D)

        return x, new_rnn_states


def make_perceiver_core(cfg, input_size):
    return PerceiverCore(cfg, input_size)


_Perceiver_REGISTERED = False


class PerceiverFactory:
    """Picklable factory that always returns PerceiverCore.

    Only registered when rnn_type='perceiver', so no fallback needed.
    """
    __slots__ = ('_num_layers',)
    def __init__(self, num_layers):
        self._num_layers = num_layers
    def __call__(self, cfg, core_input_size):
        cfg.rnn_num_layers = self._num_layers
        return PerceiverCore(cfg, core_input_size)


def register_perceiver(cfg=None):
    global _Perceiver_REGISTERED
    if _Perceiver_REGISTERED:
        return
    num_layers = cfg.rnn_num_layers if cfg is not None else 1
    factory = PerceiverFactory(num_layers)
    from sample_factory.algo.utils.context import global_model_factory
    global_model_factory().register_model_core_factory(factory)
    _Perceiver_REGISTERED = True
    print(f"[Perceiver] Registered as rnn_type='perceiver' (num_layers={num_layers})")


def get_perceiver_required_rnn_size(cfg):
    num_latents = getattr(cfg, 'perceiver_num_latents', 32)
    d_latents = getattr(cfg, 'perceiver_d_latents', 512)
    encoder = PerceiverStateEncoder(num_latents, d_latents)
    return encoder.total_size

"""GatedDeltaNet (FLA) based sequence model for sample-factory integration.

Pure-Triton chunk kernels from flash-linear-attention (fla-org). No C++
extension builds, no causal-conv1d. The recurrent (delta-rule) state plus
short-conv rolling buffers are mirrored into the sample-factory rnn buffer.

Mirrors models/mamba3_core.py so it drops into the same train.py/HPO paths.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from sample_factory.model.core import ModelCore
from sample_factory.utils.typing import Config

try:
    from fla.layers import GatedDeltaNet
    from fla.models.utils import FLACache
    GDN_AVAILABLE = True
except ImportError:
    GDN_AVAILABLE = False
    GatedDeltaNet = None
    FLACache = None


def _make_gdn_layer(d_model, head_dim, num_heads, expand_v, conv_size, layer_idx):
    return GatedDeltaNet(
        hidden_size=d_model, head_dim=head_dim, num_heads=num_heads,
        num_v_heads=num_heads, expand_v=expand_v, conv_size=conv_size,
        conv_bias=False, layer_idx=layer_idx,
    )


def _unpack_packed_sequence_2d(ps):
    data, bs, dim = ps.data, ps.batch_sizes, ps.data.shape[1]
    ml, mb = len(bs), int(bs[0].item())
    r = torch.zeros((ml, mb, dim), dtype=data.dtype, device=data.device)
    off = 0
    for t in range(ml):
        n = int(bs[t].item())
        r[t, :n] = data[off:off + n]
        off += n
    return r


def _pack_to_2d_sequence(x, orig):
    bs = orig.batch_sizes
    si, ui = getattr(orig, 'sorted_indices', None), getattr(orig, 'unsorted_indices', None)
    tot = int(bs.sum().item())
    pd = torch.empty((tot, x.shape[2]), dtype=x.dtype, device=x.device)
    off = 0
    for t in range(x.shape[0]):
        n = int(bs[t].item())
        pd[off:off + n] = x[t, :n]
        off += n
    return PackedSequence(pd, bs, si, ui)


class GDNStateEncoder:
    """Encodes/decodes GDN internal state to/from a flat tensor.

    Per-layer layout (mirrors the FLA FLACache dict entry):
      recurrent_state: (batch, num_v_heads, head_v_dim, head_k_dim)   fp32
      conv_state:      tuple(conv_q (batch, key_dim, K),
                             conv_k (batch, key_dim, K),
                             conv_v (batch, value_dim, K))
    """

    def __init__(self, num_v_heads, head_v_dim, head_k_dim, key_dim, value_dim, conv_size):
        self.num_v_heads = num_v_heads
        self.head_v_dim = head_v_dim
        self.head_k_dim = head_k_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.conv_size = conv_size
        self.recur_size = num_v_heads * head_v_dim * head_k_dim
        self.cq_size = key_dim * conv_size
        self.ck_size = key_dim * conv_size
        self.cv_size = value_dim * conv_size
        self.total_size = self.recur_size + self.cq_size + self.ck_size + self.cv_size

    def encode(self, rec, cq, ck, cv):
        b = rec.shape[0]
        return torch.cat(
            [rec.reshape(b, -1), cq.reshape(b, -1), ck.reshape(b, -1), cv.reshape(b, -1)],
            dim=-1,
        )

    def decode(self, flat):
        b = flat.shape[0]
        off = 0
        rec = flat[:, off:off + self.recur_size].reshape(b, self.num_v_heads, self.head_v_dim, self.head_k_dim)
        off += self.recur_size
        cq = flat[:, off:off + self.cq_size].reshape(b, self.key_dim, self.conv_size)
        off += self.cq_size
        ck = flat[:, off:off + self.ck_size].reshape(b, self.key_dim, self.conv_size)
        off += self.ck_size
        cv = flat[:, off:off + self.cv_size].reshape(b, self.value_dim, self.conv_size)
        off += self.cv_size
        return rec, cq, ck, cv

    def get_required_rnn_size(self):
        return self.total_size


class GDNCore(ModelCore):
    """GatedDeltaNet core that replaces GRU/LSTM in sample-factory."""

    def __init__(self, cfg, input_size):
        super().__init__(cfg)
        if not GDN_AVAILABLE:
            raise ImportError("flash-linear-attention (fla) with GatedDeltaNet not installed")
        self.cfg = cfg
        self.d_model = getattr(cfg, 'gdn_d_model', 512)
        self.num_heads = getattr(cfg, 'gdn_nheads', 8)
        self.head_dim = getattr(cfg, 'gdn_head_dim', 64)
        self.expand_v = getattr(cfg, 'gdn_expand_v', 2)
        self.conv_size = getattr(cfg, 'gdn_conv_size', 4)
        self.num_layers = getattr(cfg, 'rnn_num_layers', 1)

        # Build a reference layer to derive the exact key/value/head dims and
        # per-layer state footprint (the FLA kernels fix these from head_dim
        # + expand_v + num_heads).
        ref = _make_gdn_layer(self.d_model, self.head_dim, self.num_heads,
                              self.expand_v, self.conv_size, 0)
        self.key_dim = ref.key_dim
        self.value_dim = ref.value_dim
        self.num_v_heads = ref.num_v_heads
        self.head_k_dim = ref.head_k_dim
        self.head_v_dim = ref.head_v_dim
        del ref

        self.state_encoder = GDNStateEncoder(
            self.num_v_heads, self.head_v_dim, self.head_k_dim,
            self.key_dim, self.value_dim, self.conv_size,
        )
        required = self.state_encoder.total_size
        if cfg.rnn_size < required:
            raise ValueError(f"rnn_size ({cfg.rnn_size}) too small for GDN. Min: {required}")

        self.gdn_layers = nn.ModuleList([
            _make_gdn_layer(self.d_model, self.head_dim, self.num_heads,
                            self.expand_v, self.conv_size, i)
            for i in range(self.num_layers)
        ])
        self.input_proj = nn.Linear(input_size, self.d_model) if input_size != self.d_model else nn.Identity()
        self.output_norm = nn.LayerNorm(self.d_model)
        self.core_output_size = self.d_model

    def _forward_with_checkpointing(self, x):
        from torch.utils.checkpoint import checkpoint
        out = x
        for i, block in enumerate(self.gdn_layers):
            out = checkpoint(lambda x, b: b(x)[0], out, block, use_reentrant=False)
        return out

    def _forward_layers(self, x, cache=None):
        out = x
        for i, block in enumerate(self.gdn_layers):
            out = block(out, past_key_values=cache, use_cache=cache is not None)[0]
        return out

    def _create_inference_cache(self, batch_size, max_seqlen=256):
        return FLACache(seen_tokens=0)

    def _load_states_from_rnn(self, cache, rnn):
        ls = self.state_encoder.total_size
        b = rnn.shape[0]
        dev = rnn.device
        episode_start = rnn.norm().item() < 1e-6
        for i in range(self.num_layers):
            if episode_start:
                rec = torch.zeros(b, self.num_v_heads, self.head_v_dim, self.head_k_dim,
                                  dtype=torch.float32, device=dev)
                cq = torch.zeros(b, self.key_dim, self.conv_size, dtype=torch.float32, device=dev)
                ck = torch.zeros(b, self.key_dim, self.conv_size, dtype=torch.float32, device=dev)
                cv = torch.zeros(b, self.value_dim, self.conv_size, dtype=torch.float32, device=dev)
            else:
                flat = rnn[:, i * ls:(i + 1) * ls]
                if flat.shape[1] < ls:
                    continue
                rec, cq, ck, cv = self.state_encoder.decode(flat.float())
            cache.update(layer_idx=i, recurrent_state=rec, conv_state=(cq, ck, cv))

    def _get_inference_states(self, cache):
        all_s = []
        for i in range(self.num_layers):
            st = cache[i]
            rec = st['recurrent_state']
            cv = st['conv_state']
            all_s.append(self.state_encoder.encode(
                rec.float(), cv[0].float(), cv[1].float(), cv[2].float(),
            ))
        return torch.cat(all_s, dim=-1)

    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output)
        x = _unpack_packed_sequence_2d(head_output) if is_seq else head_output.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.input_proj(x)
        if is_seq:
            if getattr(self.cfg, 'gradient_checkpointing', True):
                x = self._forward_with_checkpointing(x)
            else:
                x = self._forward_layers(x)
            new_rnn_states = rnn_states
        else:
            bs = x.shape[0]
            ms = getattr(self.cfg, 'train_horizon', 256)
            cache = self._create_inference_cache(bs, ms)
            self._load_states_from_rnn(cache, rnn_states)
            x = self._forward_layers(x, cache=cache)
            new_rnn_states = self._get_inference_states(cache)
        x = self.output_norm(x)
        x = x.permute(1, 0, 2)
        if is_seq:
            x = _pack_to_2d_sequence(x, head_output)
        else:
            x = x.squeeze(0)
        return x, new_rnn_states


_GDN_REGISTERED = False


class GDNFactory:
    __slots__ = ('_num_layers',)

    def __init__(self, nl):
        self._num_layers = nl

    def __call__(self, cfg, cis):
        from sample_factory.model.core import default_make_core_func
        if cfg.use_rnn and getattr(cfg, 'rnn_type', 'gru') == 'gdn':
            cfg.rnn_num_layers = self._num_layers
            return GDNCore(cfg, cis)
        return default_make_core_func(cfg, cis)


def register_gdn(cfg=None):
    global _GDN_REGISTERED
    if _GDN_REGISTERED:
        return
    nl = cfg.rnn_num_layers if cfg is not None else 1
    from sample_factory.algo.utils.context import global_model_factory
    global_model_factory().register_model_core_factory(GDNFactory(nl))
    _GDN_REGISTERED = True
    print(f"[GDN] Registered as rnn_type='gdn' (num_layers={nl})")


def get_gdn_required_rnn_size(cfg):
    d_model = getattr(cfg, 'gdn_d_model', 512)
    nheads = getattr(cfg, 'gdn_nheads', 8)
    head_dim = getattr(cfg, 'gdn_head_dim', 64)
    expand_v = getattr(cfg, 'gdn_expand_v', 2)
    conv_size = getattr(cfg, 'gdn_conv_size', 4)
    ref = _make_gdn_layer(d_model, head_dim, nheads, expand_v, conv_size, 0)
    nl = getattr(cfg, 'rnn_num_layers', 1)
    return GDNStateEncoder(ref.num_v_heads, ref.head_v_dim, ref.head_k_dim,
                          ref.key_dim, ref.value_dim, conv_size).total_size * nl

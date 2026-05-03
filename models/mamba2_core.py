"""
Mamba-2 based sequence model for sample-factory integration.

Replaces GRU/LSTM with Mamba-2 (State Space Duality) for RL training.
Compatible with sample-factory's ModelCore interface.

Inference state management:
    Mamba-2 maintains internal conv_state and ssm_state that must persist
    across timesteps. We encode this state into rnn_states (which sample-factory
    manages and zeroes at episode boundaries). This ensures:
    - State persists across inference calls within an episode
    - State is properly reset at episode boundaries
    - No model-internal caching that could break with batched inference
"""

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from typing import Tuple, Optional, Dict, Any

from sample_factory.model.core import ModelCore
from sample_factory.model.model_utils import ModelModule
from sample_factory.utils.typing import Config


try:
    from mamba_ssm.modules.mamba2 import Mamba2 as Mamba2Block
    from mamba_ssm.utils.generation import InferenceParams
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    Mamba2Block = None
    InferenceParams = None


def _unpack_packed_sequence_2d(ps: PackedSequence) -> torch.Tensor:
    """
    Unpack a 2D PackedSequence (as created by sample-factory's build_rnn_inputs)
    into a 3D tensor (max_len, max_batch, dim).

    sample-factory creates PackedSequence from 2D tensors (N*T, dim),
    so .data is 2D unlike standard PyTorch PackedSequence.

    The data is laid out as: all first timesteps (sorted by length desc),
    then all second timesteps, etc. batch_sizes[i] tells how many sequences
    have length > i.

    Returns tensor with shape (max_len, max_batch, dim), padded with zeros.
    """
    data = ps.data  # (total_elements, dim)
    batch_sizes = ps.batch_sizes  # (max_len,), CPU tensor
    dim = data.shape[1]
    max_len = len(batch_sizes)
    max_batch = int(batch_sizes[0].item())

    result = torch.zeros((max_len, max_batch, dim), dtype=data.dtype, device=data.device)

    offset = 0
    for t in range(max_len):
        n = int(batch_sizes[t].item())
        result[t, :n] = data[offset:offset + n]
        offset += n

    return result


def _pack_to_2d_sequence(x_3d: torch.Tensor, orig_ps: PackedSequence) -> PackedSequence:
    """
    Pack a 3D tensor (max_len, max_batch, dim) back into a 2D PackedSequence
    matching the original structure from sample-factory.
    """
    batch_sizes = orig_ps.batch_sizes
    sorted_indices = getattr(orig_ps, 'sorted_indices', None)
    unsorted_indices = getattr(orig_ps, 'unsorted_indices', None)

    total = int(batch_sizes.sum().item())
    packed_data = torch.empty((total, x_3d.shape[2]), dtype=x_3d.dtype, device=x_3d.device)

    offset = 0
    for t in range(x_3d.shape[0]):
        n = int(batch_sizes[t].item())
        packed_data[offset:offset + n] = x_3d[t, :n]
        offset += n

    return PackedSequence(packed_data, batch_sizes, sorted_indices, unsorted_indices)


class Mamba2StateEncoder:
    """
    Encodes/decodes Mamba-2 internal state (conv_state + ssm_state) to/from
    a flat tensor that fits in rnn_states.

    This is the key mechanism that makes Mamba-2 work with sample-factory's
    state management: we store Mamba-2's internal state in rnn_states,
    which sample-factory zeroes at episode boundaries.
    """

    def __init__(self, d_ssm: int, d_conv_dim: int, nheads: int,
                 headdim: int, d_state: int, d_conv: int):
        self.d_ssm = d_ssm
        self.d_conv_dim = d_conv_dim
        self.nheads = nheads
        self.headdim = headdim
        self.d_state = d_state
        self.d_conv = d_conv

        # conv_state: (batch, d_conv_dim, d_conv)
        self.conv_state_size = d_conv_dim * d_conv
        # ssm_state: (batch, nheads, headdim, d_state)
        self.ssm_state_size = nheads * headdim * d_state
        self.total_size = self.conv_state_size + self.ssm_state_size

    def encode(self, conv_state: torch.Tensor,
               ssm_state: torch.Tensor) -> torch.Tensor:
        """
        Encode Mamba-2 state into a flat tensor.

        Args:
            conv_state: (batch, d_conv_dim, d_conv)
            ssm_state: (batch, nheads, headdim, d_state)

        Returns:
            Flat tensor: (batch, total_size)
        """
        batch = conv_state.shape[0]
        device = conv_state.device
        dtype = conv_state.dtype

        # Flatten each state
        conv_flat = conv_state.reshape(batch, -1)  # (batch, d_conv_dim * d_conv)
        ssm_flat = ssm_state.reshape(batch, -1)    # (batch, nheads * headdim * d_state)

        return torch.cat([conv_flat, ssm_flat], dim=-1)  # (batch, total_size)

    def decode(self, flat_state: torch.Tensor,
               device: torch.device = None,
               dtype: torch.dtype = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode a flat tensor back into Mamba-2 state.

        Args:
            flat_state: (batch, total_size)

        Returns:
            conv_state: (batch, d_conv_dim, d_conv)
            ssm_state: (batch, nheads, headdim, d_state)
        """
        if device is None:
            device = flat_state.device
        if dtype is None:
            dtype = flat_state.dtype

        batch = flat_state.shape[0]

        # Split into conv and ssm parts
        conv_flat = flat_state[:, :self.conv_state_size]
        ssm_flat = flat_state[:, self.conv_state_size:]

        # Reshape
        conv_state = conv_flat.reshape(batch, self.d_conv_dim, self.d_conv)
        ssm_state = ssm_flat.reshape(batch, self.nheads, self.headdim, self.d_state)

        return conv_state, ssm_state

    def get_required_rnn_size(self) -> int:
        """Return minimum rnn_size needed to store the encoded state."""
        return self.total_size


class Mamba2WithCache(nn.Module):
    """
    Wrapper around Mamba2Block that manages inference state.

    Sets layer_idx (required by mamba_ssm for state caching) and
    accepts inference_params for proper state management during inference.
    """

    def __init__(self, mamba_block: Mamba2Block, layer_idx: int):
        super().__init__()
        self.block = mamba_block
        self.block.layer_idx = layer_idx  # Required for inference cache lookup
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor,
                inference_params: Optional[InferenceParams] = None) -> torch.Tensor:
        return self.block(x, inference_params=inference_params)


class Mamba2Core(ModelCore):
    """
    Mamba-2 core that replaces GRU/LSTM in sample-factory.

    Architecture:
        obs -> encoder -> Mamba2Core -> decoder -> action/value heads

    Key design insight:
        - During training: Full sequences pass through mamba via PackedSequence.
          Episode boundaries are handled by sample-factory's build_rnn_inputs.
        - During inference: Mamba-2's internal state (conv_state + ssm_state)
          is encoded into rnn_states. Sample-factory manages these states
          and zeroes them at episode boundaries.

    Parameter mapping from sample-factory config:
        - rnn_size -> must be >= encoded state size (see get_required_rnn_size)
        - rnn_num_layers -> num Mamba2 blocks stacked
        - d_model -> Mamba hidden dimension (separate from rnn_size)
    """

    def __init__(self, cfg: Config, input_size: int):
        super().__init__(cfg)

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm not installed. Install with: "
                "pip install mamba-ssm --no-build-isolation"
            )

        self.cfg = cfg
        # d_model is the computation dimension (separate from rnn_size)
        # rnn_size must be >= encoded state size for inference
        self.d_model = getattr(cfg, 'mamba_d_model', cfg.rnn_size)
        # cfg.rnn_num_layers is overridden by the pickled factory before this
        # line runs, so it always has the correct registration-time value.
        self.num_layers = cfg.rnn_num_layers

        # Mamba-2 specific parameters with sensible defaults for RL
        self.d_state = getattr(cfg, 'mamba_d_state', 16)
        self.d_conv = getattr(cfg, 'mamba_d_conv', 4)
        self.expand = getattr(cfg, 'mamba_expand', 2)
        self.headdim = getattr(cfg, 'mamba_headdim', 64)
        self.ngroups = getattr(cfg, 'mamba_ngroups', 1)

        # Compute d_inner (used for d_ssm in Mamba-2)
        self.d_inner = int(self.expand * self.d_model)
        self.d_ssm = self.d_inner

        # nheads is computed from d_ssm, not d_model
        if self.d_ssm % self.headdim != 0:
            raise ValueError(
                f"d_ssm ({self.d_ssm}) must be divisible by mamba_headdim ({self.headdim})"
            )
        self.nheads = self.d_ssm // self.headdim

        # Conv dimension: d_ssm + 2*ngroups*d_state
        self.d_conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state

        # State encoder for inference
        self.state_encoder = Mamba2StateEncoder(
            d_ssm=self.d_ssm,
            d_conv_dim=self.d_conv_dim,
            nheads=self.nheads,
            headdim=self.headdim,
            d_state=self.d_state,
            d_conv=self.d_conv,
        )

        # Verify rnn_size is large enough per layer.
        # sample-factory multiplies rnn_size * rnn_num_layers for buffer allocation.
        required_per_layer = self.state_encoder.total_size
        if cfg.rnn_size < required_per_layer:
            raise ValueError(
                f"rnn_size ({cfg.rnn_size}) is too small for Mamba-2 state. "
                f"Minimum required per layer: {required_per_layer}. "
                f"Current config: d_model={self.d_model}, d_state={self.d_state}, "
                f"d_conv={self.d_conv}, expand={self.expand}, headdim={self.headdim}"
            )

        # Build stacked Mamba-2 blocks with pre-norm LayerNorm
        # Pre-norm is more stable for RL training than post-norm
        self.mamba_wrapped = nn.ModuleList()
        self.mamba_norms = nn.ModuleList()
        self.mamba_layers = nn.ModuleList()

        for i in range(self.num_layers):
            norm = nn.LayerNorm(self.d_model)
            block = Mamba2Block(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                headdim=self.headdim,
                ngroups=self.ngroups,
            )
            wrapped = Mamba2WithCache(block, layer_idx=i)
            self.mamba_wrapped.append(wrapped)
            self.mamba_norms.append(norm)
            self.mamba_layers.append(nn.Sequential(norm, wrapped))

        # Input projection: encoder output may differ from mamba's d_model
        if input_size != self.d_model:
            self.input_proj = nn.Linear(input_size, self.d_model)
        else:
            self.input_proj = nn.Identity()

        # core_output_size is d_model for decoder compatibility
        self.core_output_size = self.d_model

        # Inference params are transient - created fresh each inference call.
        # rnn_states (managed by sample-factory) is the source of truth.

    def _forward_with_checkpointing(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gradient checkpointing to reduce memory usage.

        Trades ~20-30% extra compute for ~50% memory savings during backprop.
        """
        from torch.utils.checkpoint import checkpoint

        def segment_forward(x, norm, mamba_layer):
            return mamba_layer(norm(x))

        output = x
        for norm, wrapped in zip(self.mamba_norms, self.mamba_wrapped):
            output = checkpoint(
                segment_forward,
                output, norm, wrapped,
                use_reentrant=False,
            )
        return output

    def _create_inference_params(self, batch_size: int, max_seqlen: int = 256):
        """
        Create fresh InferenceParams for a single inference call.

        InferenceParams is transient - it should NOT be cached across calls.
        rnn_states (managed by sample-factory) is the source of truth for
        state persistence. We decode rnn_states into InferenceParams,
        use it for the forward pass, then encode back.
        """
        params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
        )
        params.key_value_memory_dict = {}

        # Allocate per-layer state buffers
        for wrapped in self.mamba_wrapped:
            idx = wrapped.layer_idx
            conv_state, ssm_state = wrapped.block.allocate_inference_cache(
                batch_size, max_seqlen
            )
            params.key_value_memory_dict[idx] = (conv_state, ssm_state)

        return params

    def _load_states_from_rnn(self, inference_params: InferenceParams,
                               rnn_states: torch.Tensor):
        """
        Decode rnn_states into InferenceParams cache.

        Sample-factory zeroes rnn_states at episode boundaries. We detect
        this by checking if the L2 norm is below a threshold (more robust
        than exact zero check, handles numerical precision).
        """
        # Robust episode boundary detection
        # Sample-factory zeroes rnn_states on done; we use L2 norm as proxy
        # Threshold 1e-6 is safe: Mamba-2 states are typically >> 1e-6 after any processing
        state_norm = rnn_states.norm().item()
        is_episode_start = state_norm < 1e-6

        if is_episode_start:
            import logging
            logging.debug(
                f"Mamba2Core: Episode boundary detected (state_norm={state_norm:.2e})"
            )

        for wrapped in self.mamba_wrapped:
            idx = wrapped.layer_idx
            conv_state, ssm_state = inference_params.key_value_memory_dict[idx]

            if is_episode_start:
                # Episode just started - ensure clean state
                conv_state.zero_()
                ssm_state.zero_()
            else:
                # Decode state from rnn_states
                layer_state_size = self.state_encoder.total_size
                offset = idx * layer_state_size
                layer_flat = rnn_states[:, offset:offset + layer_state_size]

                if layer_flat.shape[1] < self.state_encoder.total_size:
                    # rnn_size too small for this layer - log warning
                    # This happens when rnn_size < required_rnn_size * num_layers
                    import warnings
                    warnings.warn(
                        f"Mamba2Core: rnn_size insufficient for layer {idx}. "
                        f"State not loaded. Increase rnn_size to at least "
                        f"{self.state_encoder.total_size * self.num_layers}.",
                        RuntimeWarning,
                        stacklevel=2
                    )
                else:
                    conv_decoded, ssm_decoded = self.state_encoder.decode(layer_flat)
                    conv_state.copy_(conv_decoded.to(conv_state.dtype))
                    ssm_state.copy_(ssm_decoded.to(ssm_state.dtype))

    def _get_inference_states(self, inference_params: InferenceParams) -> torch.Tensor:
        """Encode inference cache into rnn_states format."""
        device = list(inference_params.key_value_memory_dict.values())[0][0].device

        all_states = []
        for wrapped in self.mamba_wrapped:
            idx = wrapped.layer_idx
            conv_state, ssm_state = inference_params.key_value_memory_dict[idx]
            encoded = self.state_encoder.encode(conv_state, ssm_state)
            all_states.append(encoded)

        return torch.cat(all_states, dim=-1).to(device)

    def forward(self, head_output, rnn_states):
        """
        Forward pass through Mamba-2 layers.

        During training, head_output is a PackedSequence created by
        sample-factory's build_rnn_inputs(). Mamba-2 processes the full
        sequence without needing explicit state passing.

        During inference, head_output is a single-timestep tensor.
        Mamba-2's internal state is encoded in rnn_states and managed
        via transient InferenceParams (created fresh each call).

        Args:
            head_output: encoder output, shape (seq_len, batch, d_model) or PackedSequence
            rnn_states: state tensor from sample-factory. During inference,
                       contains encoded Mamba-2 state.

        Returns:
            x: mamba output, same structure as head_output
            new_rnn_states: updated state tensor with encoded Mamba-2 state
        """
        is_seq = not torch.is_tensor(head_output)

        if is_seq:
            # Training path: PackedSequence contains full sequences
            # Mamba-2 processes them without explicit state passing
            x_data = _unpack_packed_sequence_2d(head_output)
        else:
            # Regular tensor: add sequence dimension
            x_data = head_output.unsqueeze(0)  # (1, batch, dim)

        # Sample-factory uses (seq_len, batch, dim) but mamba expects (batch, seq_len, dim)
        x_data = x_data.permute(1, 0, 2)

        # Project input to mamba's d_model dimension
        x_data = self.input_proj(x_data)

        if is_seq:
            # Training: process full sequence with gradient checkpointing
            # Gradient checkpointing trades compute for memory - critical for
            # Mamba-2 which has large intermediate activations.
            # Config: cfg.gradient_checkpointing (default: True)
            if getattr(self.cfg, 'gradient_checkpointing', True):
                x_data = self._forward_with_checkpointing(x_data)
            else:
                x_data = self.mamba_layers(x_data)
        else:
            # Inference: create fresh InferenceParams, decode state from rnn_states
            batch_size = x_data.shape[0]
            max_seqlen = getattr(self.cfg, 'train_horizon', 256)

            # Create fresh InferenceParams for this call
            inference_params = self._create_inference_params(batch_size, max_seqlen)

            # Decode rnn_states into inference cache
            self._load_states_from_rnn(inference_params, rnn_states)

            # Set seqlen_offset so mamba uses step() mode for autoregressive decoding
            inference_params.seqlen_offset = 1

            # Forward through each layer with inference params
            for norm, wrapped in zip(self.mamba_norms, self.mamba_wrapped):
                x_data = wrapped(norm(x_data), inference_params=inference_params)

            # Encode updated state back into new_rnn_states
            new_rnn_states = self._get_inference_states(inference_params)

        # Convert back to sample-factory format (seq_len, batch, dim)
        x_data = x_data.permute(1, 0, 2)

        # Reconstruct output in original format
        if is_seq:
            # Pack back into 2D format that sample-factory expects
            x = _pack_to_2d_sequence(x_data, head_output)
            # During training, return rnn_states unchanged
            new_rnn_states = rnn_states
        else:
            x = x_data.squeeze(0)  # (batch, rnn_size)

        return x, new_rnn_states


def make_mamba2_core(cfg: Config, input_size: int) -> ModelCore:
    """Factory function for Mamba-2 core."""
    return Mamba2Core(cfg, input_size)


# Module-level flag to track if mamba-2 is registered
_MAMBA2_REGISTERED = False


class Mamba2Factory:
    """Picklable factory that remembers rnn_num_layers at registration time.

    Module-level globals are NOT shared across subprocesses — each worker
    imports the module fresh.  By using a callable class instance, the
    captured value travels with the pickled factory.
    """
    __slots__ = ('_num_layers',)
    def __init__(self, num_layers):
        self._num_layers = num_layers
    def __call__(self, cfg, core_input_size):
        from sample_factory.model.core import default_make_core_func
        if cfg.use_rnn and getattr(cfg, 'rnn_type', 'gru') == 'mamba2':
            # Override cfg so Mamba2Core always sees the correct value
            cfg.rnn_num_layers = self._num_layers
            return Mamba2Core(cfg, core_input_size)
        return default_make_core_func(cfg, core_input_size)


def register_mamba2(cfg=None):
    """
    Register Mamba-2 as a valid rnn_type option.

    Call this once during initialization:
        from models.mamba2_core import register_mamba2
        register_mamba2(cfg)   # pass cfg so rnn_num_layers is captured
    """
    global _MAMBA2_REGISTERED
    if _MAMBA2_REGISTERED:
        return

    num_layers = cfg.rnn_num_layers if cfg is not None else 1
    factory = Mamba2Factory(num_layers)

    from sample_factory.algo.utils.context import global_model_factory
    global_model_factory().register_model_core_factory(factory)

    _MAMBA2_REGISTERED = True
    print(f"[Mamba2] Registered as rnn_type='mamba2'  (num_layers={num_layers})")


def get_mamba2_required_rnn_size(cfg: Config) -> int:
    """
    Calculate minimum per-layer rnn_size required for Mamba-2 state encoding.

    Returns per-layer size. sample-factory multiplies by rnn_num_layers
    internally, so the effective total will be per_layer * num_layers.
    """
    d_model = getattr(cfg, 'mamba_d_model', cfg.rnn_size)
    d_state = getattr(cfg, 'mamba_d_state', 16)
    d_conv = getattr(cfg, 'mamba_d_conv', 4)
    expand = getattr(cfg, 'mamba_expand', 2)
    headdim = getattr(cfg, 'mamba_headdim', 64)
    ngroups = getattr(cfg, 'mamba_ngroups', 1)

    # nheads is computed from d_ssm, not d_model
    d_inner = int(expand * d_model)
    nheads = d_inner // headdim
    d_conv_dim = d_inner + 2 * ngroups * d_state

    encoder = Mamba2StateEncoder(d_inner, d_conv_dim, nheads, headdim, d_state, d_conv)
    return encoder.total_size  # per-layer only

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boundary_predictor import BoundaryPredictor, gumbel_sigmoid_boundaries


class CausalTransformerStack(nn.Module):
    """
    A stack of pre-norm causal Transformer encoder layers, used as the shared
    building block for all three stages of [[HourglassTransformer]] (the
    tokenization submodule, the middle transformer, and the final transformer).
    """

    def __init__(self, d_model, n_heads, n_layers, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x, key_padding_mask=None, causal=True):
        """
        Args:
            x: (batch, seq_len, d_model)
            key_padding_mask: (batch, seq_len) bool, True at positions to
                mask out (PyTorch convention)
            causal: apply a causal (subsequent-position) mask
        """
        attn_mask = None
        if causal:
            # bool, not the default float -inf mask, so it matches
            # key_padding_mask's dtype (PyTorch warns/deprecates otherwise)
            attn_mask = torch.triu(
                torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device), diagonal=1
            )
        return self.layers(x, mask=attn_mask, src_key_padding_mask=key_padding_mask, is_causal=causal)


def _finalize_boundaries(boundaries, attention_mask):
    """
    Cleans up hard boundary indicators before segment-id computation. Zeroes
    out any boundary sampled at a padded position (the Gumbel-sigmoid can,
    rarely, still sample 1 there despite a near-zero masked probability),
    then forces a boundary at each sequence's last real position (so the
    final real segment closes cleanly) and at the tensor's absolute last
    position — otherwise, if a padded tail's last position isn't itself a
    boundary, its segment id computes to num_segments, one past the valid
    range sized by num_segments. Assumes right-padding (real tokens first,
    padding after) when attention_mask is given.
    """
    boundaries = boundaries.clone()
    if attention_mask is None:
        boundaries[:, -1] = 1.0
        return boundaries
    boundaries = boundaries * attention_mask.to(boundaries.dtype)
    last_real_idx = attention_mask.sum(dim=1).long() - 1
    boundaries[torch.arange(boundaries.size(0), device=boundaries.device), last_real_idx] = 1.0
    boundaries[:, -1] = 1.0
    return boundaries


def _compute_segment_ids(boundaries):
    """
    Maps a (batch, seq_len) {0,1} boundary indicator to per-position segment
    ids, where a boundary at t closes segment id at t and t+1 starts the next
    segment id.
    """
    shifted = F.pad(boundaries[:, :-1], (1, 0), value=0.0)
    return torch.cumsum(shifted, dim=1).long()


def _segment_mean_pool(hidden_states, segment_ids, num_segments):
    """
    Mean-pools hidden states within each predicted segment.

    Args:
        hidden_states: (batch, seq_len, d_model)
        segment_ids: (batch, seq_len) long, segment id per position
        num_segments: (batch,) long, number of segments per example

    Returns:
        pooled: (batch, max_segments, d_model)
        key_padding_mask: (batch, max_segments) bool, True at padded segment
            slots beyond an example's own segment count (PyTorch convention)
    """
    batch, _, d_model = hidden_states.shape
    max_segments = int(num_segments.max().item())

    pooled_sum = hidden_states.new_zeros(batch, max_segments, d_model)
    counts = hidden_states.new_zeros(batch, max_segments)

    idx = segment_ids.unsqueeze(-1).expand(-1, -1, d_model)
    pooled_sum.scatter_add_(1, idx, hidden_states)
    counts.scatter_add_(1, segment_ids, torch.ones_like(segment_ids, dtype=hidden_states.dtype))

    pooled = pooled_sum / counts.clamp(min=1).unsqueeze(-1)
    valid = torch.arange(max_segments, device=hidden_states.device).unsqueeze(0) < num_segments.unsqueeze(1)
    key_padding_mask = ~valid
    return pooled, key_padding_mask


def _segment_upsample(pooled, segment_ids):
    """
    Duplication upsampling: broadcasts each segment's pooled representation
    back out to every position that belongs to it.
    """
    d_model = pooled.size(-1)
    idx = segment_ids.unsqueeze(-1).expand(-1, -1, d_model)
    return torch.gather(pooled, 1, idx)


class HourglassTransformer(nn.Module):
    """
    MAGNET's three-stage "hourglass" architecture (Ahia et al., 2024):

    1. Tokenization submodule: a full-resolution causal transformer produces
       per-position hidden states, which a [[BoundaryPredictor]] turns into
       segment boundaries (Eq. 1-2).
    2. Middle transformer: hidden states are mean-pooled within each
       predicted segment, and a causal transformer operates on this shorter,
       downsampled sequence of segment representations.
    3. Upsampling + final transformer: each segment's middle-transformer
       output is duplicated back out to its member positions, added as a
       skip connection to the original full-resolution hidden states, and
       passed through a final (small) causal transformer and an unembedding
       layer to produce next-token logits.
    """

    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=4,
        n_layers_tokenization=2,
        n_layers_middle=6,
        n_layers_final=2,
        d_ff=None,
        dropout=0.0,
        boundary_predictor=None,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.tokenization_transformer = CausalTransformerStack(
            d_model, n_heads, n_layers_tokenization, d_ff, dropout
        )
        self.boundary_predictor = boundary_predictor or BoundaryPredictor(d_model)
        self.middle_transformer = CausalTransformerStack(d_model, n_heads, n_layers_middle, d_ff, dropout)
        self.upsample_norm = nn.LayerNorm(d_model)
        self.final_transformer = CausalTransformerStack(d_model, n_heads, n_layers_final, d_ff, dropout)
        self.unembed = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None, boundary_predictor_kwargs=None):
        """
        Args:
            input_ids: (batch, seq_len) token ids
            attention_mask: (batch, seq_len) bool, True at real (non-padded)
                positions. Assumes right-padding.
            boundary_predictor_kwargs: extra kwargs forwarded to
                self.boundary_predictor's forward (e.g. `language_ids` for
                [[magnet.LanguageRoutedBoundaryPredictor]])

        Returns:
            logits: (batch, seq_len, vocab_size)
            boundary_probs: (batch, seq_len) soft boundary probabilities,
                for use in the binomial regularizer ([[losses.binomial_regularizer]], Eq. 3)
        """
        boundary_predictor_kwargs = boundary_predictor_kwargs or {}
        key_padding_mask = ~attention_mask if attention_mask is not None else None

        hidden_states = self.embed(input_ids)
        hidden_states = self.tokenization_transformer(hidden_states, key_padding_mask=key_padding_mask)

        boundary_probs = self.boundary_predictor(hidden_states, **boundary_predictor_kwargs)
        if attention_mask is not None:
            boundary_probs = boundary_probs * attention_mask.to(boundary_probs.dtype)

        boundaries = gumbel_sigmoid_boundaries(boundary_probs, training=self.training)
        boundaries = _finalize_boundaries(boundaries, attention_mask)

        segment_ids = _compute_segment_ids(boundaries)
        num_segments = boundaries.sum(dim=1).long()
        pooled, pooled_key_padding_mask = _segment_mean_pool(hidden_states, segment_ids, num_segments)

        pooled = self.middle_transformer(pooled, key_padding_mask=pooled_key_padding_mask)

        upsampled = _segment_upsample(pooled, segment_ids)
        hidden_states = self.upsample_norm(upsampled + hidden_states)

        hidden_states = self.final_transformer(hidden_states, key_padding_mask=key_padding_mask)
        logits = self.unembed(hidden_states)
        return logits, boundary_probs

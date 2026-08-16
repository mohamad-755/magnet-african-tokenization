import torch.nn as nn

from .boundary_predictor import BoundaryPredictor
from .hourglass_transformer import HourglassTransformer


class ScriptRoutedBoundaryPredictor(nn.Module):
    """
    Script-specific boundary predictors, routed by an input script tag
    (MAGNET, Ahia et al. 2024, Section 2.2).

    Rather than one boundary predictor shared across every language/script
    (which over-segments non-Latin scripts, per the paper's motivation), this
    module holds one [[BoundaryPredictor]] per script and routes each
    sequence in a batch to its own script's predictor using an integer
    script id supplied alongside the input.
    """

    def __init__(self, d_model, scripts, d_hidden=None):
        super().__init__()
        self.scripts = list(scripts)
        self.predictors = nn.ModuleList([BoundaryPredictor(d_model, d_hidden) for _ in self.scripts])

    def forward(self, hidden_states, script_ids):
        """
        Args:
            hidden_states: (batch, seq_len, d_model)
            script_ids: (batch,) long tensor indexing into self.scripts

        Returns:
            (batch, seq_len) boundary probabilities, each position scored by
            its example's own script-specific predictor
        """
        probs = hidden_states.new_zeros(hidden_states.shape[:2])
        for script_idx in script_ids.unique().tolist():
            mask = script_ids == script_idx
            probs[mask] = self.predictors[script_idx](hidden_states[mask])
        return probs


class MAGNET(nn.Module):
    """
    Full MAGNET model (Ahia et al., 2024): a [[HourglassTransformer]] whose
    single boundary predictor is replaced with a
    [[ScriptRoutedBoundaryPredictor]], so each language script learns its own
    segmentation granularity (Section 2.2) instead of sharing one predictor
    across scripts.
    """

    def __init__(
        self,
        vocab_size,
        scripts=("Latin", "Geez"),
        d_model=256,
        n_heads=4,
        n_layers_tokenization=2,
        n_layers_middle=6,
        n_layers_final=2,
        d_ff=None,
        dropout=0.0,
    ):
        super().__init__()
        self.scripts = list(scripts)
        boundary_predictor = ScriptRoutedBoundaryPredictor(d_model, self.scripts)
        self.hourglass = HourglassTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers_tokenization=n_layers_tokenization,
            n_layers_middle=n_layers_middle,
            n_layers_final=n_layers_final,
            d_ff=d_ff,
            dropout=dropout,
            boundary_predictor=boundary_predictor,
        )

    def forward(self, input_ids, script_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len) token ids
            script_ids: (batch,) long tensor indexing into self.scripts
            attention_mask: (batch, seq_len) bool, True at real (non-padded)
                positions

        Returns:
            logits: (batch, seq_len, vocab_size)
            boundary_probs: (batch, seq_len)
        """
        return self.hourglass(
            input_ids,
            attention_mask=attention_mask,
            boundary_predictor_kwargs={"script_ids": script_ids},
        )

import torch.nn as nn

from .boundary_predictor import BoundaryPredictor
from .hourglass_transformer import HourglassTransformer


class LanguageRoutedBoundaryPredictor(nn.Module):
    """
    Per-language boundary predictors, routed by an input language tag.

    Unlike the paper's design (Ahia et al. 2024, Section 2.2 — one
    predictor per SCRIPT, e.g. shared across every Latin-script language),
    this instantiates one [[BoundaryPredictor]] per LANGUAGE. This is this
    project's own extension, not something the paper specifies: it tests
    whether finer-grained-than-script routing changes segmentation
    equitability among languages that already share a script but differ
    in morphology (see README's motivation).
    """

    def __init__(self, d_model, languages, d_hidden=None):
        super().__init__()
        self.languages = list(languages)
        self.predictors = nn.ModuleList([BoundaryPredictor(d_model, d_hidden) for _ in self.languages])

    def forward(self, hidden_states, language_ids):
        """
        Args:
            hidden_states: (batch, seq_len, d_model)
            language_ids: (batch,) long tensor indexing into self.languages

        Returns:
            (batch, seq_len) boundary probabilities, each position scored by
            its example's own language-specific predictor
        """
        probs = hidden_states.new_zeros(hidden_states.shape[:2])
        for lang_idx in language_ids.unique().tolist():
            mask = language_ids == lang_idx
            probs[mask] = self.predictors[lang_idx](hidden_states[mask])
        return probs


class MAGNET(nn.Module):
    """
    MAGNET (Ahia et al., 2024) variant used by this project: a
    [[HourglassTransformer]] whose single boundary predictor is replaced
    with a [[LanguageRoutedBoundaryPredictor]], so each language learns its
    own segmentation granularity — a finer-grained routing than the
    paper's own script-level design (Section 2.2), see
    [[LanguageRoutedBoundaryPredictor]]'s docstring.
    """

    def __init__(
        self,
        vocab_size,
        languages,
        d_model=256,
        n_heads=4,
        n_layers_tokenization=2,
        n_layers_middle=6,
        n_layers_final=2,
        d_ff=None,
        dropout=0.0,
    ):
        super().__init__()
        self.languages = list(languages)
        boundary_predictor = LanguageRoutedBoundaryPredictor(d_model, self.languages)
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

    def forward(self, input_ids, language_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len) token ids
            language_ids: (batch,) long tensor indexing into self.languages
            attention_mask: (batch, seq_len) bool, True at real (non-padded)
                positions

        Returns:
            logits: (batch, seq_len, vocab_size)
            boundary_probs: (batch, seq_len)
        """
        return self.hourglass(
            input_ids,
            attention_mask=attention_mask,
            boundary_predictor_kwargs={"language_ids": language_ids},
        )

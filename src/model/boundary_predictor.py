import torch
import torch.nn as nn


class BoundaryPredictor(nn.Module):
    """
    Per-position segment boundary predictor (MAGNET, Ahia et al. 2024, Eq. 1).

    A 2-layer MLP over each position's hidden state, followed by a sigmoid,
    producing a boundary probability p_t in [0, 1] for every position t. A
    high p_t means the tokenization submodule believes t ends a segment.
    """

    def __init__(self, d_model, d_hidden=None):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, hidden_states):
        """
        Eq. 1: p_t = sigmoid(MLP(h_t))

        Args:
            hidden_states: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len) boundary probabilities in [0, 1]
        """
        return torch.sigmoid(self.mlp(hidden_states).squeeze(-1))


def gumbel_sigmoid_boundaries(probs, tau=1.0, hard=True, training=True, eps=1e-8):
    """
    Hard Gumbel-sigmoid stochastic reparameterization (MAGNET Eq. 2).

    Turns soft boundary probabilities into a discrete {0, 1} boundary
    indicator that stays differentiable via the straight-through estimator,
    so segmentation can be trained jointly with the language modeling
    objective. Standard binary-Concrete/Gumbel-sigmoid construction: Logistic
    noise (the difference of two Gumbel(0,1) samples) is added to the
    boundary logits before the sigmoid.

    At eval time (training=False) this collapses to a deterministic
    threshold at p_t = 0.5, with no sampling noise.
    """
    if not training:
        return (probs > 0.5).to(probs.dtype)

    probs = probs.clamp(eps, 1 - eps)
    logits = torch.log(probs) - torch.log1p(-probs)

    u = torch.rand_like(probs).clamp(eps, 1 - eps)
    logistic_noise = torch.log(u) - torch.log1p(-u)

    y_soft = torch.sigmoid((logits + logistic_noise) / tau)
    if not hard:
        return y_soft

    y_hard = (y_soft > 0.5).to(y_soft.dtype)
    return y_hard - y_soft.detach() + y_soft

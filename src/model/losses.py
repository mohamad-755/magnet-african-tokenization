import torch
import torch.nn.functional as F

from .boundary_predictor import gumbel_sigmoid_boundaries


def next_token_prediction_loss(logits, targets, attention_mask=None, ignore_index=-100):
    """
    Standard autoregressive next-token cross-entropy loss (MAGNET's language
    modeling term, combined with [[binomial_regularizer]] in the paper's
    overall training objective).

    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len) token ids, aligned with logits (this
            function performs the next-token shift internally)
        attention_mask: (batch, seq_len) bool, True at real (non-padded)
            positions
    """
    shifted_logits = logits[:, :-1, :]
    shifted_targets = targets[:, 1:]
    if attention_mask is not None:
        shifted_targets = shifted_targets.masked_fill(~attention_mask[:, 1:], ignore_index)
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        shifted_targets.reshape(-1),
        ignore_index=ignore_index,
    )


def binomial_regularizer(boundary_probs, target_rate, attention_mask=None, tau=1.0):
    """
    Binomial regularizer nudging a boundary predictor's aggregate segment
    count toward what a Binomial(N, beta) distribution predicts (MAGNET
    Eq. 3):

        regularizer = -log C(N, k) - k*log(beta) - (N-k)*log(1-beta)

    computed per sequence, where N is the sequence length, beta is the
    target compression rate, and k is a per-sequence *soft* segment count:
    the sum over the sequence of a soft Gumbel-sigmoid sample (Eq. 2,
    without hard thresholding) drawn from boundary_probs. log C(N, k) uses
    the continuous (lgamma) relaxation of the binomial coefficient, since k
    is generally not an integer:

        log C(N, k) = lgamma(N+1) - lgamma(k+1) - lgamma(N-k+1)

    This regularizes each sequence's *aggregate* boundary count k, not each
    position's boundary probability independently — a materially weaker
    constraint than per-position BCE against beta (the previous, incorrect
    implementation here), which forces every position toward beta
    regardless of the sequence's overall segment count.

    Args:
        boundary_probs: (batch, seq_len) boundary probabilities (Eq. 1), as
            returned by [[hourglass_transformer.HourglassTransformer.forward]]
        target_rate: target compression rate beta — a python float, or a
            tensor broadcastable to (batch,) for a per-script beta via
            [[script_target_rates]]
        attention_mask: (batch, seq_len) bool, True at real (non-padded)
            positions
        tau: Gumbel-sigmoid temperature (Eq. 2)
    """
    soft_boundaries = gumbel_sigmoid_boundaries(boundary_probs, tau=tau, hard=False, training=True)

    if attention_mask is not None:
        mask = attention_mask.to(soft_boundaries.dtype)
        soft_boundaries = soft_boundaries * mask
        n = mask.sum(dim=1)
    else:
        n = soft_boundaries.new_full((soft_boundaries.size(0),), soft_boundaries.size(1))

    k = soft_boundaries.sum(dim=1)

    if torch.is_tensor(target_rate):
        beta = target_rate.reshape(-1).to(k.dtype)
    else:
        beta = torch.full_like(k, float(target_rate))
    beta = beta.clamp(1e-6, 1 - 1e-6)

    log_binom_coeff = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    reg = -log_binom_coeff - k * torch.log(beta) - (n - k) * torch.log1p(-beta)
    return reg.mean()


def script_target_rates(script_ids, beta_by_script):
    """
    Expands a per-script target compression rate beta into a (batch,)
    tensor aligned with a batch's script_ids, for use as `target_rate` in
    [[binomial_regularizer]] when a [[magnet.MAGNET]] model routes through
    multiple script-specific boundary predictors (Section 2.2).

    Args:
        script_ids: (batch,) long tensor of script indices
        beta_by_script: sequence/tensor of target rates indexed by script id
    """
    beta_by_script = torch.as_tensor(beta_by_script, dtype=torch.float32, device=script_ids.device)
    return beta_by_script[script_ids]


def magnet_loss(logits, targets, boundary_probs, target_rate, reg_weight=1.0,
                 attention_mask=None, ignore_index=-100):
    """
    MAGNET's combined training objective: next-token prediction loss plus a
    lambda-weighted binomial regularizer (Eq. 3).

    Returns:
        total_loss: scalar
        components: dict with detached `lm_loss` and `reg_loss` for logging
    """
    lm_loss = next_token_prediction_loss(logits, targets, attention_mask, ignore_index)
    reg_loss = binomial_regularizer(boundary_probs, target_rate, attention_mask)
    total_loss = lm_loss + reg_weight * reg_loss
    return total_loss, {"lm_loss": lm_loss.detach(), "reg_loss": reg_loss.detach()}

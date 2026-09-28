"""KD loss variants for correction training.

SignRoundV2 (arXiv 2512.04746) section 3.4 excludes the top fraction of
losses when fitting the quantizer (``k`` = 0.1% of elements) to keep outlier
samples from dominating; the same trick applies to the output-KD term here.
Pure tensor functions so the trainer change is small and CPU-testable.

Reference: SignRoundV2, Eq. 12 (mean over the losses with the top-k values
removed).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_per_token(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                 temp: float = 2.0) -> torch.Tensor:
    """Per-token KL(teacher || student) over the selected vocab, scaled by T^2."""
    logp = F.log_softmax(student_logits.float() / temp, dim=-1)
    tlogp = F.log_softmax(teacher_logits.float() / temp, dim=-1)
    return (tlogp.exp() * (tlogp - logp)).sum(-1) * (temp ** 2)


def filtered_mean(losses: torch.Tensor, filter_frac: float = 0.0) -> torch.Tensor:
    """Mean of ``losses`` after dropping the largest ``filter_frac`` fraction."""
    flat = losses.reshape(-1).float()
    if filter_frac <= 0.0 or flat.numel() == 0:
        return flat.mean()
    # never drop everything: at least one loss survives even at filter_frac=1
    n_drop = min(int(flat.numel() * filter_frac), flat.numel() - 1)
    if n_drop <= 0:
        return flat.mean()
    return torch.sort(flat).values[:-n_drop].mean()


def kd_filtered(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                temp: float = 2.0, filter_frac: float = 0.0) -> torch.Tensor:
    """Output-KD term with optional loss filtering (`filter_frac` = 0 keeps all)."""
    return filtered_mean(kl_per_token(student_logits, teacher_logits, temp), filter_frac)

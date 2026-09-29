"""Multi-token-prediction head for the prefix (Phase E item 11).

Predict-ahead (MTP) is the practical form of the predictive-coding idea: the
head sees the current hidden state and the *next* token's embedding and predicts
the token after it.  Two payoffs are testable locally on the prefix harness:

1. **Drafting**: the head's top-1 is a draft for the main model's next-next
   token; the acceptance rate (``draft_acceptance``) measures whether a
   speculative loop would pay off.
2. **Auxiliary supervision**: adding the ``t+2`` loss may improve the main
   model's own next-token distribution (Meta/DeepSeek MTP reports), measurable
   on the same KLD/PPL gate as everything else.

The head is deliberately small — a linear mixing of RMS-normed hidden state and
embedding — and reuses the frozen final norm + lm_head, so it adds
``2*hidden²`` parameters, not a second vocabulary projection.  The native
``mtp.*`` weights in the checkpoint are *not* usable on a 4-layer prefix: they
were trained against the full 40-layer model's hidden states.

Pure torch, CPU-testable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTPHead(nn.Module):
    """``fc([rms(h_t); rms(emb(x_{t+1}))])`` -> hidden space (norm/lm_head shared)."""

    def __init__(self, hidden: int, init_std: float = 0.02):
        super().__init__()
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        nn.init.normal_(self.fc.weight, std=init_std)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        x = torch.cat([F.rms_norm(h.float(), (h.shape[-1],), eps=1e-6),
                       F.rms_norm(e.float(), (e.shape[-1],), eps=1e-6)], dim=-1)
        return self.fc(x)


def mtp_logits(model, head: MTPHead, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Head output through the model's own final norm + lm_head (both frozen)."""
    out = head(h, e)
    if out.dtype != h.dtype:
        # the head computes in fp32; norm/lm_head run in the model dtype
        out = out.to(h.dtype)
    norm = getattr(model, "norm", None)
    if norm is not None:
        out = norm(out)
    lm = getattr(model, "lm_head", None)
    if lm is None:
        raise AttributeError("model has no lm_head to share with the MTP head")
    return lm(out)


def mtp_targets(hidden: torch.Tensor, ids: torch.Tensor):
    """Inputs/targets for the ``t+2`` prediction, aligned with ``ids``.

    Returns ``(h[:, :-2], emb(ids[:, 1:-1]), ids[:, 2:])`` so the head's output
    at position ``t`` predicts ``ids[t+2]``.
    """
    if ids.shape[1] < 3:
        raise ValueError("need at least 3 tokens for a t+2 target")
    return hidden[:, :-2], ids[:, 1:-1], ids[:, 2:]


def draft_acceptance(main_logits: torch.Tensor, mtp_logits: torch.Tensor,
                     topk: int = 1) -> float:
    """Fraction of positions where the draft matches the main model's greedy token.

    ``main_logits[:, t]`` predicts ``ids[t+1]`` and ``mtp_logits[:, t]`` predicts
    ``ids[t+2]``, so the two predict the same token at ``main[:, t+1]`` vs
    ``mtp[:, t]``.  Greedy agreement is the standard first-order acceptance
    proxy; real speculative acceptance also depends on the sampling scheme.
    """
    main = main_logits[:, 1:-1].argmax(-1)
    if topk <= 1:
        return float((mtp_logits.argmax(-1) == main).float().mean())
    cand = mtp_logits.topk(topk, dim=-1).indices
    return float((cand == main.unsqueeze(-1)).any(-1).float().mean())


def chunked_ce(logits: torch.Tensor, targets: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Cross-entropy over the token axis in chunks (bounds the fp32 temporary).

    A full ``(1, T, 248k)`` fp32 softmax is ~0.5 GB; the MTP loss is a second
    one on top of the LM loss, which is where a 20 GB card would feel it.
    """
    total = logits.new_zeros(())
    n = 0
    for i in range(0, logits.shape[1], chunk):
        part = logits[:, i:i + chunk].reshape(-1, logits.shape[-1]).float()
        tgt = targets[:, i:i + chunk].reshape(-1)
        total = total + F.cross_entropy(part, tgt, reduction="sum")
        n += tgt.numel()
    return total / max(n, 1)

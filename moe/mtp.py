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
    """``fc([rms(h_t); rms(emb(x_{t+1}))])`` -> hidden space (norm/lm_head shared).

    ``layers=2`` adds one wider hidden layer (GELU between two projections);
    ``layers=1`` is the original single linear and keeps the ``fc.weight``
    parameter name for checkpoint compatibility.
    """

    def __init__(self, hidden: int, init_std: float = 0.02, layers: int = 1):
        super().__init__()
        if layers not in (1, 2):
            raise ValueError(f"layers must be 1 or 2, got {layers}")
        self.layers = layers
        if layers == 1:
            self.fc = nn.Linear(2 * hidden, hidden, bias=False)
            nn.init.normal_(self.fc.weight, std=init_std)
        else:
            self.fc1 = nn.Linear(2 * hidden, 2 * hidden, bias=False)
            self.fc2 = nn.Linear(2 * hidden, hidden, bias=False)
            nn.init.normal_(self.fc1.weight, std=init_std)
            nn.init.normal_(self.fc2.weight, std=init_std)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        x = torch.cat([F.rms_norm(h.float(), (h.shape[-1],), eps=1e-6),
                       F.rms_norm(e.float(), (e.shape[-1],), eps=1e-6)], dim=-1)
        if self.layers == 1:
            return self.fc(x)
        return self.fc2(F.gelu(self.fc1(x)))


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


def self_target_main(main_logits: torch.Tensor) -> torch.Tensor:
    """The main model's own greedy token at the draft position (``t+2``).

    ``main_logits[:, t]`` predicts ``ids[t+1]``, so the token the draft head
    must match at position ``t`` is ``main_logits[:, t+1].argmax()`` -- exactly
    what :func:`draft_acceptance` scores.  Training against this target instead
    of the corpus token is the acceptance-oriented objective: the head learns
    the main model's choices, not the text's.
    """
    return main_logits[:, 1:-1].argmax(-1).detach()


def draft_acceptance(main_logits: torch.Tensor, mtp_logits: torch.Tensor,
                     topk: int = 1) -> float:
    """Fraction of positions where the draft matches the main model's greedy token.

    ``main_logits[:, t]`` predicts ``ids[t+1]`` and ``mtp_logits[:, t]`` predicts
    ``ids[t+2]``, so the two predict the same token at ``main[:, t+1]`` vs
    ``mtp[:, t]``.  Greedy agreement is the standard first-order acceptance
    proxy; real speculative acceptance also depends on the sampling scheme.
    """
    main = self_target_main(main_logits)
    if topk <= 1:
        return float((mtp_logits.argmax(-1) == main).float().mean())
    cand = mtp_logits.topk(topk, dim=-1).indices
    return float((cand == main.unsqueeze(-1)).any(-1).float().mean())


def freeze_except_head(model, head) -> int:
    """Freeze every model parameter except the head's; return head params.

    The frozen-body drafter protocol: after the correction recipe is final,
    train the head alone, so the auxiliary objective cannot perturb the main
    model (a joint ``--mtp-weight`` arm pays a small main-gate cost).  The
    head is attached to ``model`` (``model._mtp_head``), so it must be
    unfrozen *after* the blanket freeze.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for p in head.parameters():
        p.requires_grad_(True)
    return sum(p.numel() for p in head.parameters() if p.requires_grad)


def chunked_kl(logits: torch.Tensor, ref_logits: torch.Tensor,
               temp: float = 1.0, chunk: int = 64) -> torch.Tensor:
    """Chunked ``KL(ref || softmax(logits)) * temp**2``, mean over tokens.

    The soft self-distill target: the head imitates the main model's own
    next-token distribution at the draft position (temperature ``temp``),
    which is the differentiable relaxation of the acceptance objective.  The
    ``temp**2`` scaling matches the KD convention (gradient scale invariant to
    the temperature choice).
    """
    total = logits.new_zeros(())
    n = 0
    for i in range(0, logits.shape[1], chunk):
        part = logits[:, i:i + chunk].reshape(-1, logits.shape[-1]).float() / temp
        ref = ref_logits[:, i:i + chunk].reshape(-1, ref_logits.shape[-1]).float() / temp
        total = total + F.kl_div(F.log_softmax(part, dim=-1),
                                 F.log_softmax(ref, dim=-1),
                                 log_target=True, reduction="sum")
        n += part.shape[0]
    return total / max(n, 1) * (float(temp) ** 2)


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

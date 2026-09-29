"""CPU tests for the MTP head and the acceptance metric (no model needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

MOE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOE))

import mtp as M  # noqa: E402


def test_head_shape_and_gradients():
    torch.manual_seed(0)
    head = M.MTPHead(hidden=32)
    h = torch.randn(2, 10, 32)
    e = torch.randn(2, 10, 32)
    out = head(h, e)
    assert out.shape == (2, 10, 32)
    out.sum().backward()
    assert head.fc.weight.grad is not None
    assert torch.isfinite(head.fc.weight.grad).all()


def test_mtp_targets_align_t_plus_two():
    ids = torch.arange(12).reshape(1, 12)
    h = torch.randn(1, 12, 8)
    h_in, e_in, tgt = M.mtp_targets(h, ids)
    assert h_in.shape[1] == 10 and e_in.shape[1] == 10 and tgt.shape[1] == 10
    # position 0 predicts ids[2]; input embedding is ids[1]
    assert tgt[0, 0].item() == 2 and e_in[0, 0].item() == 1
    assert tgt[0, -1].item() == 11 and e_in[0, -1].item() == 10


def test_mtp_targets_rejects_short_sequences():
    with pytest.raises(ValueError):
        M.mtp_targets(torch.randn(1, 2, 8), torch.arange(2).reshape(1, 2))


def test_draft_acceptance_perfect_and_zero():
    B, T, V = 1, 6, 5
    main = torch.zeros(B, T, V)
    mtp = torch.zeros(B, T - 2, V)
    # main[:, t] predicts t+1; make main pick token (t+1)%V and draft match it
    for t in range(T):
        main[:, t, (t + 1) % V] = 10.0
    for t in range(T - 2):
        mtp[:, t, (t + 2) % V] = 10.0
    assert M.draft_acceptance(main, mtp) == pytest.approx(1.0)
    assert M.draft_acceptance(main, mtp * 0.0) != 1.0


def test_draft_acceptance_topk_counts_a_hit_in_the_top_k():
    B, T, V = 1, 5, 6
    main = torch.zeros(B, T, V)
    mtp = torch.zeros(B, T - 2, V)
    for t in range(T):
        main[:, t, (t + 1) % V] = 10.0
    # draft has the right token at rank 2 everywhere
    for t in range(T - 2):
        mtp[:, t, (t + 2) % V] = 5.0
        mtp[:, t, (t + 3) % V] = 10.0          # wrong token wins
    assert M.draft_acceptance(main, mtp, topk=1) == pytest.approx(0.0)
    assert M.draft_acceptance(main, mtp, topk=2) == pytest.approx(1.0)


def test_mtp_logits_reuses_the_models_norm_and_lm_head():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.Identity()
            self.lm_head = torch.nn.Linear(16, 7, bias=False)

    model = FakeModel()
    head = M.MTPHead(hidden=16)
    logits = M.mtp_logits(model, head, torch.randn(1, 4, 16), torch.randn(1, 4, 16))
    assert logits.shape == (1, 4, 7)


def test_chunked_ce_matches_unchunked():
    torch.manual_seed(1)
    logits = torch.randn(1, 70, 11)
    targets = torch.randint(0, 11, (1, 70))
    full = torch.nn.functional.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
    chunked = M.chunked_ce(logits, targets, chunk=16)
    assert torch.allclose(full, chunked, atol=1e-5)

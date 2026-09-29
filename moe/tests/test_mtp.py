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


def test_chunked_kl_matches_unchunked_and_scales_with_temp():
    torch.manual_seed(2)
    logits = torch.randn(1, 70, 11)
    ref = torch.randn(1, 70, 11)
    logp = torch.log_softmax(logits.reshape(-1, 11), dim=-1)
    logr = torch.log_softmax(ref.reshape(-1, 11), dim=-1)
    full = torch.nn.functional.kl_div(logp, logr, log_target=True, reduction="batchmean")
    chunked = M.chunked_kl(logits, ref, temp=1.0, chunk=16)
    assert torch.allclose(full, chunked, atol=1e-5)
    # temperature convention: KL with both sides scaled by T, times T^2
    t = 2.0
    logp_t = torch.log_softmax(logits.reshape(-1, 11) / t, dim=-1)
    logr_t = torch.log_softmax(ref.reshape(-1, 11) / t, dim=-1)
    full_t = torch.nn.functional.kl_div(logp_t, logr_t, log_target=True,
                                        reduction="batchmean") * t * t
    assert torch.allclose(full_t, M.chunked_kl(logits, ref, temp=t, chunk=16), atol=1e-5)
    # and it differentiates
    l = M.chunked_kl(logits.requires_grad_(), ref, temp=2.0)
    l.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_self_target_main_is_the_draft_position_target_and_detached():
    B, T, V = 1, 6, 5
    main = torch.zeros(B, T, V)
    for t in range(T):
        main[:, t, (t + 1) % V] = 10.0
    main.requires_grad_(True)
    tgt = M.self_target_main(main)
    assert tgt.shape == (B, T - 2)
    # position 0 of the draft must match the main's token for t+2
    assert tgt[0, 0].item() == 2 % V
    # the target is the frozen body's opinion: no graph
    assert not tgt.requires_grad


def test_head_two_layers_shape_and_gradients():
    torch.manual_seed(3)
    head = M.MTPHead(hidden=16, layers=2)
    out = head(torch.randn(1, 5, 16), torch.randn(1, 5, 16))
    assert out.shape == (1, 5, 16)
    out.sum().backward()
    assert head.fc1.weight.grad is not None and head.fc2.weight.grad is not None
    with pytest.raises(ValueError):
        M.MTPHead(hidden=16, layers=3)
    # checkpoint compatibility: the v1 head keeps the fc.weight name
    assert "fc.weight" in M.MTPHead(hidden=16).state_dict()


def test_freeze_except_head_leaves_only_the_head_trainable():
    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

    model = Body()
    head = M.MTPHead(hidden=8)
    model._mtp_head = head
    n = M.freeze_except_head(model, head)
    assert n == sum(p.numel() for p in head.parameters())
    assert all(not p.requires_grad for p in model.proj.parameters())
    assert all(p.requires_grad for p in head.parameters())
    # gradients flow to the head only
    out = head(torch.randn(1, 3, 8), torch.randn(1, 3, 8))
    out.sum().backward()
    assert head.fc.weight.grad is not None
    assert model.proj.weight.grad is None


def test_mtp_drafter_flags_default_to_the_frozen_v1_behavior():
    import qwen35_moe_proxy as proxy
    args = proxy.build_parser().parse_args(["train"])
    assert args.mtp_weight == 0.0
    assert args.mtp_target == "corpus"
    assert args.mtp_self_temp == 2.0
    assert args.mtp_only is False
    assert args.mtp_head_layers == 1

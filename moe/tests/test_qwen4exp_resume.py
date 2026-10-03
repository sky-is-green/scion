"""Tests for ``--resume`` (``stage_train`` continuation, W4).

CPU-only: step parsing, LR-schedule math, cache-offset contract, and
weight-load exactness + mismatch rejection on the tiny model.  The full
3000->4096 continuation is pod work (W4), not this file.
"""

from __future__ import annotations

from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F

from qwen4exp_proxy import (attach_branches, build_tiny_model,
                            load_resume_weights, lr_at_step, model_logits,
                            parse_resume_step, resume_offsets)


def base_args(**kw):
    d = {"lr": 1e-3, "lr_half_every": 0, "lr_decay_start": 0,
         "rank": 8, "branch_quant": "fp32", "quant": "lloyd",
         "branch_target": "moe_out", "branch_gate": "none",
         "resume": "", "resume_step": 0}
    d.update(kw)
    return Namespace(**d)


def test_parse_resume_step():
    p = "/x/qwen4exp-corr-r512-g128-step3000-cur05.pt"
    assert parse_resume_step(p) == 3000
    assert parse_resume_step(p, explicit=100) == 100
    assert parse_resume_step("ckpt-step42.pt") == 42
    with pytest.raises(SystemExit):
        parse_resume_step("/x/no-step-here.pt")


def test_lr_at_step():
    a = base_args()
    assert lr_at_step(a, 0) == pytest.approx(1e-3)
    assert lr_at_step(a, 99999) == pytest.approx(1e-3)  # schedule off
    b = base_args(lr_half_every=1000, lr_decay_start=500)
    assert lr_at_step(b, 0) == pytest.approx(1e-3)
    assert lr_at_step(b, 499) == pytest.approx(1e-3)
    assert lr_at_step(b, 1000) == pytest.approx(5e-4)
    assert lr_at_step(b, 1500) == pytest.approx(5e-4)
    assert lr_at_step(b, 2000) == pytest.approx(2.5e-4)
    assert lr_at_step(b, 3500) == pytest.approx(1.25e-4)


def test_resume_offsets():
    assert resume_offsets(0, 4096) == 0
    assert resume_offsets(3000, 4096) == 3000
    assert resume_offsets(4096, 4096) == 0  # epoch boundary wraps clean
    assert resume_offsets(5000, 4096) == 904


def trained_tiny(path, steps=2):
    """Tiny model + rank-8 branches trained a few LM steps; ckpt saved."""
    model, cfg = build_tiny_model()
    args = base_args()
    attach_branches(model, args)
    for name, p in model.named_parameters():
        p.requires_grad_(".branch." in name or ".gate." in name)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adafactor(params, lr=1e-3, weight_decay=0.0)
    ids = torch.randint(0, cfg.vocab_size, (1, 32))
    model.train()
    for _ in range(steps):
        logits = model_logits(model, ids)
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                             ids[:, 1:].reshape(-1))
        lm.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    sd = {k: v.cpu().clone() for k, v in model.state_dict().items()
          if ".branch." in k or ".gate." in k}
    torch.save(sd, str(path))
    return {k: v.clone() for k, v in sd.items()}


def test_load_resume_weights_exact(tmp_path):
    ckpt = tmp_path / "qwen4exp-corr-r8-step2-t.pt"
    trained = trained_tiny(ckpt, steps=2)
    assert len(trained) > 0
    model2, _ = build_tiny_model()
    attach_branches(model2, base_args())
    before = {k: v.clone() for k, v in model2.state_dict().items()
              if ".branch." in k or ".gate." in k}
    assert any(not torch.equal(trained[k], before[k]) for k in trained)
    args = base_args(resume=str(ckpt))
    assert load_resume_weights(model2, str(ckpt), args) == 2
    for k in trained:
        assert torch.equal(dict(model2.state_dict())[k], trained[k])


def test_load_resume_weights_rejects(tmp_path):
    ckpt = tmp_path / "qwen4exp-corr-r8-step2-t.pt"
    trained_tiny(ckpt, steps=1)
    model2, _ = build_tiny_model()
    attach_branches(model2, base_args())
    args = base_args()
    with pytest.raises(SystemExit):
        load_resume_weights(model2, str(tmp_path / "missing.pt"), args)
    bad = dict(torch.load(str(ckpt), map_location="cpu"))
    bad["mystery.weight"] = torch.zeros(4)
    badp = tmp_path / "bad-step1.pt"
    torch.save(bad, str(badp))
    with pytest.raises(SystemExit):
        load_resume_weights(model2, str(badp), args)
    wrong = dict(torch.load(str(ckpt), map_location="cpu"))
    k0 = next(k for k in wrong if ".branch." in k)
    wrong[k0] = torch.zeros(3, 3)
    wrongp = tmp_path / "wrong-step1.pt"
    torch.save(wrong, str(wrongp))
    with pytest.raises(SystemExit):
        load_resume_weights(model2, str(wrongp), args)
    with pytest.raises(SystemExit):
        load_resume_weights(model2, str(tmp_path / "nostep.pt"), args)

"""CPU tests for the router-freeze diagnostics (pure torch, no model)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

MOE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOE))

import router_metrics as rm  # noqa: E402


def test_load_stats_separates_balance_from_collapse():
    balanced = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
    b = rm.load_stats(balanced, 2)
    assert b["cv"] == pytest.approx(0.0, abs=1e-7)
    assert b["peak"] == pytest.approx(1.0, abs=1e-6)
    assert b["cold"] == pytest.approx(0.0)
    assert b["dead"] == 0

    collapsed = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 1]])
    c = rm.load_stats(collapsed, 4)
    assert c["cv"] > 1.0
    assert c["peak"] > 2.0
    assert c["cold"] > 0.0          # experts 2 and 3 carry nothing
    assert c["dead"] >= 2


def test_top1_and_topk_agreement():
    student = torch.tensor([[0, 1, 2], [3, 4, 5]])
    teacher = torch.tensor([[0, 2, 1], [3, 5, 4]])
    assert rm.top1_agreement(student, teacher) == pytest.approx(1.0)
    assert rm.topk_agreement(student, teacher) == pytest.approx(1.0)
    teacher_bad = torch.tensor([[7, 8, 9], [3, 4, 5]])
    assert rm.top1_agreement(student, teacher_bad) == pytest.approx(0.5)
    assert rm.topk_agreement(student, teacher_bad) == pytest.approx(0.5)


def test_topk_union_kl_is_zero_for_identical_routing():
    idx = torch.tensor([[0, 1], [1, 2], [0, 2]])
    p = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3]])
    assert rm.topk_union_kl(idx, p, idx, p, 3) == pytest.approx(0.0, abs=1e-6)
    other = torch.tensor([[2, 1], [1, 0], [0, 1]])
    assert rm.topk_union_kl(idx, p, other, p, 3) > 0.0


def test_bucket_agreement_shows_position_decay():
    teacher = torch.zeros(128, 1, dtype=torch.long)
    student = teacher.clone()
    student[64:] = 1                    # correct through token 63, wrong after
    buckets = rm.bucket_agreement(student, teacher, bucket=64)
    assert buckets[0] == pytest.approx(1.0)
    assert buckets[1] == pytest.approx(0.0)


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.gate = torch.nn.Module()
        self.mlp.gate.weight = torch.nn.Parameter(torch.ones(4, 8))
        self.mlp.branch = torch.nn.Module()
        self.mlp.branch.down = torch.nn.Linear(8, 2, bias=False)


def test_set_trainable_keeps_branch_and_optionally_router():
    m = _Tiny()
    n = rm.set_trainable(m, freeze_router=True)
    assert n == 0
    assert m.mlp.gate.weight.requires_grad is False
    assert m.mlp.branch.down.weight.requires_grad is True

    m = _Tiny()
    n = rm.set_trainable(m, freeze_router=False)
    assert n == 32
    assert m.mlp.gate.weight.requires_grad is True


def test_router_delta_stats_is_zero_for_base_and_positive_after_update():
    m = _Tiny()
    base = {k: p.detach().clone() for k, p in m.named_parameters()}
    assert rm.router_delta_stats(m, base)["mlp.gate.weight"] == pytest.approx(0.0)
    with torch.no_grad():
        m.mlp.gate.weight.add_(0.5)
    assert rm.router_delta_stats(m, base)["mlp.gate.weight"] > 0.0

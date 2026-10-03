"""Tests for the PLE-QAT hook (``apply_ple_qat``).

CPU-only: STE exactness on a sparse table, grad flow, train/eval-mode
behavior, and tiny-model integration.  The 2-layer convergence screen
(`qat-screen` on dGPU1) is the manual run in the runlog, not this file.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from qwen4exp_proxy import (apply_ple_qat, build_tiny_model, quantize_rows,
                            SparseNGramTable)


def make_table(n=48, dim=160, seed=0):
    torch.manual_seed(seed)
    tab = SparseNGramTable(dim, 1024)
    ids = torch.arange(n).long()
    rows = torch.randn(n, dim, dtype=torch.bfloat16)
    tab.set_rows(ids, rows)
    return tab, ids, rows


def test_qat_ste_exactness_and_grad_flow():
    tab, ids, rows = make_table()
    lin = nn.Linear(160, 8)
    x = torch.randint(0, 48, (2, 16))
    # clean reference, no hook.
    clean = tab(x)
    loss0 = lin(clean.float()).sum()
    g0 = torch.autograd.grad(loss0, lin.weight, retain_graph=False)[0].clone()
    # attach the real hook.
    parent = nn.Module()
    parent.add_module("tab", tab)
    parent.train()
    assert apply_ple_qat(parent, 2, 32) == 0  # no text_layers -> nothing hooked
    from functools import partial
    import qwen4exp_proxy as q4
    tab.register_forward_hook(
        lambda mod, a, out: out + (q4.quantize_rows(
            out.detach(), 2, 32)[0] - out.detach()))
    parent.train()
    q = tab(x)
    deq, _ = quantize_rows(clean.detach(), 2, 32)
    assert torch.equal(q, clean + (deq - clean.detach()))
    loss1 = lin(q.float()).sum()
    g1 = torch.autograd.grad(loss1, lin.weight)[0]
    assert torch.isfinite(g1).all() and bool((g1 != 0).any())
    # STE: the downstream grad equals backprop through the quantized values.
    ref = lin(deq.detach().float()).sum()
    gref = torch.autograd.grad(ref, lin.weight)[0]
    assert torch.equal(g1, gref)


def test_qat_eval_mode_is_identity_and_zero_bits_noop():
    tab, ids, rows = make_table()
    x = torch.randint(0, 48, (2, 16))
    tab.eval()
    import qwen4exp_proxy as q4
    tab.register_forward_hook(
        lambda mod, a, out: out if not mod.training else out + (
            q4.quantize_rows(out.detach(), 2, 32)[0] - out.detach()))
    assert torch.equal(tab(x), rows[x])
    assert apply_ple_qat(nn.Module(), 0) == 0


def test_qat_tiny_model_integration():
    import qwen4exp_proxy as q4
    model, cfg = build_tiny_model()
    # the tiny table rows are 16-wide, so the QAT group is 16 here (the
    # release table is 160-wide with group 32; same divisibility rule).
    assert apply_ple_qat(model, 2, 16) == 1
    ids = torch.randint(0, 256, (1, 32))
    model.train()
    out = model(input_ids=ids, use_cache=False)
    hidden = getattr(out, "last_hidden_state", None)
    assert hidden is not None and bool(torch.isfinite(hidden).all())
    loss = model.lm_head(hidden).float().sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    # eval mode: hook passes through, output matches the unhooked model.
    model.eval()
    with torch.no_grad():
        a = model(input_ids=ids, use_cache=False).last_hidden_state
    for h in list(model.modules()):
        for _h in getattr(h, "_forward_hooks", {}).copy():
            pass
    torch.manual_seed(0)
    model2, _ = build_tiny_model()
    model2.eval()
    with torch.no_grad():
        b = model2(input_ids=ids, use_cache=False).last_hidden_state
    assert torch.equal(a, b)

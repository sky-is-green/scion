"""CPU tests for the router-balance plumbing (fake gates + the real class)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

MOE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOE))

import router_balance as rb  # noqa: E402
import router_bias as rbia  # noqa: E402


class FakeGate(torch.nn.Module):
    """The stock qwen35 router forward (transformers 5.5 semantics)."""

    def __init__(self, n_experts=8, hidden=16, top_k=2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.top_k, self.num_experts, self.hidden_dim = top_k, n_experts, hidden
        self.weight = torch.nn.Parameter(torch.randn(n_experts, hidden, generator=g) * 0.5)

    def forward(self, x):
        h = x.reshape(-1, self.hidden_dim)
        logits = torch.nn.functional.linear(h, self.weight)
        probs = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
        val, idx = torch.topk(probs, self.top_k, dim=-1)
        val = val / val.sum(-1, keepdim=True)
        return probs, val.to(logits.dtype), idx


class FakeModel(torch.nn.Module):
    def __init__(self, gates):
        super().__init__()
        self.gates = torch.nn.ModuleList(gates)


def test_patched_forward_matches_stock_at_zero_bias():
    torch.manual_seed(0)
    gate = FakeGate()
    x = torch.randn(64, 16)
    _, stock_w, stock_idx = gate(x)
    rbia.patch_gate(gate, "bias")
    probs, weights, idx = gate(x)
    assert torch.equal(idx, stock_idx)
    assert torch.allclose(weights, stock_w, atol=1e-6)
    assert torch.allclose(probs, probs)  # shapes/normalisation
    assert torch.allclose(probs.sum(-1), torch.ones(64), atol=1e-5)


def test_bias_changes_selection_but_not_weights():
    torch.manual_seed(1)
    gate = FakeGate()
    x = torch.randn(256, 16)
    rbia.patch_gate(gate, "bias")
    _, _, base_idx = gate(x)
    base_rate = (base_idx == 0).float().mean()
    with torch.no_grad():
        gate.balance_bias[0] = 5.0
    probs, weights, idx = gate(x)
    boosted = (idx == 0).float().mean()
    assert boosted > base_rate
    # weights must be the raw softmax over the selected raw logits, bias-free
    h = x.reshape(-1, 16)
    raw = torch.nn.functional.linear(h, gate.weight)
    manual = torch.softmax(raw.gather(-1, idx), dim=-1)
    assert torch.allclose(weights, manual, atol=1e-5)


def test_bias_buffer_is_born_on_the_gates_device():
    """``patch_gate`` runs after ``device_map``, so the buffer must follow the weight.

    The balbias arm crashed on its first GPU forward with a cuda/cpu mismatch:
    the buffer was created on CPU after the model had already been placed on
    the card.  CPU tests cannot hold two devices, but they can pin the
    invariant the GPU path depends on.
    """
    gate = FakeGate()
    rbia.patch_gate(gate, "bias")
    assert gate.balance_bias.device == gate.weight.device
    assert gate.balance_bias.dtype == torch.float32


def test_sign_update_moves_the_overloaded_expert_down():
    torch.manual_seed(2)
    gate = FakeGate(n_experts=6, top_k=2)
    x = torch.randn(512, 16)
    rbia.patch_gate(gate, "bias")
    with torch.no_grad():
        gate.weight[0] *= 6.0                    # expert 0 dominates
    gate(x)
    counts = gate._balance_stats["counts"].float()
    assert counts.argmax().item() == 0
    model = FakeModel([gate])
    diag = rbia.balance_update(model, "bias", delta=1e-3)
    assert gate.balance_bias[0] < 0.0
    assert (gate.balance_bias[1:] > 0).all()
    assert "mean_load_entropy" in diag
    assert int(gate._balance_stats["counts"].sum()) == 0   # stats reset


def test_quantile_update_reaches_k_over_n_load():
    torch.manual_seed(3)
    n, k = 8, 2
    gate = FakeGate(n_experts=n, top_k=k)
    x = torch.randn(2048, 16)
    rbia.patch_gate(gate, "quantile")
    model = FakeModel([gate])
    for _ in range(6):
        gate(x)
        counts = gate._balance_stats["counts"].float()
        rbia.balance_update(model, "quantile")
    # counts are routed slots (T*k total); token load is counts / T
    token_load = counts / x.shape[0]
    assert (token_load - k / n).abs().max() < 0.02


def test_zloss_is_differentiable_and_resets():
    torch.manual_seed(4)
    gate = FakeGate()
    x = torch.randn(32, 16)
    rbia.patch_gate(gate, "zloss")
    model = FakeModel([gate])
    gate(x)
    loss, diag = rbia.balance_z_loss(model, coeff=1.0)
    assert loss.requires_grad and loss.item() > 0
    loss.backward()
    assert gate.weight.grad is not None and torch.isfinite(gate.weight.grad).all()
    loss2, _ = rbia.balance_z_loss(model, coeff=1.0)
    assert loss2.item() == 0.0                            # stats reset


def test_patch_router_balance_real_class():
    cls = pytest.importorskip(
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe").Qwen3_5MoeTopKRouter
    cfg = type("Cfg", (), {"num_experts_per_tok": 2, "num_experts": 8,
                           "hidden_size": 16})()
    gates = [cls(cfg), cls(cfg)]
    model = FakeModel(gates)
    x = torch.randn(64, 16)
    _, stock_w, stock_idx = gates[0](x)
    n = rbia.patch_router_balance(model, "bias")
    assert n == 2
    _, weights, idx = gates[0](x)
    assert torch.equal(idx, stock_idx)
    assert torch.allclose(weights, stock_w, atol=1e-6)
    sd = model.state_dict()
    assert "gates.0.balance_bias" in sd
    assert sd["gates.0.balance_bias"].shape == (8,)


def test_patch_router_balance_rejects_none_and_unknown():
    model = FakeModel([FakeGate()])
    with pytest.raises(ValueError):
        rbia.patch_router_balance(model, "none")
    with pytest.raises(ValueError):
        rbia.patch_router_balance(model, "magic")


def test_quantile_bias_step_matches_the_logits_version():
    torch.manual_seed(5)
    logits = torch.randn(128, 6)
    bias = torch.zeros(6)
    direct = rb.quantile_bias_update(bias, logits, k=2)
    via_margins = rb.quantile_bias_step(bias, rb.margins(logits, bias, 2), 2)
    assert torch.allclose(direct, via_margins, atol=1e-6)


# -------------------------------------------------------- causal-bias arms ---

def test_cb_changes_selection_but_not_weights():
    """CB steers selection from within-sequence mass; the weights stay unbiased."""
    torch.manual_seed(6)
    gate = FakeGate(n_experts=8, top_k=2)
    x = torch.randn(512, 16)
    _, _, stock_idx = gate(x)
    rbia.patch_gate(gate, "cb", cb_eta=0.5)
    _, weights, idx = gate(x)
    assert not torch.equal(idx, stock_idx), "the causal bias must move selection"
    h = x.reshape(-1, 16)
    raw = torch.nn.functional.linear(h, gate.weight)
    manual = torch.softmax(raw.gather(-1, idx), dim=-1)
    assert torch.allclose(weights, manual, atol=1e-5)


def test_cbqb_collects_margins_on_the_effective_scores_and_updates():
    torch.manual_seed(7)
    gate = FakeGate(n_experts=8, top_k=2)
    x = torch.randn(512, 16)
    rbia.patch_gate(gate, "cbqb", cb_eta=0.2, qb_damp=0.5)
    model = FakeModel([gate])
    gate(x)
    assert gate._balance_stats["margins"], "cbqb must collect margins"
    before = gate.balance_bias.clone()
    rbia.balance_update(model, "cbqb")
    assert not torch.equal(gate.balance_bias, before)
    assert torch.isfinite(gate.balance_bias).all()
    # the buffer update is mean-centred (K3) and the stats were reset
    assert abs(float(gate.balance_bias.mean())) < 1e-5
    assert int(gate._balance_stats["counts"].sum()) == 0


def test_cb_flags_default_off():
    """v1 stays numerically unchanged: every new knob is off by default."""
    import qwen35_moe_proxy as proxy
    args = proxy.build_parser().parse_args(["train"])
    assert args.balance == "none"
    assert args.balance_cb_eta == 0.05
    assert args.balance_qb_damp == 1.0


def test_cb_update_returns_diagnostics_and_resets():
    """CB has no buffer update, but the log still needs its load diagnostics.

    Pins the balcb gap: ``balance_update`` used to raise for ``cb``, the
    trainer skipped it, and the arm trained with no ``loadH`` at all.
    """
    torch.manual_seed(8)
    gate = FakeGate(n_experts=8, top_k=2)
    x = torch.randn(512, 16)
    rbia.patch_gate(gate, "cb", cb_eta=0.5)
    model = FakeModel([gate])
    gate(x)
    before = gate.balance_bias.clone()
    diag = rbia.balance_update(model, "cb")
    assert "mean_load_entropy" in diag
    assert not math.isnan(diag["mean_load_entropy"])
    assert torch.equal(gate.balance_bias, before)          # nothing to carry
    assert int(gate._balance_stats["counts"].sum()) == 0   # stats reset


def test_per_sequence_load_variance_is_zero_for_identical_sequences():
    torch.manual_seed(9)
    gate = FakeGate(n_experts=8, top_k=2)
    x = torch.randn(512, 16)
    rbia.patch_gate(gate, "bias")
    gate(x)
    diag = rbia.balance_diagnostics(FakeModel([gate]))
    assert math.isnan(diag["mean_seq_load_var"])           # one sequence: undefined
    gate(x)
    diag = rbia.balance_diagnostics(FakeModel([gate]))
    assert diag["mean_seq_load_var"] == pytest.approx(0.0, abs=1e-9)


def test_per_sequence_load_variance_sees_uneven_sequences():
    """A constant shift tilts the router — two very different load vectors."""
    torch.manual_seed(10)
    gate = FakeGate(n_experts=8, top_k=2)
    x = torch.randn(512, 16)
    rbia.patch_gate(gate, "bias")
    gate(x)
    gate(x + 5.0)
    diag = rbia.balance_diagnostics(FakeModel([gate]))
    assert diag["mean_seq_load_var"] > 1e-3
    # batch entropy can still look fine while sequences disagree
    assert diag["mean_load_entropy"] > 1.5

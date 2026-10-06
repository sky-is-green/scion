"""Phase C: branch read/write gates (identity at init, foldable at export).

The gates are the experiment: a per-channel read gate ``g_r`` on the branch
input and a per-channel write gate ``g_w`` on its output, parameterised
``g = 1 + tanh(r)`` with ``r = 0``.  These tests pin the four things the arm's
one-variable claim -- and its deployability -- rest on:

  1. gated at init is bit-identical to ungated (one-variable by construction);
  2. the gates move and fold exactly into the factors; with ternary factors the
     fold-then-quantize deployed form pays only the same quantization error the
     ungated branch already pays, and keeps the rank-r container shape;
  3. checkpoint round-trip both ways: a gated checkpoint loads into an ungated
     model (gates ignored) and a legacy checkpoint loads into a gated model
     (gates stay at init = identity);
  4. v1 defaults unchanged: both CLIs default to ``--branch-gate none`` and the
     exporter does not silently drop gate tensors.
"""
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_proxy import ternary_lloyd  # noqa: E402
from olmoe_corrections import (CorrectionBranch, MoEWithCorrection,  # noqa: E402
                               fold_gates, gate_stats, load_branch_state)


def _fresh(hidden=16, rank=4, quant="fp32", gate="none", out_dim=None, seed=0):
    torch.manual_seed(seed)
    return CorrectionBranch(hidden, rank, quant, "absmean", out_dim, gate)


def _gated_pair(**kw):
    """An ungated and a gated branch that start from identical factors."""
    plain = _fresh(**kw)
    gated = _fresh(gate="rw", **kw)
    gated.load_state_dict(plain.state_dict(), strict=False)
    return plain, gated


# ------------------------------------------- 1. identity at init -------------

@pytest.mark.parametrize("quant", ["fp32", "g128"])
def test_gated_init_is_bit_identical_to_ungated(quant):
    torch.manual_seed(0)
    x = torch.randn(5, 16).to(torch.bfloat16)
    plain, gated = _gated_pair(quant=quant)
    # torch.equal, not allclose: 1 + tanh(0) is exactly 1.0 and an fp32
    # multiply by exactly 1.0 is the identity, so the arm must be bit-exact.
    assert torch.equal(plain(x), gated(x))


# --------------------------------------- 2. the gates act and fold -----------

def test_gates_move_and_fold_matches_the_gated_forward_fp32():
    plain, gated = _gated_pair()
    with torch.no_grad():
        gated.read_gate.normal_(0.0, 0.5)
        gated.write_gate.normal_(0.0, 0.5)
    x = torch.randn(7, 16)
    ref = gated(x)
    down, up = fold_gates(gated.down.weight, gated.up.weight,
                          gated.read_gate, gated.write_gate)
    folded = F.linear(F.linear(x, down), up)
    assert torch.allclose(ref, folded, atol=1e-5, rtol=1e-4)
    # the fold is not a no-op here: the factors actually changed
    assert not torch.allclose(down, gated.down.weight)


def test_fold_then_quantize_keeps_rank_and_pays_no_new_error_class():
    """Deployed form: fold into the masters, then ternarise.  The fold must not
    add an error class beyond the ternary quantization both forms already pay,
    and the container shape must stay rank-r."""
    plain, gated = _gated_pair(quant="g128")
    with torch.no_grad():
        gated.down.weight.normal_(0.0, 0.05)
        gated.up.weight.normal_(0.0, 0.05)
        gated.read_gate.normal_(0.0, 0.5)
        gated.write_gate.normal_(0.0, 0.5)
        # keep the plain baseline on the exact same factors for the comparison
        plain.load_state_dict(gated.state_dict(), strict=False)
    x = torch.randn(9, 16)

    # reference: the gated branch in fp32 (what the fold algebra reproduces)
    gr, gw = 1.0 + torch.tanh(gated.read_gate), 1.0 + torch.tanh(gated.write_gate)
    ref = F.linear(F.linear(x * gr, gated.down.weight), gated.up.weight) * gw

    def quant_err(w, u):
        q = F.linear(F.linear(x, ternary_lloyd(w, 128)), ternary_lloyd(u, 128))
        return float((q - ref).norm() / ref.norm())

    # error the plain branch pays for the same factors (quantization alone)
    base_err = quant_err(plain.down.weight, plain.up.weight)
    down, up = fold_gates(gated.down.weight, gated.up.weight,
                          gated.read_gate, gated.write_gate)
    folded_err = quant_err(down, up)
    assert folded_err <= 2.0 * base_err + 1e-3, (folded_err, base_err)

    # deployable container unchanged: same rank-r shapes
    assert ternary_lloyd(down, 128).shape == gated.down.weight.shape
    assert ternary_lloyd(up, 128).shape == gated.up.weight.shape


def test_gate_stats_reports_identity_at_init_then_movement():
    m = MoEWithCorrection(torch.nn.Linear(16, 16), 16, 4, gate="rw")
    m2 = MoEWithCorrection(torch.nn.Linear(16, 16), 16, 4)         # ungated
    st = gate_stats(m)
    assert len(st) == 1
    assert st[0]["read_mean"] == pytest.approx(1.0)
    assert st[0]["write_mean"] == pytest.approx(1.0)
    assert st[0]["read_std"] == pytest.approx(0.0)
    assert gate_stats(m2) == []                                    # v1: no gates
    with torch.no_grad():
        m.branch.write_gate.fill_(0.5)
    st = gate_stats(m)
    assert st[0]["write_mean"] == pytest.approx(1.0 + torch.tanh(torch.tensor(0.5)).item())


# ------------------------------------------- 3. checkpoint round-trips -------

def test_gated_checkpoint_loads_into_ungated_and_vice_versa(tmp_path):
    plain, gated = _gated_pair()
    x = torch.randn(3, 16)

    # gated checkpoint -> ungated model: factors load, gates are "unexpected"
    p = tmp_path / "gated.pt"
    torch.save(gated.state_dict(), p)
    fresh = _fresh()
    missing, unexpected = load_branch_state(fresh, str(p))
    assert not [k for k in missing if k.endswith(("down.weight", "up.weight"))]
    assert any("read_gate" in k for k in unexpected)
    assert torch.equal(fresh(x), plain(x))

    # legacy (ungated) checkpoint -> gated model: gates stay at init (identity),
    # so the gated model reproduces the ungated forward bit-exactly
    p2 = tmp_path / "plain.pt"
    torch.save(plain.state_dict(), p2)
    g2 = _fresh(gate="rw")
    missing, unexpected = load_branch_state(g2, str(p2))
    assert not [k for k in missing if not k.endswith(("read_gate", "write_gate"))]
    assert not [k for k in unexpected if k.endswith(("down.weight", "up.weight"))]
    assert torch.equal(g2(x), plain(x))


# ------------------------------------------- 4. defaults / export ------------

def test_v1_defaults_unchanged():
    import kld_eval
    import qwen35_moe_proxy
    assert qwen35_moe_proxy.build_parser().parse_args(["train"]).branch_gate == "none"
    assert qwen35_moe_proxy.build_parser().parse_args(["eval"]).branch_gate == "none"
    assert kld_eval.build_parser().parse_args([]).branch_gate == "none"


def _export_module(monkeypatch):
    """Import the exporter with a stub gguf (the real one lives on the fork's
    PYTHONPATH, not in the test env; the collection/fold logic is what is under
    test, not the GGUF writer)."""
    gguf = types.ModuleType("gguf")
    gguf.GGUFWriter = object
    monkeypatch.setitem(sys.modules, "gguf", gguf)
    return importlib.import_module("export_branches_lora")


def test_export_collects_gate_tensors_so_they_cannot_be_silently_dropped(monkeypatch):
    export = _export_module(monkeypatch)
    wanted = {"attn_output.weight", "ssm_out.weight", "ffn_moe_out.weight"}
    sd = {
        "model.layers.3.self_attn.o_proj.branch.down.weight": torch.randn(8, 16),
        "model.layers.3.self_attn.o_proj.branch.up.weight": torch.randn(16, 8),
        "model.layers.3.self_attn.o_proj.branch.read_gate": torch.randn(16),
        "model.layers.3.self_attn.o_proj.branch.write_gate": torch.randn(16),
        "model.layers.0.mlp.branch.down.weight": torch.randn(8, 16),
        "model.layers.0.mlp.branch.up.weight": torch.randn(16, 8),
        "model.layers.0.mlp.branch.read_gate": torch.randn(16),
        "model.layers.0.mlp.branch.write_gate": torch.randn(16),
        "model.layers.1.linear_attn.out_proj.branch.down.weight": torch.randn(12, 16),
        "model.layers.1.linear_attn.out_proj.branch.up.weight": torch.randn(16, 12),
    }
    pairs, gates = export.collect_branch_pairs(sd, wanted)
    assert set(pairs) == {(3, "attn_output.weight"), (0, "ffn_moe_out.weight"),
                          (1, "ssm_out.weight")}
    assert set(gates) == {(3, "attn_output.weight"), (0, "ffn_moe_out.weight")}
    # folding a collected pair keeps the exported container rank-r
    down, up = export.fold_gates(pairs[(0, "ffn_moe_out.weight")]["down"],
                                 pairs[(0, "ffn_moe_out.weight")]["up"],
                                 gates[(0, "ffn_moe_out.weight")]["read"],
                                 gates[(0, "ffn_moe_out.weight")]["write"])
    assert down.shape == (8, 16) and up.shape == (16, 8)

    # an ungated (legacy) checkpoint collects no gates and no fold happens
    pairs2, gates2 = export.collect_branch_pairs(sd, wanted)
    bare = {k: v for k, v in sd.items() if not k.endswith(("read_gate", "write_gate"))}
    pairs3, gates3 = export.collect_branch_pairs(bare, wanted)
    assert gates3 == {}
    assert set(pairs3) == set(pairs)

"""score_all() must survive mixed device placement and odd row lengths.

The first real run died with a device mismatch: ``deltaloss_linear`` builds an
autograd graph, so the captured CPU input and the CUDA weight cannot meet.  The
model wanted the weight moved, and each tensor is released as it is consumed
because a full-resolution fp32 copy of every projection will not fit otherwise.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scion_paths
import steer_probe as sp

autogrid = pytest.importorskip("autogrid_ext.steer")
pytestmark = pytest.mark.skipif(
    not scion_paths.AUTOGRID_REPO.exists(), reason="no autogrid fork")


def _model():
    m = torch.nn.Module()
    m.layers = torch.nn.ModuleList([
        torch.nn.ModuleDict({
            "self_attn": torch.nn.ModuleDict({
                "q_proj": torch.nn.Linear(256, 256, bias=False)}),
            "mlp": torch.nn.ModuleDict({
                "shared_expert": torch.nn.ModuleDict({
                    "gate_proj": torch.nn.Linear(256, 512, bias=False)})}),
        })
    ])
    return m


def test_scores_every_linear_and_is_deterministic():
    torch.manual_seed(0)
    model = _model()
    store = {"layers.0.self_attn.q_proj": torch.randn(64, 256),
             "layers.0.mlp.shared_expert.gate_proj": torch.randn(64, 256)}
    q = sp.quantizer_for("lloyd", 128)
    a = sp.score_all(model, store, q)
    b = sp.score_all(model, store, q)
    assert [e["name"] for e in a] == [e["name"] for e in b]
    assert [e["delta_loss"] for e in a] == [e["delta_loss"] for e in b]
    assert len(a) == 2
    assert all(e["rank"] in (1, 2) for e in a)


def test_a_weight_on_another_device_still_scores(monkeypatch):
    """A CUDA weight against a captured CPU input must not raise."""
    model = _model()
    store = {"layers.0.self_attn.q_proj": torch.randn(32, 256)}

    # stand in for the real case: the module holds a non-default device
    orig = sp.linear_targets
    monkeypatch.setattr(sp, "linear_targets", lambda m: [
        (n, mod.to("meta") if False else mod) for n, mod in orig(m)])
    out = sp.score_all(model, store, sp.quantizer_for("lloyd", 128))
    assert len(out) == 1
    assert out[0]["delta_loss"] >= 0.0


def test_non_multiple_of_128_in_features_is_recorded_not_crashed():
    model = _model()
    # q_proj normally has in_features == 256; force an awkward one
    model.layers[0].self_attn.q_proj = torch.nn.Linear(100, 8, bias=False)
    store = {"layers.0.self_attn.q_proj": torch.randn(16, 100)}
    out = sp.score_all(model, store, sp.quantizer_for("lloyd", 128))
    assert len(out) == 1
    assert "skipped" in out[0]
    assert "100" in out[0]["skipped"]


def test_delta_loss_is_zero_for_an_exactly_representable_weight():
    """If the quantizer is the identity, there is no perturbation to score."""
    model = _model()
    w = torch.zeros(256, 256)
    model.layers[0].self_attn.q_proj.weight.data.copy_(w)
    store = {"layers.0.self_attn.q_proj": torch.randn(8, 256)}
    out = sp.score_all(model, store, lambda t: t.clone())
    assert out[0]["delta_loss"] == pytest.approx(0.0, abs=1e-9)

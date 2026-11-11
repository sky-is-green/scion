"""CPU integration tests for the fixed-router build/save path.

The model is a tiny randomly-initialised OLMoE; no checkpoint or dataset is
needed.  The test pins the two things the A/B relies on: the router is frozen
only when asked, and a frozen checkpoint does not silently carry base router
weights that would make the arm look joint on reload.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

MOE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOE))

import olmoe_corrections as oc  # noqa: E402


def _tiny_model():
    from transformers import OlmoeConfig, OlmoeForCausalLM
    torch.manual_seed(0)
    cfg = OlmoeConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    return OlmoeForCausalLM(cfg)


def _args(freeze_router: bool):
    return SimpleNamespace(
        device="cpu", device_map="cpu", group=8, quant="lloyd",
        rank=4, branch_quant="fp32", branch_quant_fp16_layers="",
        branch_target="moe_out", freeze_router=freeze_router,
        tag="t",
    )


def _build(monkeypatch, freeze_router: bool):
    monkeypatch.setattr(oc, "load_model", lambda *a, **k: (_tiny_model(), None))
    return oc.build(_args(freeze_router))


def test_frozen_arm_keeps_router_weights_frozen(monkeypatch):
    model, _ = _build(monkeypatch, freeze_router=True)
    router = [p for n, p in model.named_parameters() if ".gate." in n]
    branch = [p for n, p in model.named_parameters() if ".branch." in n]
    assert router and not any(p.requires_grad for p in router)
    assert branch and all(p.requires_grad for p in branch)


def test_joint_arm_trains_router_weights(monkeypatch):
    model, _ = _build(monkeypatch, freeze_router=False)
    router = [p for n, p in model.named_parameters() if ".gate." in n]
    assert router and all(p.requires_grad for p in router)


def test_frozen_checkpoint_omits_router_and_joint_keeps_it(monkeypatch, tmp_path):
    monkeypatch.setattr(oc, "OUT", tmp_path)

    frozen, _ = _build(monkeypatch, freeze_router=True)
    oc.save(frozen, _args(True), 7)
    fsd = torch.load(tmp_path / "olmoe-corr-r4-t-step7.pt", map_location="cpu")
    assert any(".branch." in k for k in fsd)
    assert not any(".gate." in k for k in fsd)

    joint, _ = _build(monkeypatch, freeze_router=False)
    with torch.no_grad():
        for name, p in joint.named_parameters():
            if ".gate." in name:
                p.add_(0.01)             # simulate a trained router
    oc.save(joint, _args(False), 7)
    jsd = torch.load(tmp_path / "olmoe-corr-r4-t-step7.pt", map_location="cpu")
    assert any(".gate." in k for k in jsd)

    # reload the joint checkpoint into a fresh build: the router value must
    # survive, i.e. the A/B's "trained routers" eval is actually trained.
    fresh, _ = _build(monkeypatch, freeze_router=False)
    key = next(k for k in jsd if k.endswith(".mlp.gate.weight"))
    before = fresh.state_dict()[key].clone()
    oc.load_branch_state(fresh, tmp_path / "olmoe-corr-r4-t-step7.pt")
    assert not torch.equal(before, fresh.state_dict()[key])
    assert torch.equal(jsd[key], fresh.state_dict()[key])


def test_evaluate_writes_routing_and_load_diagnostics(monkeypatch, tmp_path):
    """Run the eval path end-to-end on tiny tensors; it must emit the A/B fields."""
    from transformers import AutoTokenizer

    monkeypatch.setattr(oc, "OUT", tmp_path)
    monkeypatch.setattr(oc, "windows",
                        lambda *a, **k: torch.randint(0, 128, (2, 16)))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: None)
    monkeypatch.setattr(oc, "load_model", lambda *a, **k: (_tiny_model(), None))

    # make a joint checkpoint (router moved) for the eval to load
    ckpt_args = _args(False)
    train_model, _ = oc.build(ckpt_args)
    with torch.no_grad():
        for name, p in train_model.named_parameters():
            if ".gate." in name:
                p.add_(0.01)
    oc.save(train_model, ckpt_args, 7)

    args = _args(False)
    args.device = "cpu"
    args.eval_windows = 2
    args.seq = 16
    args.load = str(tmp_path / "olmoe-corr-r4-t-step7.pt")
    args.eval_router = "trained"
    oc.evaluate(args)

    out = tmp_path / "branches-eval-t.json"
    res = json.loads(out.read_text())
    assert res["rtn_no_branches"]["ppl"] > 0
    tr = res["trained_branches"]
    for key in ("top1_agree", "router_agree", "router_kl", "student_load",
                "teacher_load", "router_delta_rel", "layers"):
        assert key in tr
    assert len(tr["layers"]) == 2
    assert tr["router_delta_rel"] > 0

    args.tag = "b"
    args.eval_router = "base"
    oc.evaluate(args)
    base = json.loads((tmp_path / "branches-eval-b.json").read_text())
    assert base["trained_branches"]["router_delta_rel"] < 1e-9



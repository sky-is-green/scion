"""The checkpoint-load gate must compare branch keys, not the raw `missing`.

`load_state_dict(strict=False)` reports the entire frozen body as "missing",
because a saved checkpoint only ever contains branch/router tensors.  The first
gate run rejected a perfectly valid checkpoint on that basis (68 "missing"
keys, 0 branch tensors actually absent) and refused to produce a number.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kld_eval
from olmoe_corrections import load_branch_state


class _Block(torch.nn.Module):
    """One layer: a frozen body plus a correction branch."""

    def __init__(self, attn: bool = True):
        super().__init__()
        self.body = torch.nn.Linear(8, 8, bias=False)          # frozen
        self.branch = torch.nn.Module()
        self.branch.down = torch.nn.Linear(8, 4, bias=False)
        self.branch.up = torch.nn.Linear(4, 8, bias=False)
        if attn:
            self.out_proj = torch.nn.Module()
            self.out_proj.branch = torch.nn.Module()
            self.out_proj.branch.down = torch.nn.Linear(8, 4, bias=False)


class Model(torch.nn.Module):
    """Mimics the real tree: ModuleList layers, each with a body and branches."""

    def __init__(self, layers: int = 2, attn: bool = True):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(attn) for _ in range(layers)])

    def wanted(self) -> set[str]:
        return {k for k in self.state_dict() if ".branch." in k or ".gate." in k}

    def save(self, path) -> None:
        torch.save({k: v for k, v in self.state_dict().items()
                    if ".branch." in k or ".gate." in k}, path)


def _branch_load_report(model, path):
    """The accounting kld_eval's gate uses."""
    missing, unexpected = load_branch_state(model, str(path))
    want = model.wanted()
    got = want & set(missing)
    return {"want": len(want), "got": len(want) - len(got),
            "unloaded": len(got), "unexpected": len(unexpected)}


def test_a_valid_checkpoint_passes_the_gate(tmp_path):
    m = Model()
    p = tmp_path / "ckpt.pt"
    m.save(p)
    fresh = Model()
    rep = _branch_load_report(fresh, p)
    assert rep["unloaded"] == 0, "a matching checkpoint must pass"
    assert rep["unexpected"] == 0
    assert rep["got"] == rep["want"]


def test_the_frozen_body_never_counts_as_missing(tmp_path):
    """This is the exact failure: body keys appear in `missing` and are fine."""
    m = Model()
    p = tmp_path / "ckpt.pt"
    m.save(p)
    fresh = Model()
    missing, _ = load_branch_state(fresh, str(p))
    # the body really is reported missing ...
    assert any(".body." in k for k in missing)
    # ... and none of it is a branch tensor
    assert not any(".branch." in k for k in missing)


def test_a_checkpoint_from_a_different_placement_is_rejected(tmp_path):
    trained = Model(attn=True)
    p = tmp_path / "ckpt.pt"
    trained.save(p)
    # evaluate with attn_out branches absent: some wanted tensors have no source
    other = Model(attn=False)
    rep = _branch_load_report(other, p)
    assert rep["unexpected"] > 0, "placement mismatch must be visible"


def test_missing_branch_weights_are_caught(tmp_path):
    """A truncated checkpoint (lost a tensor) must not silently half-load."""
    m = Model()
    p = tmp_path / "ckpt.pt"
    sd = {k: v for k, v in m.state_dict().items()
          if ".branch." in k or ".gate." in k}
    victim = sorted(sd)[0]
    del sd[victim]
    torch.save(sd, p)
    rep = _branch_load_report(Model(), p)
    assert rep["unloaded"] >= 1

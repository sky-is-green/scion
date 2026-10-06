"""save() must not let two arms clobber each other's checkpoint.

It fired for real: arm A (top-50) and arm C (top-512) both resolved to
``qwen35-corr-r512-g128-step4096.pt`` and arm C was about to overwrite arm A's
final weights.  The stop rule already says "never change an arm's flags
mid-run", and the arm identity has to reach the filename or that rule is
unenforceable.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import qwen35_moe_proxy as proxy


@pytest.fixture
def fake_out(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy, "OUT", tmp_path)
    return tmp_path


class Tiny(torch.nn.Module):
    """Nested so the state-dict keys look like the real ones (``.branch.``)."""

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential()
        self.mlp.branch = torch.nn.Linear(2, 2, bias=False)
        self.mlp.frozen = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.ones_(self.mlp.branch.weight)


def _args(**kw):
    base = dict(rank=512, branch_quant="g128", tag="")
    base.update(kw)
    return SimpleNamespace(**base)


def test_tag_disambiguates_two_arms(fake_out):
    proxy.save(Tiny(), _args(tag="armA"), 4096)
    proxy.save(Tiny(), _args(tag="armC"), 4096)
    names = sorted(p.name for p in fake_out.glob("*.pt"))
    assert names == ["qwen35-corr-r512-g128-step4096-armA.pt",
                     "qwen35-corr-r512-g128-step4096-armC.pt"]


def test_untagged_name_is_unchanged(fake_out):
    """The default must stay byte-identical, so existing v1 runs still resolve."""
    proxy.save(Tiny(), _args(), 16)
    assert (fake_out / "qwen35-corr-r512-g128-step16.pt").exists()


def test_fp32_branch_keeps_the_legacy_no_tag_form(fake_out):
    proxy.save(Tiny(), _args(branch_quant="fp32"), 8)
    assert (fake_out / "qwen35-corr-r512-step8.pt").exists()


def test_only_branch_and_gate_tensors_are_saved(fake_out):
    proxy.save(Tiny(), _args(tag="armA"), 1)
    sd = torch.load(fake_out / "qwen35-corr-r512-g128-step1-armA.pt",
                    map_location="cpu", weights_only=False)
    assert list(sd) == ["mlp.branch.weight"]

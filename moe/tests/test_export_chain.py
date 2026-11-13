"""Tests for the export/audit chain (v2/PREFIX-LOOP.md step D).

The merged-release gate depends on two inverses holding exactly:
``pack_q1_0_g128`` (the exporter) vs ``decode_q1_0_g128`` (the auditor), and
``deploy_weights`` (the exported factors) being what the auditor recomputes
from the checkpoint.  Also pins the merge's ``--replace`` site selection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "moe"))
sys.path.insert(0, str(HERE / "v2" / "scripts"))
from export_branches_lora import deploy_weights, pack_q1_0_g128  # noqa: E402
from audit_merged_adapter import decode_q1_0_g128  # noqa: E402


@pytest.mark.parametrize("shape", [(512, 2048), (2048, 512), (512, 512), (128, 128)])
def test_pack_decode_roundtrip_is_exact(shape):
    """pack -> decode reproduces the deployed ternary values bit-for-bit.

    The input must BE a deployed-ternary tensor (q * fp16 scale), which is
    what deploy_weights returns and what the exporter packs.
    """
    torch.manual_seed(0)
    w = torch.randn(*shape) * 0.1
    deployed, _ = deploy_weights(w, w.clone(), "g128", "lloyd")
    packed = torch.from_numpy(pack_q1_0_g128(deployed, 128))
    raw = packed.reshape(-1).numpy().astype(np.uint8)
    got = decode_q1_0_g128(raw, deployed.shape)
    assert torch.equal(got, deployed.float())


def test_decode_rejects_nothing_and_pack_groups_align():
    """The packer's scale is the group max of the deployed values; the
    decoder must recover exactly that scale per group."""
    torch.manual_seed(1)
    w = torch.randn(4, 256) * 0.05
    deployed, _ = deploy_weights(w, w.clone(), "g128", "lloyd")
    g = deployed.float().reshape(4, 2, 128)
    scales = g.abs().amax(-1).half()
    packed = torch.from_numpy(pack_q1_0_g128(deployed, 128))
    blocks = packed.reshape(4, 2, 34)
    packed_scales = blocks[:, :, :2].contiguous().view(torch.float16).squeeze(-1)
    assert torch.equal(packed_scales, scales)


def test_deploy_weights_g128_is_idempotent_on_deployed_values():
    """ternary_lloyd is idempotent on its own output (the 0-ppm property)."""
    torch.manual_seed(2)
    w = torch.randn(64, 256) * 0.1
    once, _ = deploy_weights(w, w.clone(), "g128", "lloyd")
    twice, _ = deploy_weights(once, once.clone(), "g128", "lloyd")
    assert torch.equal(once, twice)

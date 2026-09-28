"""CPU tests for the W3 DeltaLoss probe's pure ranking pieces."""
import pytest
import torch

import conftest
from steer_probe import layer_of, quantizer_for, rank_entries, role_of

needs_autogrid = pytest.mark.skipif(
    conftest._AUTOGRID is None, reason="AUTOGRID_REPO checkout not found")


# ------------------------------------------------------------------ roles ----

def test_role_of_covers_every_fp_linear_site():
    assert role_of("layers.7.self_attn.q_proj") == "attn"
    assert role_of("layers.7.self_attn.o_proj") == "attn"
    assert role_of("layers.3.linear_attn.in_proj_qkv") == "gdn"
    assert role_of("layers.3.linear_attn.out_proj") == "gdn"
    assert role_of("layers.0.mlp.shared_expert.gate_proj") == "shared_expert"
    # the shared-expert gate is a swept nn.Linear, not the router and not
    # unattributable "other" -- it was the top-scoring tensor in the first run
    assert role_of("layers.0.mlp.shared_expert_gate") == "shared_expert_gate"
    # the ternary target and the router are labelled so they can be excluded
    assert role_of("layers.0.mlp.experts.gate_up_proj") == "expert_bank"
    assert role_of("layers.0.mlp.gate") == "router"
    assert role_of("lm_head") == "other"


def test_role_of_does_not_confuse_the_two_gates():
    """``mlp.gate`` is the router; ``mlp.shared_expert_gate`` is not."""
    assert role_of("layers.0.mlp.gate") != role_of("layers.0.mlp.shared_expert_gate")
    # a longer name ending in shared_expert_gate must still match only one rule
    assert role_of("layers.0.mlp.shared_expert_gate") == "shared_expert_gate"


def test_layer_of():
    assert layer_of("layers.29.linear_attn.out_proj") == 29
    assert layer_of("lm_head") is None


# ---------------------------------------------------------------- ranking ----

def test_rank_entries_sorts_desc_and_normalises():
    entries = [
        {"name": "a", "delta_loss": 1.0},
        {"name": "b", "delta_loss": 3.0},
        {"name": "c", "delta_loss": 2.0},
    ]
    out = rank_entries(entries)
    assert [e["name"] for e in out] == ["b", "c", "a"]
    assert [e["rank"] for e in out] == [1, 2, 3]
    assert out[0]["norm"] == 1.0
    assert out[-1]["norm"] == 1 / 3
    assert abs(sum(e["share"] for e in out) - 1.0) < 1e-9


def test_rank_entries_does_not_mutate_its_input():
    entries = [{"name": "a", "delta_loss": 1.0}, {"name": "b", "delta_loss": 2.0}]
    rank_entries(entries)
    assert [e["name"] for e in entries] == ["a", "b"]
    assert "rank" not in entries[0]


def test_rank_entries_handles_an_all_zero_sweep():
    """A prefix where nothing moved must still produce a usable ranking."""
    out = rank_entries([{"name": "a", "delta_loss": 0.0},
                        {"name": "b", "delta_loss": 0.0}])
    assert [e["name"] for e in out] == ["a", "b"]
    # shares of an all-zero sweep are genuinely 0, and must not be NaN
    assert all(e["share"] == 0.0 and e["norm"] == 0.0 for e in out)


# -------------------------------------------------------------- quantizer ----

@needs_autogrid
def test_quantizer_for_rejects_unknown_names():
    with pytest.raises(KeyError):
        quantizer_for("gptq")


@needs_autogrid
def test_quantizer_for_lloyd_is_ternary_and_grouped():
    quant = quantizer_for("lloyd", 128)
    torch.manual_seed(0)
    w = torch.randn(4, 256) * 0.3
    q = quant(w)
    assert q.shape == w.shape
    # every group of 128 must take at most three distinct levels
    for row in q.reshape(-1, 128):
        assert len(torch.unique(row)) <= 3


@needs_autogrid
def test_quantizer_reconstruction_is_better_than_nothing():
    """Sanity: the deployed rule must actually reduce weight error."""
    quant = quantizer_for("lloyd", 128)
    torch.manual_seed(1)
    w = torch.randn(8, 128) * 0.25
    err = (quant(w) - w).pow(2).mean()
    assert err < w.pow(2).mean()

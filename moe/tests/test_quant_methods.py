"""CPU tests for the quant-method pieces: CAT-Q, KD loss filtering, AYOT packing."""
import json
from types import SimpleNamespace

import pytest
import torch

from ayot import load_prompts, load_traces, mix_windows, windows_from_texts
from catq import (catq_dequantize, catq_reconstruct, hard_ternarize,
                  soft_ternarize, ternary_catq)
from kd_loss import filtered_mean, kd_filtered, kl_per_token
from moe_proxy import ternary_absmean


# ------------------------------------------------------------------ catq ----

def test_soft_ternarize_limits():
    w = torch.tensor([[-2.0, -0.25, 0.25, 2.0]])
    near_identity = soft_ternarize(w, 1e-3, 0.5)
    assert torch.allclose(near_identity, w, atol=5e-3)
    near_hard = soft_ternarize(w, 500.0, 0.5)
    assert torch.allclose(near_hard, hard_ternarize(w, 0.5), atol=5e-3)


def test_catq_reconstruct_shape_and_ternary():
    torch.manual_seed(0)
    w = torch.randn(4, 128) * 0.5 + 0.3
    codes, scales = catq_reconstruct(w, group=128, steps=60)
    assert codes.shape == w.shape
    assert scales.shape == (4, 1)
    assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert (scales > 0).all()
    assert catq_dequantize(codes, scales, 128).shape == w.shape


def test_catq_reconstruct_3d_bank():
    torch.manual_seed(1)
    w = torch.randn(2, 4, 128) * 0.4
    codes, scales = catq_reconstruct(w, group=128, steps=30)
    assert codes.shape == w.shape
    assert scales.shape == (2, 4, 1)
    q = catq_dequantize(codes, scales, 128)
    assert q.shape == w.shape


def test_catq_beats_absmean_on_shifted_outlier_groups():
    torch.manual_seed(0)
    w = torch.randn(2, 128) * 0.4 + 0.35
    w[0, :4] *= 8.0
    mse_absmean = ((w - ternary_absmean(w, 128)) ** 2).mean().item()
    mse_catq = ((w - ternary_catq(w, 128, steps=200)) ** 2).mean().item()
    assert mse_catq < mse_absmean


# --------------------------------------------------------------- kd loss ----

def test_kl_zero_for_identical():
    x = torch.randn(2, 5)
    assert torch.allclose(kl_per_token(x, x, temp=2.0), torch.zeros(2), atol=1e-5)


def test_filtered_mean_drops_top_fraction():
    losses = torch.tensor([1.0, 2.0, 3.0, 100.0])
    assert filtered_mean(losses, 0.25).item() == pytest.approx(2.0)
    assert filtered_mean(losses, 0.0).item() == pytest.approx(26.5)
    assert filtered_mean(losses, 1.0).item() == pytest.approx(1.0)


def test_kd_filtered_matches_plain_when_off():
    torch.manual_seed(0)
    s, t = torch.randn(3, 7), torch.randn(3, 7)
    plain = kl_per_token(s, t, temp=2.0).mean()
    assert torch.allclose(kd_filtered(s, t, 2.0, 0.0), plain, atol=1e-6)


# ------------------------------------------------------------------ ayot ----

class FakeTok:
    def __call__(self, text, return_tensors=None):
        ids = torch.arange(1, len(text) + 1).unsqueeze(0)
        return SimpleNamespace(input_ids=ids)


def test_mix_windows_fraction_and_determinism():
    general = torch.arange(100).reshape(10, 10)
    agentic = torch.full((4, 10), -1)
    mixed = mix_windows(general, agentic, frac=0.2, seed=0)
    assert (mixed < 0).sum().item() // 10 == 2
    assert torch.equal(mixed, mix_windows(general, agentic, frac=0.2, seed=0))
    assert torch.equal(mix_windows(general, agentic, frac=0.0, seed=0), general)


def test_mix_windows_errors():
    with pytest.raises(ValueError):
        mix_windows(torch.zeros(4, 2), torch.zeros(1, 2), frac=0.5, seed=0)
    with pytest.raises(ValueError):
        mix_windows(torch.zeros(4, 2), torch.zeros(4, 2), frac=1.5)


def test_windows_from_texts_and_jsonl(tmp_path):
    text = "abcdefghij" * 20
    w = windows_from_texts(FakeTok(), [text], n=3, seq=8, seed=0)
    assert w.shape == (3, 8)
    with pytest.raises(ValueError):
        windows_from_texts(FakeTok(), ["ab"], n=1, seq=8, seed=0)

    p = tmp_path / "traces.jsonl"
    p.write_text(json.dumps({"question": "q1"}) + "\n"
                 + json.dumps({"trace": "t1"}) + "\n")
    assert load_prompts(p) == ["q1"]
    assert load_traces(p) == ["t1"]

"""CPU tests for the W1 gate instrument's math (no transformers, no datasets)."""
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from kld_eval import (host_memory_guard, kld_decompose, kld_from_logprobs,
                      kld_per_token, log_probs, tail_sample_estimate, tail_stats,
                      top1_agreement)
from olmoe_proxy import windows


def test_log_probs_match_softmax():
    torch.manual_seed(0)
    x = torch.randn(70, 17) * 3
    lp = log_probs(x, chunk=8)
    assert lp.shape == (70, 17)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(70), atol=1e-5)
    # chunking must not change the answer
    assert torch.allclose(lp, F.log_softmax(x, dim=-1), atol=1e-6)


def test_kld_is_zero_for_identical_distributions():
    torch.manual_seed(1)
    t = torch.randn(40, 23)
    assert torch.allclose(kld_per_token(t, t), torch.zeros(40), atol=1e-6)


def test_kld_is_nonnegative_and_ordering_sensitive():
    torch.manual_seed(2)
    t = torch.randn(40, 23)
    near = kld_per_token(t, t + 0.05 * torch.randn_like(t))
    far = kld_per_token(t, t + 2.0 * torch.randn_like(t))
    assert (near >= -1e-6).all() and (far >= -1e-6).all()
    assert far.mean() > near.mean()


def test_kld_sees_the_tail_the_top_k_kd_never_looks_at():
    """The whole point: full-vocab KLD charges for a wrong tail.

    A 1000-wide teacher carrying 99.7% of its mass outside the argmax is
    exactly the case a top-50 KD term and PPL cannot see, and exactly the case
    the 1.7B canary failure mode lived in.  A student that is right on the argmax
    but wrong everywhere else still has to pay.
    """
    V = 1000
    teacher = [1.0] + [0.0] * (V - 1)
    perfect = list(teacher)
    inverted = [0.0, 1.0] + [0.0] * (V - 2)        # gold at rank 2 (canary mode)
    concentrated = [1.0, 3.0] + [0.0] * (V - 2)   # tail mass dumped on one token

    def kld(student):
        return kld_per_token(torch.tensor([teacher]),
                             torch.tensor([student])).item()

    assert kld(perfect) == pytest.approx(0.0, abs=1e-5)
    assert kld(inverted) > 0.0
    assert kld(concentrated) > 0.0
    # misranking the gold token costs less than scrambling the tail mass
    assert kld(concentrated) > kld(inverted)


def test_kld_renormalises_an_unnormalised_student():
    t = torch.tensor([[2.0, 0.0]])
    lp = log_probs(t)
    shifted = lp + 3.0                      # not a valid log-distribution
    assert kld_from_logprobs(lp, shifted).abs().max().item() < 1e-6


def test_decomposition_sums_to_the_full_kld():
    """The three chain-rule pieces are the metric, exactly, token by token."""
    torch.manual_seed(4)
    t = log_probs(torch.randn(64, 37) * 2)
    s = log_probs(torch.randn(64, 37) * 2)
    dec = kld_decompose(t, s, k=7, chunk=8)
    full = kld_from_logprobs(t, s, chunk=8)
    assert torch.allclose(dec["full"], full, atol=1e-5)
    assert torch.allclose(dec["marginal"] + dec["support"] + dec["tail"], full,
                          atol=1e-5)
    # pieces are non-negative, and the shares are a partition of the total
    pieces = torch.stack([dec["marginal"], dec["support"], dec["tail"]])
    assert (pieces >= -1e-5).all()
    shares = pieces / full
    assert torch.allclose(shares.sum(0), torch.ones(64), atol=1e-4)


def test_decomposition_is_zero_at_a_match():
    torch.manual_seed(6)
    t = log_probs(torch.randn(16, 29))
    dec = kld_decompose(t, t, k=5)
    for name in ("marginal", "support", "tail", "full"):
        assert dec[name].abs().max() < 1e-5


def test_decomposition_puts_a_tail_only_error_in_the_tail_piece():
    """A student wrong only on the complement must charge the tail piece.

    This is the discrimination the measurement exists for: the top-k KD term
    cannot see this error, so if the gate's remaining KLD lives here, the next
    objective term has to be tail-conditional (TAD's D_KL2), not another top-k
    rebalance.
    """
    torch.manual_seed(7)
    t = torch.randn(8, 50) * 2
    t_lp = log_probs(t)
    idx = t.topk(10, dim=-1).indices
    mask = torch.zeros_like(t, dtype=torch.bool).scatter_(-1, idx, True)
    s = t.clone()
    noise = torch.randn_like(s)
    noise = noise - noise.mean()           # keep the mass shift small
    s[~mask] = s[~mask] + noise[~mask]     # reshape the complement, not its mass
    dec = kld_decompose(t_lp, log_probs(s), k=10)
    assert dec["tail"].mean() > 0.0
    assert dec["tail"].mean() > 10 * dec["support"].mean()
    # the support conditional is untouched (its logits never moved), so the only
    # other charge is the small mass shift
    assert dec["support"].mean() < 1e-5
    assert dec["marginal"].mean() < dec["tail"].mean()


def test_decomposition_rejects_k_out_of_range():
    t = log_probs(torch.randn(4, 8))
    with pytest.raises(ValueError):
        kld_decompose(t, t, k=8)
    with pytest.raises(ValueError):
        kld_decompose(t, t, k=0)


def test_tail_sample_estimate_is_exact_when_the_tail_is_uniform():
    """Uniform tail ⇒ the sample mean IS the p-weighted expectation."""
    torch.manual_seed(8)
    t = torch.randn(6, 20)
    t[:, :5] += 5.0                      # top-5 guaranteed
    t[:, 5:] = -3.0                      # uniform tail over the complement
    s = log_probs(torch.randn(6, 20) * 2)
    k = 5
    tl = log_probs(t)
    exact = kld_decompose(tl, s, k)["tail"]
    est = tail_sample_estimate(tl, s, k, m=20 - k, replacement=False)
    assert torch.allclose(est, exact, atol=1e-5)


def test_tail_sample_estimate_is_exact_for_a_degenerate_tail():
    """When the teacher's tail is one token, the sample must hit it and be exact."""
    t = torch.full((3, 10), -30.0)
    t[:, 0], t[:, 1], t[:, 7] = 0.0, -1.0, -2.0      # top-2 = {0,1}; tail = {7}
    s = torch.full((3, 10), -30.0)
    s[:, 0], s[:, 1], s[:, 8], s[:, 7] = -0.5, -1.5, -3.0, -4.0
    tl, sl = log_probs(t), log_probs(s)
    exact = kld_decompose(tl, sl, k=2)["tail"]
    est = tail_sample_estimate(tl, sl, 2, m=4)
    assert torch.allclose(est, exact, atol=1e-5)


def test_tail_sample_estimate_converges_and_has_less_variance_with_more():
    torch.manual_seed(9)
    t = log_probs(torch.randn(256, 64) * 2)
    s = log_probs(torch.randn(256, 64) * 2)
    k = 8
    exact = kld_decompose(t, s, k)["tail"]
    few = tail_sample_estimate(t, s, k, m=2)
    many = tail_sample_estimate(t, s, k, m=64)
    assert many.std() < few.std()
    assert abs(many.mean() - exact.mean()) < 0.1 * abs(exact.mean()) + 1e-3


def test_tail_sample_estimate_rejects_bad_k():
    t = log_probs(torch.randn(4, 8))
    with pytest.raises(ValueError):
        tail_sample_estimate(t, t, k=8, m=2)


def test_host_memory_guard_refuses_a_cache_that_cannot_coexist_with_the_reload(monkeypatch):
    """The guard protects the host peak, which the bf16 loader cut to ~7 GB.

    Session-4 numbers: at 18 GB available the real gate cache (4.1 GB) fits
    (18 - 9 = 9 >= 4.1), while an oversized 11 GB cache is refused.  The 00:20
    OOM was 16 windows (~8 GB) on top of the old fp32 reload.
    """
    import builtins
    import io

    real_open = builtins.open

    def fake_open(path, *a, **k):
        if path == "/proc/meminfo":
            return io.StringIO("MemTotal:       30000000 kB\n"
                               "MemAvailable:   18000000 kB\n")
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", fake_open)
    with pytest.raises(SystemExit):
        host_memory_guard(need_bytes=11_000_000_000)     # 11 > 18-9 GB budget
    host_memory_guard(need_bytes=4_100_000_000)          # the real gate: fits


def test_tail_stats_flags_an_outlier_token():
    per_token = torch.full((1000,), 0.01)
    per_token[417] = 9.0
    s = tail_stats(per_token)
    assert s["n_tokens"] == 1000
    assert s["argmax_token"] == 417
    assert abs(s["max"] - 9.0) < 1e-6
    assert abs(s["mean"] - (0.01 * 999 + 9.0) / 1000) < 1e-6
    # p99.9 of 1000 tokens is the second-worst; it must already separate the
    # outlier from the body, which is exactly the gate signal
    assert s["p99.9"] == pytest.approx(0.01)
    assert s["p99.9"] < s["max"]
    for key in ("p50", "p90", "p99"):
        assert s[key] == pytest.approx(0.01)


def test_p999_is_reported_as_unresolved_at_the_default_sample_size():
    """4088 tokens is the real default: p99.9 there is the 5th-worst token."""
    flat = torch.rand(4088)
    flat[:4] = torch.tensor([50.0, 40.0, 30.0, 20.0])
    s = tail_stats(flat)
    assert s["n_tokens"] == 4088
    assert s["p999_resolved"] is False
    # 4088 - round(0.999 * 4087) == 5: p99.9 sits 5th from the top.  The four
    # planted outliers are above it, so p99.9 lands on the *background* -- one
    # token of slack is all that separates it from "the 4th worst outlier".
    assert s["p999_rank"] == 5
    assert s["max"] == pytest.approx(50.0)
    assert s["p99.9"] < 20.0          # 4 outliers are strictly above it
    assert s["p99.9"] == pytest.approx(flat.max(), rel=1.0)


def test_p999_rank_scales_as_one_in_a_thousand():
    """The rank is n/1000 by construction -- the resolution is the sample size."""
    assert tail_stats(torch.rand(1000))["p999_rank"] == 2
    assert tail_stats(torch.rand(4088))["p999_rank"] == 5
    assert tail_stats(torch.rand(20000))["p999_rank"] == 21
    assert tail_stats(torch.rand(100_000))["p999_rank"] == 101


def test_p999_resolves_only_once_there_are_100k_tokens():
    """P999_MIN_RANK=100 means ~1e5 tokens, which no local run can reach.

    Pinned so the threshold is not quietly loosened: the honest consequence is
    that every local p99.9 number is really a top-few statistic, and the gate
    has to lean on `max` plus `worst_tokens` until this runs at scale.
    """
    five = torch.tensor([90.0, 80.0, 70.0, 60.0, 50.0])
    small = tail_stats(torch.cat([five, torch.rand(20000)]))
    assert small["p999_resolved"] is False
    assert small["p99.9"] < small["max"]

    big = tail_stats(torch.cat([five, torch.rand(100_000)]))
    assert big["p999_resolved"] is True
    assert big["p99.9"] < big["max"]


def test_p999_rank_is_one_for_a_tiny_sample():
    """A handful of tokens makes p99.9 literally the max."""
    s = tail_stats(torch.tensor([0.1, 0.2, 0.3, 0.9]))
    assert s["p999_rank"] == 1
    assert s["p999_resolved"] is False
    assert s["p99.9"] == pytest.approx(s["max"])


def test_tail_stats_percentiles_are_monotone():
    torch.manual_seed(3)
    per_token = torch.randn(5000).abs() * 0.1
    s = tail_stats(per_token)
    assert s["p50"] <= s["p90"] <= s["p99"] <= s["p99.9"] <= s["max"]


def test_eval_windows_are_1d_rows_not_batches(monkeypatch):
    """``windows()`` returns [n, seq], so each window is a 1-D row.

    The teacher pass originally sized its memory estimate with ``d.shape[1]``,
    which raised IndexError on a 1-D row -- so the instrument had never actually
    run.  Pinned against the real packing function, with the dataset fetch
    stubbed out (the CPU venv has no ``datasets``).
    """
    rows = [{"text": "alpha beta gamma " * 2000}]

    def _load_dataset(*a, **k):
        return rows                      # windows() just iterates the result
    monkeypatch.setitem(sys.modules, "datasets",
                        SimpleNamespace(load_dataset=_load_dataset))

    class _Tok:
        def __call__(self, text, return_tensors="pt"):
            assert return_tensors == "pt"
            ids = torch.arange(0, 4000)
            return SimpleNamespace(input_ids=[ids])

    data = windows(_Tok(), 3, 512, 0, "fineweb", max_chars=10_000)
    assert data.dim() == 2 and data.shape == (3, 512)
    row = data[0]
    assert row.dim() == 1
    # the expression the teacher pass uses, on the real layout
    assert row.numel() - 1 == 511
    with pytest.raises(IndexError):
        row.shape[1]          # the bug that broke the first gate run


def test_tail_stats_rejects_empty():
    with pytest.raises(ValueError):
        tail_stats(torch.zeros(0))


def test_top1_agreement():
    t = torch.tensor([[3.0, 1.0], [0.0, 5.0], [1.0, 2.0]])
    assert top1_agreement(log_probs(t), log_probs(t)) == pytest.approx(1.0)
    flipped = torch.tensor([[1.0, 3.0], [0.0, 5.0], [4.0, 2.0]])
    assert top1_agreement(log_probs(t), log_probs(flipped)) == pytest.approx(1 / 3)

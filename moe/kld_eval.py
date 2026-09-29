"""Full-vocabulary KLD between the FP prefix teacher and the corrected student.

The W1 gate instrument (``docs/TAIL-EXPERIMENT-PLAN.md`` step 1).  Correction
training only ever matches the teacher's **top-50** logits, so the rest of the
student distribution is unconstrained -- and PPL cannot see that, because a
ternary body can hold mean likelihood while the tail is wrong.  The gate is
written against per-token ``KL(teacher || student)`` over the *whole*
vocabulary, reported as mean / p99 / p99.9 / max.

Both sides are the same N-layer prefix, so the vocabularies line up token for
token:

  teacher : the prefix exactly as loaded (FP banks, no branches)
  student : banks ternarised in place (``--quant``) + correction branches
            (``--branch-*``) + ``--load`` checkpoint

The student is built by ``qwen35_moe_proxy.build_student``, the same helper the
train/eval stages use, so this instrument cannot drift from the stage it
measures.

The teacher pass runs first and parks its log-probabilities on the CPU, because
ternarisation rewrites the banks in place and the two passes cannot share one
resident model.  That costs ``n_windows * (seq-1) * vocab * 4`` bytes of host
memory -- about 4 GB for the default 8 windows at seq 512 with a 248k vocab --
so the estimate is printed before the pass starts.

**Read the tail stats with the sample size in hand.**  ``tail_stats`` reports
``p999_rank``, the number of tokens at or above the p99.9 position, because that
percentile is only n/1000-th from the top by construction.  At the default 8
windows that is the 5th-worst token, so ``p99.9`` and ``max`` are nearly the same
measurement and ``p99_resolved`` is false.  The gate should lean on ``max`` and
``worst_tokens`` (which record window and position, so a spike is inspectable
rather than merely reported) until the instrument runs at >= 1e5 tokens, which
needs more eval windows than host memory allows for a 248k vocab.

Example (4-layer prefix on the free card):

    HIP_VISIBLE_DEVICES=1 python moe/kld_eval.py --prefix-layers 4 \\
        --device cuda:0 --quant lloyd --branch-quant g128 --branch-target both \\
        --rank 512 --load $MOE/qwen35-corr-r512-g128-step4096.pt \\
        --out $MOE/kld-armA.json

The pure math (log-probs, per-token KLD, tail stats) is import-safe without
transformers/datasets so it can be unit-tested on CPU; see
``moe/tests/test_kld_eval.py``.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ART = Path(os.environ.get("MOE_ARTIFACTS", HERE / "artifacts"))
OUT = ART / "qwen35"


# ------------------------------------------------------------------- math ----

def log_probs(logits: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """fp32 log-softmax over the vocab, chunked over tokens to bound memory."""
    parts = [F.log_softmax(logits[i:i + chunk].float(), dim=-1)
             for i in range(0, logits.shape[0], chunk)]
    return torch.cat(parts)


def kld_from_logprobs(tlp: torch.Tensor, slp: torch.Tensor,
                      chunk: int = 32) -> torch.Tensor:
    """Per-token ``KL(teacher || student)`` from two [T, V] log-prob blocks.

    ``tlp`` is assumed already normalised; ``slp`` is renormalised here so a
    student that has drifted out of the simplex cannot produce a negative KLD.
    """
    out = []
    for i in range(0, tlp.shape[0], chunk):
        t = tlp[i:i + chunk]
        s = slp[i:i + chunk] - torch.logsumexp(slp[i:i + chunk], dim=-1, keepdim=True)
        out.append((t.exp() * (t - s)).sum(-1))
    return torch.cat(out)


def kld_per_token(teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                  chunk: int = 32) -> torch.Tensor:
    """Convenience wrapper: logits in, per-token full-vocab KLD out."""
    return kld_from_logprobs(log_probs(teacher_logits, chunk),
                             log_probs(student_logits, chunk), chunk)


def kld_decompose(tlp: torch.Tensor, slp: torch.Tensor, k: int,
                  chunk: int = 32) -> dict[str, torch.Tensor]:
    """Chain-rule split of the full-vocab KLD over the teacher's top-k support.

    ``tlp``/``slp`` are [T, V] log-prob blocks; ``slp`` is renormalised here
    (same guard as ``kld_from_logprobs``).  For each token, with S = teacher
    top-k and w = mass on S, the full-vocab KL splits exactly as::

        KL(t||s) = KL_bin(w_t || w_s)                    <- "marginal"
                 + w_t     * KL(p_t(.|S)  || p_s(.|S))   <- "support"
                 + (1-w_t) * KL(p_t(.|~S) || p_s(.|~S))  <- "tail"

    The returned tensors are the three *weighted* pieces (which sum to the
    full-vocab KLD per token), the total, and the support masses.  This is the
    measurement that decides which piece of the gate's own metric the next
    objective term should target: the top-k KD loss optimises only "support",
    the residual-mass term only "marginal", and "tail" is TAD's D_KL2 /
    TA-OPD's lower-bound bias (see RESEARCH-HANDOFF §7.1a).  The complement
    masses are computed in log space (``logsumexp`` of the masked log-probs),
    never as ``1 - w`` in probability space.
    """
    if not 0 < k < tlp.shape[-1]:
        raise ValueError(f"k={k} must be in (0, V={tlp.shape[-1]})")
    out: dict[str, list[torch.Tensor]] = {name: [] for name in
                                          ("marginal", "support", "tail", "full",
                                           "w_t", "w_s")}
    for i in range(0, tlp.shape[0], chunk):
        t = tlp[i:i + chunk].float()
        s = slp[i:i + chunk].float()
        s = s - torch.logsumexp(s, dim=-1, keepdim=True)
        idx = t.topk(k, dim=-1).indices
        log_wt = torch.logsumexp(t.gather(-1, idx), dim=-1)
        log_ws = torch.logsumexp(s.gather(-1, idx), dim=-1)
        mask = torch.zeros_like(t, dtype=torch.bool).scatter_(-1, idx, True)
        log_tt = torch.logsumexp(t.masked_fill(mask, float("-inf")), dim=-1)
        log_ts = torch.logsumexp(s.masked_fill(mask, float("-inf")), dim=-1)

        # marginal: binary KL between the support masses
        marginal = (log_wt.exp() * (log_wt - log_ws)
                    + log_tt.exp() * (log_tt - log_ts))

        # support: teacher-weighted conditional KL on the support
        tS, sS = t.gather(-1, idx), s.gather(-1, idx)
        log_pt = tS - log_wt.unsqueeze(-1)
        log_ps = sS - log_ws.unsqueeze(-1)
        support = log_wt.exp() * (log_pt.exp() * (log_pt - log_ps)).sum(-1)

        # tail: teacher-weighted conditional KL on the complement
        log_pt_t = t - log_tt.unsqueeze(-1)
        log_ps_t = s - log_ts.unsqueeze(-1)
        kl_tail = ((log_pt_t.exp() * (log_pt_t - log_ps_t))
                   .masked_fill(mask, 0.0).sum(-1))
        tail = log_tt.exp() * kl_tail

        for name, value in (("marginal", marginal), ("support", support),
                            ("tail", tail), ("w_t", log_wt.exp()),
                            ("w_s", log_ws.exp())):
            out[name].append(value)
        out["full"].append(marginal + support + tail)
    return {name: torch.cat(parts) for name, parts in out.items()}


def tail_sample_estimate(tlp: torch.Tensor, slp: torch.Tensor, k: int,
                         m: int, chunk: int = 32,
                         replacement: bool = True) -> torch.Tensor:
    """Sample-based estimate of the tail-conditional KL (TAD's D_KL2).

    The exact term needs per-token probabilities over the ~V-k complement,
    which a top-k cache cannot store.  Drawing ``m`` tokens per position from
    the *teacher's own tail conditional* gives an unbiased estimate of the KL::

        KL(p_t(.|~S) || p_s(.|~S))
            = E_{v ~ p_t(.|~S)}[ log p_t(v) - log p_s(v) ]  - log tail_t + log tail_s

    (the first expectation is what the samples estimate; the normalisers are
    exact logsumexps over the complement).  The result is weighted by the
    teacher's tail mass, so it estimates the same "tail" piece as
    ``kld_decompose`` and matches TAD's ``alpha_K * D_KL2``.  The teacher side
    of the sampled expectation is a constant for the student, so its gradient is
    unbiased too (Sparse Logit Sampling, ACL 2025).  ``replacement=False`` with
    ``m`` equal to the complement size makes the estimate exact -- used by the
    tests.

    This is the eval-side validation of the estimator the cache stage would
    store; the sampling is per-row multinomial over the masked tail.
    """
    if not 0 < k < tlp.shape[-1]:
        raise ValueError(f"k={k} must be in (0, V={tlp.shape[-1]})")
    out = []
    for i in range(0, tlp.shape[0], chunk):
        t = tlp[i:i + chunk].float()
        s = slp[i:i + chunk].float()
        s = s - torch.logsumexp(s, dim=-1, keepdim=True)
        mask = torch.zeros_like(t, dtype=torch.bool)
        mask.scatter_(-1, t.topk(k, dim=-1).indices, True)
        t_masked = t.masked_fill(mask, float("-inf"))
        s_masked = s.masked_fill(mask, float("-inf"))
        log_tt = torch.logsumexp(t_masked, dim=-1, keepdim=True)
        log_ts = torch.logsumexp(s_masked, dim=-1, keepdim=True)
        probs = (t_masked - log_tt).exp()
        idx = torch.multinomial(probs, m, replacement=replacement)
        # log p_t(v) - log p_s(v) for the samples, with exact tail normalisers
        diff = ((t.gather(-1, idx) - log_tt) - (s.gather(-1, idx) - log_ts))
        # weight by the teacher's tail mass, matching kld_decompose's "tail"
        # piece and TAD's alpha_K * D_KL2 (alpha_K = 1 - w_t)
        out.append(diff.mean(-1) * log_tt.exp().squeeze(-1))
    return torch.cat(out)


#: percentiles reported by default; p99.9 and max are the gate, the rest is
#: context so a tail move can be told apart from a uniform shift.
PERCENTILES = (50.0, 90.0, 99.0, 99.9)

#: Minimum number of tokens at-or-above the p99.9 position for that percentile
#: to carry information beyond the top few outliers.  8 windows x 511 tokens =
#: 4088 tokens, where p99.9 is the 5th-worst observation -- statistically
#: indistinguishable from the max.  Resolving it properly needs more eval
#: windows (memory-bound; see the module docstring on the teacher cache).
P999_MIN_RANK = 100


def tail_stats(per_token: torch.Tensor) -> dict:
    """Mean / percentiles / max of the per-token KLD, plus the worst token.

    ``p999_above`` is the number of tokens at or beyond the p99.9 value: it is
    the honest order-statistic rank behind that percentile.  When it is small,
    p99.9 and max are the same measurement and only ``max`` is meaningful.
    """
    flat = per_token.reshape(-1).float()
    if flat.numel() == 0:
        raise ValueError("no tokens")
    ordered = torch.sort(flat).values
    worst = int(flat.argmax())
    out = {
        "n_tokens": int(flat.numel()),
        "mean": float(flat.mean()),
        "max": float(ordered[-1]),
        "argmax_token": worst,
        "p50": float(torch.quantile(ordered, 0.50)),
    }
    for p in PERCENTILES:
        if p == 50.0:
            continue
        # exact order statistics; torch.quantile's interpolation would smear the
        # very tail this instrument exists to resolve
        idx = min(int(round(p / 100.0 * (ordered.numel() - 1))), ordered.numel() - 1)
        out[f"p{p:g}"] = float(ordered[idx])
    p999_idx = min(int(round(0.999 * (ordered.numel() - 1))), ordered.numel() - 1)
    # how many tokens sit at or above the p99.9 position, counting from the top.
    # This is the honest resolution of the percentile: at 4088 tokens it is 5,
    # i.e. p99.9 is the 5th-worst token and carries no more information than
    # the max.  (A count of values >= p99.9 would be confounded by ties and by
    # however many background samples land there.)
    out["p999_rank"] = int(ordered.numel() - p999_idx)
    out["p999_resolved"] = bool(out["p999_rank"] >= P999_MIN_RANK)
    return out


def topk_coverage(teacher_logits_full: torch.Tensor,
                  cached_vals: torch.Tensor) -> dict:
    """How much teacher mass a top-k cache actually holds.

    ``teacher_logits_full`` is [T, V] of raw teacher logits and ``cached_vals``
    is the [T, k] top-k of the same row. Renormalising ``cached_vals`` on its
    own returns 1.0 by construction and says nothing -- the cache stores raw
    logits, not normalised logprobs. The real quantity needs the full-vocab
    normaliser:

        captured = sum_{v in topk} exp(logit_v) / sum_{all v} exp(logit_v)

    This is what decides whether a sampled residual-mass tail term is worth
    building, and it is only computable here, where the full teacher pass
    already exists.
    """
    full = teacher_logits_full.float()
    top = cached_vals.float()
    if full.shape[0] == 0:
        return {"n_tokens": 0, "top_k": int(top.shape[-1]),
                "mean_topk_mass": 0.0, "min_topk_mass": 0.0,
                "p01_topk_mass": 0.0, "mean_residual_mass": 0.0}
    denom = torch.logsumexp(full, dim=-1)
    numer = torch.logsumexp(top, dim=-1)
    captured = (numer - denom).exp().clamp(0.0, 1.0)
    return {
        "n_tokens": int(captured.numel()),
        "top_k": int(top.shape[-1]),
        "mean_topk_mass": float(captured.mean()),
        "min_topk_mass": float(captured.min()),
        "p01_topk_mass": float(torch.quantile(captured, 0.01)),
        "mean_residual_mass": float(1.0 - captured.mean()),
    }


def _coverage(tcache: list[torch.Tensor], raw_top: list[torch.Tensor],
              k: int) -> dict:
    """Top-k mass coverage from the parked full log-probs.

    The parked tensor is the *full-vocab* log-softmax, so ``exp`` of a log-prob
    is the true probability mass, and summing the top-k of those recovers exactly
    what a top-k cache holds. No renormalisation over the cached entries (that
    returns 1.0 by construction and measures nothing) and no second teacher pass.

    ``raw_top`` is accepted so the caller can pass both consistently; only ``k``
    and the parked rows are needed.
    """
    captured = [torch.topk(lp, k, dim=-1).values.exp().sum(-1).clamp(0.0, 1.0)
                for lp in tcache]
    c = torch.cat(captured)
    return {
        "n_tokens": int(c.numel()),
        "top_k": int(k),
        "mean_topk_mass": float(c.mean()),
        "min_topk_mass": float(c.min()),
        "p01_topk_mass": float(torch.quantile(c, 0.01)),
        "mean_residual_mass": float(1.0 - c.mean()),
    }


def top1_agreement(tlp: torch.Tensor, slp: torch.Tensor) -> float:
    """Fraction of tokens where teacher and student agree on the argmax token.

    Reported next to the KLD because it is the quantity the 1.7B canary failure
    mode (gold answer stuck at rank 2 behind near-ties) actually showed.
    """
    return float((tlp.argmax(-1) == slp.argmax(-1)).float().mean())


# ------------------------------------------------------------------- run -----

def host_memory_guard(need_bytes: int, reserve_gb: float = 9.0) -> None:
    """Refuse a teacher cache that cannot coexist with the prefix reload.

    ``load_prefix`` loads the prefix on the host before moving it to the GPU,
    and the parked teacher log-probs stay in host RAM for the whole student
    pass.  Sixteen windows at seq 512 with a 248k vocab is ~8 GB of cache;
    together with the old fp32 host construction that OOM'd the 30 GB box
    (session 3, 00:20), which is where the original 18 GB reserve came from.
    The loader now constructs in bf16 and uses ``assign=True``, and a measured
    4-layer load peaks at **7.0 GB** (2026-09-29, session 4), so the reserve is
    9 GB -- re-measure it if the loader changes again.  Reads MemAvailable and
    refuses loudly instead of dying mid-pass.
    """
    try:
        with open("/proc/meminfo") as f:
            avail = next(int(line.split()[1]) * 1024 for line in f
                         if line.startswith("MemAvailable"))
    except (OSError, StopIteration):
        return
    budget = avail - int(reserve_gb * 1e9)
    if need_bytes > budget:
        raise SystemExit(
            f"teacher cache needs {need_bytes / 1e9:.1f} GB but only "
            f"{budget / 1e9:.1f} GB is available after the ~{reserve_gb:.0f} GB "
            f"prefix reload; lower --eval-windows (8 is safe on this host)")


@torch.no_grad()
def teacher_pass(model, data, device, chunk: int, vocab: int, top_k: int = 0):
    """Park fp32 teacher log-probs per window, plus the teacher's own entropy.

    The teacher entropy/peak are the reference the student's numbers are read
    against, so they are measured here rather than recomputed later. When
    ``top_k`` is set, the raw top-k logits are returned too, so the caller can
    measure what fraction of teacher mass a top-k cache would hold.
    """
    from qwen35_moe_proxy import model_logits

    # ``windows()`` hands back one 1-D row per window, so the token count is the
    # number of elements, not shape[1] (which raises on a 1-D row).
    need = sum(d.numel() - 1 for d in data) * vocab * 4
    print(f"teacher cache: {need/1e9:.2f} GB of host memory "
          f"({len(data)} windows x {vocab} vocab)", flush=True)
    raw_top: list[torch.Tensor] = []
    out, ents, tops = [], [], []
    for i in range(len(data)):
        ids = data[i:i + 1].to(device)
        raw = model_logits(model, ids)[0, :-1]
        lp = log_probs(raw, chunk).cpu()
        out.append(lp)
        ents.append(float(-(lp.exp() * lp).sum(-1).mean()))
        tops.append(float(lp.exp().max(-1).values.mean()))
        # stash the raw top-k logits so the caller can compute coverage against a
        # cache, using the full-vocab normaliser rather than renormalising over
        # the cached entries (which is 1.0 by construction and means nothing)
        if top_k:
            out[-1] = lp
            raw_top.append(raw.topk(top_k, dim=-1).values.cpu())
        print(f"  teacher {i+1}/{len(data)} entropy {ents[-1]:.3f} peak {tops[-1]:.4f}",
              flush=True)
    return out, sum(ents) / len(ents), sum(tops) / len(tops), raw_top


@torch.no_grad()
def student_pass(model, data, device, tcache, chunk: int, decompose_topk: int = 0,
                 tail_samples: int = 0):
    """Per-token KLD, top-1 agreement, mean entropy and mean top-1 mass.

    Entropy and peak mass come along for free and are what make the KLD number
    readable: see the note in ``main``.  When ``decompose_topk`` is set, the
    per-token chain-rule pieces are accumulated too and returned as a fifth
    value; ``tail_samples`` additionally estimates the tail piece from that many
    sampled tail tokens (``tail_sample_estimate``).
    """
    from qwen35_moe_proxy import model_logits

    per_token, agrees = [], []
    ents, tops = [], []
    dec_parts: list[dict[str, torch.Tensor]] = []
    for i in range(len(data)):
        ids = data[i:i + 1].to(device)
        slp = log_probs(model_logits(model, ids)[0, :-1], chunk).cpu()
        kld = kld_from_logprobs(tcache[i], slp, chunk)
        per_token.append(kld)
        agrees.append(top1_agreement(tcache[i], slp))
        ents.append(float(-(slp.exp() * slp).sum(-1).mean()))
        tops.append(float(slp.exp().max(-1).values.mean()))
        if decompose_topk:
            dec = kld_decompose(tcache[i], slp, decompose_topk, chunk)
            if tail_samples:
                dec["tail_est"] = tail_sample_estimate(
                    tcache[i], slp, decompose_topk, tail_samples, chunk)
            dec_parts.append(dec)
        del slp
        print(f"  student {i+1}/{len(data)} mean kld {kld.mean():.4f} "
              f"max {kld.max():.4f} top1 {agrees[-1]:.4f} "
              f"H {ents[-1]:.3f} peak {tops[-1]:.4f}", flush=True)
    result = (torch.cat(per_token), sum(agrees) / len(agrees),
              sum(ents) / len(ents), sum(tops) / len(tops))
    if decompose_topk:
        dec = {name: torch.cat([d[name] for d in dec_parts])
               for name in dec_parts[0]}
        return result + (dec,)
    return result


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as a function so tests can check flags without running anything."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix-layers", type=int, default=4,
                    help="N-layer prefix; must match the run that produced --load")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-windows", type=int, default=8,
                    help="held-out windows (seed 999, wikitext) -- same as stage_eval")
    ap.add_argument("--worst", type=int, default=16,
                    help="how many of the worst per-token KLDs to record with their "
                         "window/position, so the tail is inspectable")
    ap.add_argument("--measure-topk", type=int, default=0,
                    help="also report how much teacher mass a top-k of this size "
                         "holds, against the full-vocab normaliser. 0 = off. This "
                         "is what decides whether a sampled residual-mass tail term "
                         "is worth building, and it is only computable here, where "
                         "the full teacher pass already runs.")
    ap.add_argument("--decompose-topk", type=int, default=0,
                    help="also split the full-vocab KLD into the three chain-rule "
                         "pieces (marginal / support-conditional / tail-conditional) "
                         "over the teacher's top-k. 0 = off. Teacher and student are "
                         "co-resident here, so it is one extra reduction per token, "
                         "and it says which piece the next objective term should "
                         "target (RESEARCH-HANDOFF §7.1a).")
    ap.add_argument("--tail-samples", type=int, default=0,
                    help="with --decompose-topk: estimate the tail-conditional piece "
                         "from this many sampled tail tokens per position (unbiased; "
                         "Sparse Logit Sampling) and report it against the exact "
                         "piece. This validates the estimator before the cache stores "
                         "samples for a D_KL2 loss term. 0 = off.")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=32, help="token chunk for log_softmax")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--quant", choices=["absmean", "lloyd", "catq"], default="lloyd")
    ap.add_argument("--catq-steps", type=int, default=200)
    ap.add_argument("--catq-lr", type=float, default=0.05)
    ap.add_argument("--catq-gamma", type=float, default=0.8)
    ap.add_argument("--catq-s0", type=float, default=30.0)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--branch-quant", choices=["fp32", "g128", "rank"], default="g128")
    ap.add_argument("--branch-target", choices=["moe_out", "attn_out", "both"], default="both")
    ap.add_argument("--branch-gate", choices=["none", "rw"], default="none",
                    help="must match the checkpoint's training --branch-gate: "
                         "'rw' builds the gated branch (Phase C) so a gated "
                         "checkpoint's read_gate/write_gate tensors load")
    ap.add_argument("--load", default="", help="branch checkpoint (omit = uncorrected body)")
    ap.add_argument("--balance", choices=["none", "bias", "quantile", "zloss",
                                          "cb", "cbqb"],
                    default="none",
                    help="must match the checkpoint's training --balance: bias "
                         "arms carry a per-expert bias buffer in the checkpoint, "
                         "and the model needs the same patched gate to load it "
                         "(same --balance-cb-eta / --balance-qb-damp too)")
    ap.add_argument("--balance-cb-eta", type=float, default=0.05,
                    help="CB nudge scale (must match training)")
    ap.add_argument("--balance-qb-damp", type=float, default=1.0,
                    help="quantile step damping (must match training)")
    ap.add_argument("--split", default="wikitext")
    ap.add_argument("--seed", type=int, default=999,
                    help="window seed; 999 is the W1 gate's wikitext seed. Use a "
                         "disjoint seed with --split fineweb for a training-side "
                         "curriculum probe (see moe/curriculum.py)")
    ap.add_argument("--out", default="", help="write the stats JSON here")
    ap.add_argument("--model-dir", default="", help="override $MOE_ARTIFACTS/empero-hf")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    from transformers import AutoConfig, AutoTokenizer

    from olmoe_corrections import load_branch_state
    from olmoe_proxy import windows
    from qwen35_moe_proxy import MODEL, build_student, load_prefix, patch_experts

    model_dir = Path(args.model_dir) if args.model_dir else MODEL
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"no checkpoint at {model_dir}; pass --model-dir or set "
                         f"MOE_ARTIFACTS (the FP empero-hf shards are needed for the "
                         f"teacher pass)")
    patch_experts(args.group)

    tok = AutoTokenizer.from_pretrained(model_dir)
    cfg = AutoConfig.from_pretrained(model_dir)
    tcfg = getattr(cfg, "text_config", cfg)
    vocab = int(tcfg.vocab_size)
    data = windows(tok, args.eval_windows, args.seq, args.seed, args.split)
    print(f"KLD eval: {args.prefix_layers}-layer prefix, vocab {vocab}, "
          f"{len(data)} windows x {args.seq} tokens, quant={args.quant} "
          f"branch={args.branch_target}/{args.branch_quant} "
          f"gate={args.branch_gate} load={args.load or 'none'}",
          flush=True)
    # Refuse a teacher cache that cannot coexist with the prefix reload.  The
    # parked log-probs live in host RAM through the student pass, and load_prefix
    # materialises the prefix in fp32 on the host (~17 GB for 4 layers) first --
    # 16 windows OOM'd the 30 GB box at 00:20 (session 3); 8 windows is the
    # tested-safe size.
    need = sum(d.numel() - 1 for d in data) * vocab * 4
    host_memory_guard(need)

    # ---- teacher: FP prefix, banks untouched, no branches
    model, _, _, _ = load_prefix(args.prefix_layers, args.device, model_dir=model_dir)
    tcache, t_ent, t_top, raw_top = teacher_pass(model, data, args.device, args.chunk,
                                                 vocab, top_k=args.measure_topk)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- student: same prefix, ternarised banks + branches + checkpoint
    model, _, _, _ = load_prefix(args.prefix_layers, args.device, model_dir=model_dir)
    if args.balance != "none":
        # Same patched gate the training arm used, so the saved balance_bias
        # buffer exists and loads.  Default none leaves the instrument untouched.
        from router_bias import patch_router_balance
        n_gates = patch_router_balance(model, args.balance,
                                       args.balance_cb_eta, args.balance_qb_damp)
        print(f"router balance: {args.balance} on {n_gates} gates", flush=True)
    n_tr = build_student(model, args)
    print(f"student built: trainable {n_tr/1e6:.2f}M", flush=True)
    if args.load:
        missing, unexpected = load_branch_state(model, args.load)
        # an MTP head in the checkpoint is not part of the KLD instrument
        unexpected = [k for k in unexpected if not k.startswith("_mtp_head.")]
        # The checkpoint only ever holds branch/router tensors, so
        # load_state_dict reports the whole frozen body as "missing" -- that is
        # expected and not a signal.  What matters is that every branch/gate
        # tensor this model *wants* actually came from the file.
        want = {k for k in model.state_dict() if ".branch." in k or ".gate." in k}
        got = want & set(missing)
        print(f"loaded {args.load}: branch tensors {len(want) - len(got)}/{len(want)}, "
              f"unexpected={len(unexpected)} (frozen body keys in `missing` "
              f"({len(missing)}) are expected)", flush=True)
        if got or unexpected:
            raise SystemExit(
                f"checkpoint does not match this prefix/placement: "
                f"{len(got)} wanted branch/router tensor(s) did not load, "
                f"{len(unexpected)} did not belong. Re-run with the same "
                f"--prefix-layers/--branch-target/--rank as training.")
    if args.decompose_topk:
        per_token, agree, ent, top1p, dec = student_pass(
            model, data, args.device, tcache, args.chunk, args.decompose_topk,
            args.tail_samples)
    else:
        if args.tail_samples:
            raise SystemExit("--tail-samples needs --decompose-topk (the exact "
                             "tail piece it is validated against)")
        per_token, agree, ent, top1p = student_pass(
            model, data, args.device, tcache, args.chunk)
        dec = None

    res = tail_stats(per_token)
    if not res["p999_resolved"]:
        print(f"WARNING: at {res['n_tokens']} tokens, p99.9 is only the "
              f"{res['p999_rank']}-worst token -- treat 'max' and 'worst_tokens' "
              f"as the tail signal, not p99.9. Raise --eval-windows to resolve it.",
              flush=True)
    # the worst individual tokens, with where they came from, so a tail move can
    # be inspected rather than taken on faith
    flat = per_token.reshape(-1).float()
    k = min(args.worst, flat.numel())
    top = torch.topk(flat, k)
    seq = args.seq - 1
    res["worst_tokens"] = [
        {"token": int(t), "window": int(t) // seq, "pos": int(t) % seq,
         "kld": float(v)}
        for v, t in zip(top.values, top.indices)
    ]
    # Entropy and peak mass are what make a KLD number interpretable.  A
    # correction branch that sharpens the student (entropy well below the
    # teacher's) can cut PPL while *raising* full-vocab KLD, because KL charges
    # for every token the teacher spreads mass onto that the student has since
    # crushed.  Without these two numbers, "KLD got worse" is uninterpretable.
    res["teacher_entropy_nats"] = round(float(t_ent), 4)
    res["teacher_top1_prob"] = round(float(t_top), 4)
    res["student_entropy_nats"] = round(float(ent), 4)
    res["student_top1_prob"] = round(float(top1p), 4)
    # a student markedly *sharper* than the teacher has been pushed away from
    # the teacher's distribution, which is the usual reason KLD rises while PPL
    # falls; say so in the record rather than leaving a bare number
    res["sharper_than_teacher"] = bool(ent < t_ent - 0.5)
    if raw_top:
        # Coverage of the top-k the caches were built with, against the teacher's
        # full-vocab normaliser. This is the number that decides whether a
        # sampled residual-mass tail term is worth building.
        cov = _coverage(tcache, raw_top, args.measure_topk)
        res["topk_coverage"] = cov
        print(f"top-{args.measure_topk} mass: mean {cov['mean_topk_mass']:.8f} "
              f"min {cov['min_topk_mass']:.8f} "
              f"residual {cov['mean_residual_mass']:.3e}", flush=True)
    if dec is not None:
        # The chain-rule split of the gate's own metric: which piece of the
        # full-vocab KLD does the next objective term need to target?  Shares
        # are of the mean; the pieces sum to the full KLD per token.
        pieces = ("marginal", "support", "tail")
        total_mean = float(dec["full"].mean())
        res["kld_decomposition"] = {
            "top_k": args.decompose_topk,
            "mean_full": total_mean,
            "pieces": {
                name: {
                    "mean": float(dec[name].mean()),
                    "share": float(dec[name].mean() / total_mean) if total_mean else None,
                    "max": float(dec[name].max()),
                } for name in pieces
            },
            "mean_support_mass_teacher": float(dec["w_t"].mean()),
            "mean_support_mass_student": float(dec["w_s"].mean()),
        }
        for name in pieces:
            p = res["kld_decomposition"]["pieces"][name]
            print(f"KLD piece {name:8}: mean {p['mean']:.4f} "
                  f"({p['share']:.1%} of {total_mean:.4f}) max {p['max']:.4f}",
                  flush=True)
        print(f"support mass: teacher {dec['w_t'].mean():.4f} "
              f"student {dec['w_s'].mean():.4f}", flush=True)
        if "tail_est" in dec:
            est = float(dec["tail_est"].mean())
            exact = float(dec["tail"].mean())
            res["kld_decomposition"]["tail_estimate"] = {
                "samples": args.tail_samples,
                "mean": est,
                "exact_mean": exact,
                "ratio": (est / exact) if exact else None,
            }
            print(f"tail estimate: {est:.4f} from {args.tail_samples} samples "
                  f"vs exact {exact:.4f} (ratio {est / exact:.3f})", flush=True)
    res.update({
        "top1_agreement": round(agree, 6),
        "checkpoint": args.load,
        "prefix_layers": args.prefix_layers,
        "quant": args.quant,
        "branch_target": args.branch_target,
        "branch_quant": args.branch_quant,
        "branch_gate": args.branch_gate,
        "rank": args.rank,
        "eval_windows": args.eval_windows,
        "seq": args.seq,
    })
    print(json.dumps(res, indent=2), flush=True)
    out = Path(args.out) if args.out else OUT / "kld-eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

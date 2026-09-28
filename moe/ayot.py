"""AYOT-style calibration plumbing (CPU-side pieces).

Source: ScaleQ-1.58 / AYOT, arXiv 2608.01078 (Intel Labs China).  The finding:
ternary PTQ of a reasoning LLM collapses on reasoning tasks unless the
calibration input distribution contains the model's own reasoning traces.
Recipe shape: generate traces with the FP teacher on calibration questions,
mix them into the calibration rows (~10% of rows), ternarize as usual.

The GPU pieces (trace generation in ``ayot_gen.py``, cache/train in
``qwen35_moe_proxy.py``) produce and consume a JSONL of traces; this module
holds the packing so both are CPU-tested and ready to run.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def windows_from_texts(tok, texts, n: int, seq: int, seed: int = 0) -> torch.Tensor:
    """``n`` token windows of length ``seq`` sampled from a list of strings."""
    if not texts:
        raise ValueError("no texts")
    ids = tok("\n\n".join(texts), return_tensors="pt").input_ids[0]
    if ids.numel() < seq + 1:
        raise ValueError(f"text too short: {ids.numel()} tokens < {seq + 1}")
    rng = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - seq - 1, (n,), generator=rng)
    return torch.stack([ids[s:s + seq] for s in starts])


def mix_windows(general: torch.Tensor, agentic: torch.Tensor,
                frac: float = 0.1, seed: int = 0) -> torch.Tensor:
    """Replace ``round(frac * n)`` rows of ``general`` with agentic windows."""
    if not 0.0 <= frac <= 1.0:
        raise ValueError("frac must be in [0, 1]")
    n = general.shape[0]
    n_agentic = int(round(n * frac))
    if agentic.shape[0] < n_agentic:
        raise ValueError(f"need {n_agentic} agentic windows, have {agentic.shape[0]}")
    out = general.clone()
    if n_agentic:
        rng = torch.Generator().manual_seed(seed)
        rows = torch.randperm(n, generator=rng)[:n_agentic]
        picks = torch.randint(0, agentic.shape[0], (n_agentic,), generator=rng)
        out[rows] = agentic[picks]
    return out


def _load_jsonl(path) -> list[dict]:
    recs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def load_traces(path) -> list[str]:
    """Read a JSONL of traces (``{"trace": ...}`` or ``{"text": ...}``)."""
    texts = [rec.get("trace") or rec.get("text")
             for rec in _load_jsonl(path)]
    texts = [t for t in texts if t]
    if not texts:
        raise ValueError(f"no traces in {path}")
    return texts


def load_prompts(path) -> list[str]:
    """Read a JSONL of calibration questions (question/prompt/text keys)."""
    texts = [rec.get("question") or rec.get("prompt") or rec.get("text")
             for rec in _load_jsonl(path)]
    texts = [t for t in texts if t]
    if not texts:
        raise ValueError(f"no prompts in {path}")
    return texts

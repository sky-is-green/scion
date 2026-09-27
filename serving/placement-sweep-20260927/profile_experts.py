"""Expert-hit concentration profile for OLMoE (placement sweep, 2026-09-27).

Question: if individual experts could be pinned in VRAM (Strata/FreeToken style),
how much of the routing mass would a per-layer top-K cache actually cover, and
how many bytes is that in the deployed Q1_0_g128 container?

Runs the HF checkpoint on one GPU, reads router logits, counts top-k hits per
(layer, expert) over a slice of wiki.test.raw. Output: expert-hits.json + log.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

MODEL = "/home/penis/Desktop/work/hivebench/artifacts/ternary/moe/olmoe-hf"
CORPUS = "/home/penis/Desktop/work/ternary-serve/wiki.test.raw"
OUT = Path("/home/penis/Desktop/work/ternary-serve/placement-sweep-20260927")
TOKENS = 16384          # 32 x 512-token chunks
CHUNK = 512
K_HITS = (4, 8, 12, 16, 24, 32, 48, 64)
BPW = 2.125             # Q1_0_g128

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    torch.manual_seed(0)
    log(f"loading {MODEL} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda:0",
            output_router_logits=True)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0",
            output_router_logits=True)
    model.eval()
    log(f"loaded in {time.time()-t0:.0f}s | device: {next(model.parameters()).device}")

    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts
    top_k = model.config.num_experts_per_tok
    log(f"layers={n_layers} experts={n_experts} top_k={top_k}")

    text = Path(CORPUS).read_text(errors="ignore")
    ids = tok(text, return_tensors="pt").input_ids
    n_chunks = min(TOKENS // CHUNK, ids.shape[1] // CHUNK)
    log(f"profiling {n_chunks} chunks x {CHUNK} tokens of {CORPUS}")

    counts = torch.zeros(n_layers, n_experts, dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        for c in range(n_chunks):
            batch = ids[:, c * CHUNK:(c + 1) * CHUNK].to(model.device)
            out = model(batch, output_router_logits=True)
            rl = getattr(out, "router_logits", None)
            if rl is None:
                log("ERROR: router_logits unavailable")
                return 2
            for li, logits in enumerate(rl):
                top = logits.topk(top_k, dim=-1).indices            # [B*T, k]
                counts[li] += torch.bincount(top.reshape(-1).cpu(),
                                             minlength=n_experts)
            if (c + 1) % 8 == 0:
                log(f"  {c+1}/{n_chunks} chunks ({time.time()-t0:.0f}s)")
    total_hits = int(counts.sum())

    per_expert_params = 3 * 2048 * 1024          # gate+up+down for OLMoE
    bytes_per_expert = per_expert_params * BPW / 8.0

    coverage = {}
    for k in K_HITS:
        per_layer = []
        for li in range(n_layers):
            row = counts[li]
            keep = row.topk(k).indices
            per_layer.append(float(row[keep].sum()) / max(int(row.sum()), 1))
        coverage[k] = {
            "mean": sum(per_layer) / len(per_layer),
            "min": min(per_layer),
            "max": max(per_layer),
            "vram_mib": k * bytes_per_expert * n_layers / 1024 / 1024,
        }

    result = {
        "model": MODEL,
        "corpus": CORPUS,
        "chunks": n_chunks, "tokens": n_chunks * CHUNK,
        "layers": n_layers, "experts": n_experts, "top_k": top_k,
        "total_hits": total_hits,
        "bytes_per_expert_q1_0_g128": bytes_per_expert,
        "coverage_by_top_k": coverage,
        "counts": counts.tolist(),
    }
    (OUT / "expert-hits.json").write_text(json.dumps(result, indent=1))

    log("\ntop-K expert cache coverage (mean over layers):")
    log(f"  {'K':>3} {'cache MiB':>10} {'mean':>7} {'min':>7} {'max':>7}")
    for k, c in coverage.items():
        log(f"  {k:>3} {c['vram_mib']:>10.1f} {100*c['mean']:>6.1f}% "
            f"{100*c['min']:>6.1f}% {100*c['max']:>6.1f}%")
    log(f"\nwrote {OUT / 'expert-hits.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

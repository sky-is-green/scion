"""Per-linear input Hessians for the V2 GPTQ path (Clef f16 body).

Runs the f16 Clef model (streamed, bf16, recurrent GDN so the statistics match
the CPU deployment) over wikitext windows and accumulates ``H = XᵀX/N`` for
every tensor V2 ternarises (176 rotated + 24 unrotated ``ssm_out``).  Hooks
flush one layer at a time to disk, so host memory stays bounded (~1 GB peak per
layer instead of ~30 GB for all 200 Hessians).

Outputs ``<out>/<gguf_tensor_name>.hessian.npy`` (fp32, input basis = primal;
the converter rotates them with ``run_quant.rotate_hessian`` for the folded
weights) and a manifest with token counts and provenance.

Usage:
    .venv-rocm/bin/python dense/clef_v2_hessians.py \
        --out .../v2/hessians --windows 48 --seq 512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WORKSPACE = Path("/home/penis/Desktop/work")
sys.path.insert(0, str(WORKSPACE / "bonsai2-ternary-forensics"))

from clef_dense_load import load_text_model_streamed, patch_recurrent_gdn  # noqa: E402
from clef_cache import generic_windows  # noqa: E402
import clef_head as H  # noqa: E402

TARGET_SUFFIXES = {
    # linear-attention (GDN) layers
    "attn_qkv.weight": "linear_attn.in_proj_qkv",
    "attn_gate.weight": "linear_attn.in_proj_z",
    "ssm_out.weight": "linear_attn.out_proj",
    # full-attention layers
    "attn_q.weight": "self_attn.q_proj",
    "attn_k.weight": "self_attn.k_proj",
    "attn_v.weight": "self_attn.v_proj",
    "attn_output.weight": "self_attn.o_proj",
    # every layer
    "ffn_gate.weight": "mlp.gate_proj",
    "ffn_up.weight": "mlp.up_proj",
    "ffn_down.weight": "mlp.down_proj",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="/home/penis/Desktop/work/models/clef-flash-ternary")
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=48)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer-limit", type=int, default=0, help="debug: first N layers")
    ap.add_argument("--group", type=int, default=4, help="layers captured per pass")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = H.load_tokenizer(args.model)
    windows = generic_windows(tok, args.windows, args.seq, seed=args.seed)
    print(f"corpus: {len(windows)} windows x {args.seq} tokens", flush=True)

    print("loading f16 body (bf16, recurrent GDN) ...", flush=True)
    model, _, loaded = load_text_model_streamed(
        Path(args.model) / "clef-flash-f16.gguf", device=args.device,
        dtype=torch.bfloat16)
    patch_recurrent_gdn(model)
    model.config.use_cache = False
    model.eval()
    print(f"model loaded: {loaded} params", flush=True)

    n_layers = args.layer_limit or model.config.num_hidden_layers
    modules = dict(model.named_modules())
    target_modules: dict[str, list[str]] = {}
    for i in range(n_layers):
        layer_type = getattr(model.layers[i], "layer_type", "")
        for gguf_suffix, hf_suffix in TARGET_SUFFIXES.items():
            if gguf_suffix in ("attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"):
                if layer_type != "linear_attention":
                    continue
                module = f"layers.{i}.{hf_suffix}"
            elif gguf_suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight",
                                 "attn_output.weight"):
                if layer_type != "full_attention":
                    continue
                module = f"layers.{i}.{hf_suffix}"
            else:
                module = f"layers.{i}.{hf_suffix}"
            if module not in modules:
                raise SystemExit(f"module not found: {module} (layer_type {layer_type!r})")
            target_modules[f"blk.{i}.{gguf_suffix}"] = (module, i)
    print(f"capturing {len(target_modules)} target tensors over {n_layers} layers",
          flush=True)

    saved: dict[str, int] = {}
    t0 = time.time()

    # Layer-group passes: hooks only on one group per pass, accumulators persist
    # across all windows for that group, then flush to disk and free.  A single
    # streaming pass would bound memory but reset the sums every forward.
    for g0 in range(0, n_layers, args.group):
        g1 = min(g0 + args.group, n_layers)
        acc: dict[str, torch.Tensor] = {}
        count: dict[str, int] = {}

        def make_hook(name: str):
            def hook(_module, inputs, _output):
                x = inputs[0].detach()
                if x.ndim > 2:
                    x = x.reshape(-1, x.shape[-1])
                x = x.to(dtype=torch.float32, device="cpu")
                if name not in acc:
                    acc[name] = torch.zeros((x.shape[1], x.shape[1]),
                                            dtype=torch.float32)
                    count[name] = 0
                acc[name] += x.transpose(0, 1) @ x
                count[name] += x.shape[0]
            return hook

        handles = []
        for name, (module, layer) in target_modules.items():
            if g0 <= layer < g1:
                handles.append(modules[module].register_forward_hook(make_hook(name)))
        try:
            with torch.no_grad():
                for w, ids in enumerate(windows):
                    inp = torch.tensor([ids], dtype=torch.long, device=args.device)
                    model(input_ids=inp, attention_mask=torch.ones_like(inp),
                          use_cache=False)
        finally:
            for h in handles:
                h.remove()
        for name, a in acc.items():
            h = (a / count[name]).numpy()
            if h.shape[0] != h.shape[1]:
                raise SystemExit(f"{name}: non-square Hessian {h.shape}")
            np.save(out / f"{name}.hessian.npy", h.astype(np.float32))
            saved[name] = count[name]
        print(f"  layers {g0}-{g1-1}: {len(acc)} tensors, {len(windows)} windows "
              f"({time.time()-t0:.0f}s)", flush=True)
        del acc, count

    missing = set(target_modules) - set(saved)
    if missing:
        raise SystemExit(f"captured {len(saved)}/{len(target_modules)}; missing "
                         f"{len(missing)} e.g. {sorted(missing)[:3]}")
    manifest = {
        "model": str(Path(args.model) / "clef-flash-f16.gguf"),
        "dtype": "bfloat16", "gdn": "recurrent", "windows": len(windows),
        "seq": args.seq, "seed": args.seed,
        "tensors": {name: {"tokens": saved[name]} for name in sorted(saved)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(saved)} Hessians to {out} ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

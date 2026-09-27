"""Materialise the ternary qwen3.5-MoE prefix to an HF dir for AUTOGRID scans.

Streams tensors from the local shards straight into a bf16 model (about one
model copy of host RAM), quantises the fused expert banks with the deployable
Lloyd g128 rule, and saves a standard safetensors checkpoint plus tokenizer.

Run it alone (no training concurrently) and under a memory cap:
  systemd-run --user --scope -p MemoryMax=14G ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from olmoe_corrections import quantize_bank_inplace  # noqa: E402
from qwen35_moe_proxy import MODEL, text_layers  # noqa: E402

ART = Path(os.environ.get("MOE_ARTIFACTS", HERE / "artifacts"))


def load_prefix_streaming(n_layers: int):
    """Build the prefix model and copy local shard tensors in one at a time."""
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel
    from safetensors import safe_open

    cfg = AutoConfig.from_pretrained(MODEL)
    tcfg = cfg.text_config
    tcfg.num_hidden_layers = n_layers
    tcfg.layer_types = list(tcfg.layer_types)[:n_layers]
    if hasattr(tcfg, "mtp_num_hidden_layers"):
        tcfg.mtp_num_hidden_layers = 0
    model = Qwen3_5MoeTextModel(tcfg)
    model.to(dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL)

    idx = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    lm_shard = idx.get("lm_head.weight")
    if lm_shard and (MODEL / lm_shard).exists():
        with safe_open(MODEL / lm_shard, framework="pt") as f:
            out_features = f.get_slice("lm_head.weight").get_shape()[0]
        model.lm_head = torch.nn.Linear(tcfg.hidden_size, out_features, bias=False)
        model.lm_head.to(dtype=torch.bfloat16)

    sd = model.state_dict()
    handles: dict[str, object] = {}
    loaded = skipped = 0
    with torch.no_grad():
        for key, shard in idx.items():
            if key.startswith("model.language_model."):
                stripped = key[len("model.language_model."):]
                if stripped.startswith("layers."):
                    if int(stripped.split(".")[1]) >= n_layers:
                        continue
                elif stripped not in ("embed_tokens.weight", "norm.weight"):
                    continue
            elif key == "lm_head.weight":
                stripped = "lm_head.weight"
            else:
                continue
            path = MODEL / shard
            if not path.exists():
                skipped += 1
                continue
            if shard not in handles:
                handles[shard] = safe_open(path, framework="pt", device="cpu")
            t = handles[shard].get_tensor(key)
            if stripped in sd:
                sd[stripped].copy_(t)
                loaded += 1
            del t
    print(f"streamed {loaded} tensors ({skipped} skipped, shard(s) missing)",
          flush=True)

    final_norm = getattr(model, "norm", None)
    if final_norm is not None and final_norm.weight.abs().sum().item() == 0:
        with torch.no_grad():
            final_norm.weight.fill_(1.0)
    model.eval()
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = Path(args.out) if args.out else ART / "qwen35-prefix-ternary-hf"
    model, tok = load_prefix_streaming(args.layers)
    for i, layer in enumerate(text_layers(model)):
        experts = layer.mlp.experts
        quantize_bank_inplace(experts.gate_up_proj, args.group, kind="lloyd")
        quantize_bank_inplace(experts.down_proj, args.group, kind="lloyd")
        print(f"quantised layer {i}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"saved ternary prefix to {out}", flush=True)


if __name__ == "__main__":
    main()

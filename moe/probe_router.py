"""E1 — MoE router-sensitivity probe (prefix, local).

Loads embedding + the first N layers of a Qwen3.5/3.6-MoE checkpoint from local
safetensors shards (no full-model residency), then compares an FP forward pass
with a rotate+absmean-RTN ternary pass over the routed experts.

Measures, per layer:
  - router top-k agreement (fraction of shared experts among the 8 chosen)
  - hidden-state relative L2 drift
  - expert weight relative error (rotated basis)

Usage:
  HIP_VISIBLE_DEVICES=1 python probe_router.py \
      --model-dir $MOE_ARTIFACTS/empero-hf --layers 4 --windows 4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
ART = Path(os.environ.get("MOE_ARTIFACTS", HERE / "artifacts"))
sys.path.insert(0, str(HERE.parent))          # public repo root

from scion_moe import rotation as R  # noqa: E402
from scion_moe.quant import quantize_rtn_absmean  # noqa: E402


def load_prefix(model_dir: Path, n_layers: int, device: str):
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

    cfg = AutoConfig.from_pretrained(model_dir)
    tcfg = cfg.text_config
    tcfg.num_hidden_layers = n_layers
    tcfg.layer_types = list(tcfg.layer_types)[:n_layers]
    if hasattr(tcfg, "mtp_num_hidden_layers"):
        tcfg.mtp_num_hidden_layers = 0
    model = Qwen3_5MoeTextModel(tcfg).to(dtype=torch.bfloat16, device=device)
    tok = AutoTokenizer.from_pretrained(model_dir)

    import json as _json
    idx = _json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}
    state = {}
    for key, shard in idx.items():
        if not key.startswith("model.language_model."):
            continue
        stripped = key[len("model.language_model."):]
        if stripped.startswith("layers."):
            if int(stripped.split(".")[1]) >= n_layers:
                continue
        elif stripped not in ("embed_tokens.weight",):
            continue
        h = handles.get(shard)
        if h is None:
            h = handles[shard] = safe_open(model_dir / shard, framework="pt", device="cpu")
        state[stripped] = h.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return model, tok, missing, unexpected


def windows_from_wikitext(tok, n_windows: int, seq_len: int, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(ids) - seq_len - 1, size=n_windows)
    return torch.stack([ids[s:s + seq_len] for s in starts])


def rotate_quant_restore(w: torch.Tensor, seed: int, group: int = 128,
                         chunk: int = 16):
    """Rotate the last (input) axis, absmean-RTN, restore basis.

    Processes `chunk` rows at a time so peak host RAM stays bounded
    (the archive's helpers upcast to float64; a full [256,1024,2048] expert
    tensor in float64 is ~4.3 GB per temporary).

    Returns (w_effective, rel_err, zero_share).
    """
    w_np = w.detach().to(torch.float32).cpu().numpy()
    width = w_np.shape[-1]
    rots = R.rotations_for(width, seed)
    out = np.empty_like(w_np)
    err2 = 0.0
    ref2 = 0.0
    zeros = 0
    total = 0
    for s in range(0, w_np.shape[0], chunk):
        part = w_np[s:s + chunk]
        w_rot = R.absorb_input(part, rots)
        q = quantize_rtn_absmean(w_rot, group_size=group)
        w_hat_rot = q.dequantize()
        w_hat = R.unabsorb_input(w_hat_rot, rots)
        out[s:s + chunk] = w_hat
        err2 += float(np.sum((w_hat - part) ** 2))
        ref2 += float(np.sum(part ** 2))
        zeros += int((q.codes == 0).sum())
        total += q.codes.size
        del w_rot, q, w_hat_rot, w_hat, part
        gc.collect()
    del w_np
    gc.collect()
    rel = float(np.sqrt(err2 / (ref2 + 1e-12)))
    return torch.from_numpy(out.astype(np.float32)), rel, zeros / max(total, 1)


@torch.no_grad()
def run(model, ids, device, hooks_out):
    return model(input_ids=ids.to(device), output_hidden_states=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=16,
                    help="experts per quantization slice (host-RAM bound)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ART / "e1-router-probe.json"))
    ap.add_argument("--attn-too", action="store_true",
                    help="also ternarize attention/GDN linears (arm B)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    model, tok, missing, unexpected = load_prefix(model_dir, args.layers, args.device)
    print(f"loaded prefix: {args.layers} layers; missing={len(missing)} unexpected={len(unexpected)}")

    ids = windows_from_wikitext(tok, args.windows, args.seq_len, args.seed)

    def capture():
        """Run forward, return (hidden_states per layer, router top-8 per layer)."""
        rec = {}
        handles = []
        for i, layer in enumerate(model.layers):
            def hook(mod, inp, out, i=i):
                rec[i] = out[2].detach().cpu() if isinstance(out, tuple) else None
            handles.append(layer.mlp.gate.register_forward_hook(hook))
        with torch.no_grad():
            out = model(input_ids=ids.to(args.device), output_hidden_states=True,
                        use_cache=False)
        for h in handles:
            h.remove()
        hs = [h.detach().float().cpu() for h in out.hidden_states]
        return hs, rec

    fp_hs, fp_route = capture()
    print("FP pass done")

    # --- ternary arm: quantize routed experts in place ----------------------
    report = {"layers": args.layers, "windows": args.windows, "seq_len": args.seq_len,
              "seed": args.seed, "group": args.group, "attn_too": args.attn_too,
              "experts": [], "per_layer": []}
    for i, layer in enumerate(model.layers):
        for name in ("gate_up_proj", "down_proj"):
            p = getattr(layer.mlp.experts, name)
            seed_off = 0 if name == "gate_up_proj" else 1
            w_hat, rel, zero = rotate_quant_restore(p.data, args.seed + i * 17 + seed_off,
                                                    args.group, args.chunk)
            p.data = w_hat.to(p.dtype).to(p.device)
            report["experts"].append({"layer": i, "param": name, "rel_err": round(rel, 5),
                                      "zero_share": round(zero, 4)})
            print(f"  L{i} {name}: rel_err {rel:.4f} zero_share {zero:.3f}")

    if args.attn_too:
        for i, layer in enumerate(model.layers):
            for modname in ("linear_attn", "self_attn"):
                mod = getattr(layer, modname, None)
                if mod is None:
                    continue
                for pname, p in list(mod.named_parameters()):
                    if p.dim() != 2 or pname.endswith("norm.weight"):
                        continue
                    w_hat, rel, _ = rotate_quant_restore(p.data, args.seed + i * 31, args.group)
                    p.data = w_hat.to(p.dtype).to(p.device)

    tern_hs, tern_route = capture()
    print("ternary pass done")

    # --- metrics ------------------------------------------------------------
    for i in range(args.layers):
        a, b = fp_route.get(i), tern_route.get(i)
        if a is None or b is None:
            continue
        inter = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean().item()
        hs_rel = float((tern_hs[i + 1] - fp_hs[i + 1]).norm() /
                       (fp_hs[i + 1].norm() + 1e-12))
        report["per_layer"].append({
            "layer": i,
            "router_topk_agreement": round(inter, 5),
            "hidden_rel_l2": round(hs_rel, 5),
        })
        print(f"  L{i}: router agree {inter:.4f}  hidden rel L2 {hs_rel:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

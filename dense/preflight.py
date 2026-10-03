"""Rental/local preflight: prove the training environment matches the deployment.

Run this as the *first* action on any new machine, before a single training
step.  It builds the f16 backbone from the GGUF with the reverse loader, patches
the recurrent GDN, verifies the fast-path libraries are absent (we deliberately
use the pure-torch recurrent form), runs a fixed token probe, and requires the
post-norm hidden states to match a reference captured on the local CPU bridge.

Exit code is non-zero on any mismatch, so a guardrail script can abort.

    python dense/preflight.py \
        --gguf .../clef-flash-f16.gguf \
        --tokens /tmp/opencode/tokens.txt \
        --reference /tmp/opencode/f16_cpu.npy \
        --min-cos 0.999
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clef_dense_load import load_text_model_streamed, patch_recurrent_gdn  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as M  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokens", required=True, help="space-separated token ids")
    ap.add_argument("--reference", default="", help="reference .npy [T,4096]")
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="float32")
    args = ap.parse_args()

    print(f"torch {torch.__version__}  cuda/hip {torch.version.cuda or torch.version.hip}")
    print(f"fast path available: {M.is_fast_path_available} (must be False)")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA/ROCm device visible")
    free, total = torch.cuda.mem_get_info()
    print(f"device {torch.cuda.get_device_name(0)}: free {free/1e9:.1f}/{total/1e9:.1f} GB")
    if M.is_fast_path_available:
        print("FAIL: fla/causal_conv1d are installed; the forward will not match "
              "the CPU deployment. Uninstall them.")
        return 2

    dt = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model, _, loaded = load_text_model_streamed(args.gguf, device=args.device, dtype=dt)
    n = patch_recurrent_gdn(model)
    model.eval()
    ids = torch.tensor([list(map(int, Path(args.tokens).read_text().split()))],
                       device=args.device)
    with torch.no_grad():
        hs = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   use_cache=False).last_hidden_state[0].float().cpu().numpy()
    print(f"loaded {loaded} params, patched {n} GDN; hidden {hs.shape}")

    if args.reference:
        try:
            ref = np.load(args.reference)
        except Exception:
            ref = np.fromfile(args.reference, dtype=np.float32)
        if ref.shape != hs.shape:
            ref = ref.reshape(hs.shape)
        cos = float((hs * ref).sum() / (np.linalg.norm(hs) * np.linalg.norm(ref)))
        per = (hs * ref).sum(1) / (np.linalg.norm(hs, axis=1) * np.linalg.norm(ref, axis=1) + 1e-12)
        print(f"cos vs reference: {cos:.5f} (min {args.min_cos})")
        print(f"per-token cos: first8 {[round(float(x),4) for x in per[:8]]} "
              f"min {float(per.min()):.4f} at {int(per.argmin())}/{len(per)}")
        if cos < args.min_cos:
            print("FAIL: forward does not match the deployment reference")
            return 3
    print("PREFLIGHT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export a trained MTP head to the sidecar GGUF the runtime will load.

The sidecar carries only what the runtime cannot get from the target itself:

  mtp.fc1.weight  [width, 2*hidden]   (GGUF ne = [2*hidden, width])
  mtp.fc2.weight  [hidden, width]
  metadata: format/version/hidden/width/source + optional acceptance numbers

The head's output goes through the *target's own* output_norm + output
tensors, and its inputs are the target's post-norm hidden + the token
embedding, so those are deliberately NOT duplicated here.

Verification (`--verify`) reloads the GGUF, reconstructs the head and checks a
forward against the source checkpoint on random inputs -- the round-trip
guard for the dim convention.

Usage:
  python moe/mtp_sidecar_export.py --head $MOE/qwen35/mtp-release-head-v2.pt \
    --out $MOE/qwen35/mtp-sidecar-v2.gguf --source qwen35-release \
    --accept-fineweb 0.483 --accept-wikitext 0.360 --verify
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from scion_paths import GGUF_PY


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="checkpoint with _mtp_head.*")
    ap.add_argument("--out", required=True, help="sidecar .gguf to write")
    ap.add_argument("--gguf-py", default=str(GGUF_PY))
    ap.add_argument("--source", default="", help="release/model id for metadata")
    ap.add_argument("--accept-fineweb", type=float, default=None)
    ap.add_argument("--accept-wikitext", type=float, default=None)
    ap.add_argument("--verify", action="store_true",
                    help="reload the sidecar and check a forward vs the source")
    return ap


def head_forward(fc1, fc2, h, e):
    """Mirror of mtp.MTPHead (rms inputs, gelu, no norm/head here)."""
    def rms(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    x = torch.cat([rms(h), rms(e)], dim=-1)
    return (fc2 @ torch.nn.functional.gelu(fc1 @ x.T)).T


def main() -> None:
    args = build_parser().parse_args()
    import sys
    sys.path.insert(0, args.gguf_py)
    from gguf import GGUFWriter

    sd = torch.load(args.head, map_location="cpu")
    fc1 = sd["_mtp_head.fc1.weight"].float().numpy()
    fc2 = sd["_mtp_head.fc2.weight"].float().numpy()
    width, two_h = fc1.shape
    hidden = fc2.shape[0]
    if fc2.shape[1] != width or two_h != 2 * hidden:
        raise SystemExit(f"head shape mismatch: fc1 {fc1.shape}, fc2 {fc2.shape}")

    w = GGUFWriter(args.out, arch="mtp")
    w.add_string("mtp.format", "scion-mtp-head")
    w.add_uint32("mtp.version", 1)
    w.add_uint32("mtp.hidden_size", int(hidden))
    w.add_uint32("mtp.width", int(width))
    if args.source:
        w.add_string("mtp.source", args.source)
    if args.accept_fineweb is not None:
        w.add_float32("mtp.acceptance.fineweb", float(args.accept_fineweb))
    if args.accept_wikitext is not None:
        w.add_float32("mtp.acceptance.wikitext", float(args.accept_wikitext))
    # fp16 keeps the sidecar ~50 MB at the current width; gguf-py reverses the
    # numpy shape, so GGUF ne = [2*hidden, width] for fc1 as the loader expects.
    w.add_tensor("mtp.fc1.weight", fc1.astype(np.float16))
    w.add_tensor("mtp.fc2.weight", fc2.astype(np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}: hidden {hidden}, width {width}")

    if args.verify:
        from gguf import GGUFReader
        r = GGUFReader(args.out)
        got = {}
        for t in r.tensors:
            # gguf-py reverses dims on write, so the reader hands arrays back
            # in the original torch layout [out, in] -- no transpose needed.
            got[t.name] = torch.from_numpy(np.asarray(t.data).copy()).float()
        torch.manual_seed(0)
        h = torch.randn(4, hidden)
        e = torch.randn(4, hidden)
        ref = head_forward(torch.from_numpy(fc1), torch.from_numpy(fc2), h, e)
        out = head_forward(got["mtp.fc1.weight"], got["mtp.fc2.weight"], h, e)
        md = float((ref - out).abs().max())
        rel = md / max(float(ref.abs().max()), 1e-6)
        print(f"verify: max abs diff {md:.3e} (rel {rel:.2e})")
        if rel > 1e-3:
            raise SystemExit("round-trip mismatch -- dim convention is wrong")
        print("verify OK")


if __name__ == "__main__":
    main()

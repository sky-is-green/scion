#!/usr/bin/env python3
"""Turn strata `--dump-final-r` dumps into fork-form MTP probe files.

strata's generate --dump-final-r appends, per processed position, the residual
after the last layer: hc x n_embd floats (hc = 4 streams x 2560 = 10240) - the
strata MTP drafter's input.  The fork's sidecar lane instead consumes the
LM-head input, which the fork's build_hc_mix computes from that residual:

    xn    = rms_norm_4streams(r) * w_norm            # eps = 1e-6
    lo    = silu(w_down @ xn / hc)                   # [320]
    gate  = sigmoid(w_up @ lo)                       # [10240]
    gated = xn * gate
    mixed = mean_over_streams(gated)                 # [2560]  == t_embd
    argmax_t = argmax(lm_head @ mixed)

The weights come from the Qwen/Qwen3.8-Flash-Next BF16 checkpoint (fetched by
mtp_fetch.py / a range read into --tensors), not from the body GGUF: the
Q6_K dequant path isn't implemented in gguf-py, and BF16 is the authoritative
form of the same tensors (the Q6_K noise shows up as ~1e-3 differences that
the --validate step quantifies against the fork probe's ground-truth windows).

Usage:
  python v2/scripts/mixdown_probe.py --tensors DIR --dump finalr.bin \
      --seq 512 --n-windows 1104 --out v2/traces/mtp-probe [--windows-from 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

HC = 4
N_EMBD = 2560
EPS = 1e-6
ROW_FLOATS = HC * N_EMBD


def bf16(path: Path) -> torch.Tensor:
    raw = np.fromfile(path, dtype="<f2")
    return torch.from_numpy(raw.astype(np.float32))


def load_weights(tdir: Path) -> dict[str, torch.Tensor]:
    head = bf16(tdir / "lm_head.weight.bin").reshape(248320, N_EMBD).to(torch.float16).cuda()
    w_norm = bf16(tdir / "mtp.hyper_connection_mixer.hc_norm.weight.bin").reshape(ROW_FLOATS).cuda()
    w_down = bf16(tdir / "mtp.hyper_connection_mixer.input_mix_weight_down.weight.bin").reshape(320, ROW_FLOATS).cuda()
    w_up = bf16(tdir / "mtp.hyper_connection_mixer.input_mix_weight_up.weight.bin").reshape(ROW_FLOATS, 320).cuda()
    for name, t in (("head", head), ("w_norm", w_norm), ("w_down", w_down), ("w_up", w_up)):
        assert torch.isfinite(t.float()).all(), f"{name} has non-finite values"
    return {"head": head, "w_norm": w_norm, "w_down": w_down, "w_up": w_up}


def read_window(dump, w: int, seq: int, n_recs: int):
    """Rows of window w from the posrec dump: [seq, 10240] fp32 + argmax int32
    (the model's sampled token at that position = the teacher-forced token) +
    the positions, aligned by the records' explicit pos field."""
    rec_floats = 4 + ROW_FLOATS
    start = w * seq
    recs = np.asarray(dump[start * rec_floats:(start + seq) * rec_floats]).reshape(seq, rec_floats)
    pos = recs[:, 0].astype(np.int64)
    rows = recs[:, 4:].copy()
    am = recs[:, 1].astype(np.int32)
    return rows, am, pos


def mix_rows(rows: torch.Tensor, W: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """rows [n, 10240] fp32 -> (mixed [n, 2560] fp32, argmax [n] int64)."""
    x = rows.view(-1, HC, N_EMBD)
    xn = x / torch.sqrt((x * x).mean(-1, keepdim=True) + EPS)     # per-stream RMS
    xn = (xn * W["w_norm"].view(1, HC, N_EMBD)).reshape(-1, ROW_FLOATS)
    lo = torch.nn.functional.silu(xn @ W["w_down"].T / HC)
    gate = torch.sigmoid(lo @ W["w_up"].T)
    gated = (xn * gate).view(-1, HC, N_EMBD)
    mixed = gated.mean(dim=1)
    logits = mixed.to(torch.float16) @ W["head"].T                # cublas accumulates fp32
    return mixed, logits.argmax(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n-windows", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows-from", type=int, default=0)
    ap.add_argument("--validate", type=int, default=0,
                    help="compare the first N windows against fork-probe ground truth")
    args = ap.parse_args()

    W = load_weights(Path(args.tensors))
    print("mixer weights loaded (BF16 -> fp32/fp16 on GPU)", flush=True)

    dump = np.memmap(args.dump, dtype="<f4", mode="r")
    # records: {pos int64, tok int64, 10240 floats} - 40,976 B each; rows land by
    # their explicit pos, so a missing position 0 (the one batched-prefill row)
    # simply leaves a hole that windows fill with a mirror of the next row.
    rec_floats = 4 + ROW_FLOATS
    n_recs = len(dump) // rec_floats
    print(f"dump: {len(dump)/1e9:.2f} GB -> {n_recs:,} position records", flush=True)
    use = min(args.n_windows, n_recs // args.seq) - args.windows_from

    if args.validate:
        for w in range(args.validate):
            rows, am, pos = read_window(dump, w, args.seq, n_recs)
            ref_h = np.fromfile(Path(args.out) / f"win_{w:02d}_h.bin", dtype="<f4").reshape(args.seq, N_EMBD)
            ref_a = np.fromfile(Path(args.out) / f"win_{w:02d}_argmax.bin", dtype="<i4")
            ref = torch.from_numpy(ref_h).cuda()
            cos = torch.nn.functional.cosine_similarity(mixed_rows(rows, W), ref, dim=-1).mean()
            agree = (am.astype(np.int64) == ref_a.astype(np.int64)).mean()
            print(f"window {w}: cos {cos.item():.6f} | argmax agree {agree:.4f} | "
                  f"max|h| ref {np.abs(ref_h).max():.3f}", flush=True)
        return 0

    out = Path(args.out)
    for w in range(args.windows_from, args.windows_from + use):
        rows, am, pos = read_window(dump, w, args.seq, n_recs)
        t = torch.from_numpy(rows).cuda()
        mixed, am_t = mix_rows(t, W)
        h = mixed.cpu().numpy().astype("<f4")
        if pos[0] != w * args.seq:            # row 0 missing (batched-prefill row)
            h[0] = h[1]
            am[0] = am[1]
        h.tofile(out / f"win_{w:02d}_h.bin")
        am.astype("<i4").tofile(out / f"win_{w:02d}_argmax.bin")
        if (w - args.windows_from) % 50 == 0:
            print(f"  window {w} (pos {pos[0]}..{pos[-1]})", flush=True)
    print(f"wrote {use} windows to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

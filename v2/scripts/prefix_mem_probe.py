#!/usr/bin/env python3
"""4-layer prefix memory probe for the FlashNext P2 port (PLAN 13a step A).

Runs the REAL ``stage_train`` path (load_prefix -> --bank-gguf seed ->
build_student -> balance -> one training step) at a chosen window length and
configuration, and reports peak VRAM, host high-water RSS and wall time.  The
cache is synthetic (``build_trace_cache.py`` record schema, top-512 + tail-64,
random values) so the LM/KD numbers are meaningless; the point is the memory
and the step time at the recipe geometry (rank 512, both placements, bias
balance, tail weights on).

Usage (one condition per process, so allocator peaks are independent):
  python v2/scripts/prefix_mem_probe.py --seq 4096 --mode compact --ckpt \
      --group-size 64 --device cuda:1

Conditions:
  --mode dense     v1 path: banks seeded then quantised in place (bf16 dense)
  --mode compact   P2 port: banks held as deployed codes+scales (from the GGUF)
  --ckpt           add --grad-checkpoint --checkpoint-mode group --expert-group-size N
  --chunked-ce     add --chunked-ce (vocab-side softmax bound)
Output: one JSON line "PROBE {...}" on stdout; exit 3 on OOM (also JSON).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "moe"))


def host_rss_kib() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return -1


def build_cache(seq: int, top_k: int = 512, tail: int = 64) -> Path:
    """A synthetic cache in the build_trace_cache.py record schema."""
    out = Path(os.environ.get("MOE_PROBE_DIR", "/tmp/opencode/qwen35-memprobe"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"cache-seq{seq}-top{top_k}-tail{tail}.pt"
    if path.exists():
        return path
    g = torch.Generator().manual_seed(1001 + seq)
    rec = {
        "ids": torch.randint(240000, 248000, (seq,), dtype=torch.int32,
                             generator=g),
        "idx": torch.stack([torch.randperm(248320, generator=g)[:top_k]
                            for _ in range(seq - 1)]).reshape(1, seq - 1, top_k
                                                              ).to(torch.int32),
        "val": (-torch.rand(1, seq - 1, top_k, generator=g) * 8).to(torch.float16),
        "w": torch.rand(1, seq - 1, generator=g).to(torch.float16),
        "tidx": torch.randint(0, 248320, (1, seq - 1, tail), generator=g,
                              dtype=torch.int32),
        "tlp": (-torch.rand(1, seq - 1, tail, generator=g) * 12).to(torch.float16),
    }
    torch.save([rec], path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--mode", choices=["dense", "compact"], required=True)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--chunked-ce", action="store_true")
    ap.add_argument("--no-tail", action="store_true",
                    help="kd-tail-weight 0 / kd-tailcond-weight 0 (isolate the "
                         "tail terms' memory share; the recipe has them on)")
    ap.add_argument("--snapshot", metavar="PATH",
                    help="record the allocator memory history and dump a "
                         "snapshot to PATH on OOM (attribution only)")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--rank", type=int, default=512)
    args_cli = ap.parse_args()

    import qwen35_moe_proxy as q35

    cache = build_cache(args_cli.seq)
    argv = [
        "train", "--prefix-layers", "4",
        "--device", args_cli.device, "--device-map", args_cli.device,
        "--cache-file", str(cache),
        "--seq", str(args_cli.seq),
        "--quant", "lloyd", "--branch-quant", "g128", "--branch-target", "both",
        "--rank", str(args_cli.rank),
        "--bank-gguf", os.environ.get(
            "SCION_GGUF",
            "models-p310/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf"),
        "--kd-weight", "2.0", "--kd-tail-weight", "2.0",
        "--kd-tailcond-weight", "3.0", "--temp", "2.0",
        "--balance", "bias",
        "--steps", str(args_cli.steps), "--log-every", "1", "--ckpt-every", "0",
        "--tag", "memprobe",
    ]
    if args_cli.mode == "compact":
        argv += ["--compact-banks"]
    if args_cli.ckpt:
        argv += ["--grad-checkpoint", "--checkpoint-mode", "group",
                 "--expert-group-size", str(args_cli.group_size)]
    if args_cli.chunked_ce:
        argv += ["--chunked-ce"]
    if args_cli.no_tail:
        argv += ["--kd-tail-weight", "0", "--kd-tailcond-weight", "0"]

    args = q35.build_parser().parse_args(argv)
    cond = (f"{args_cli.mode}{'+ckpt' if args_cli.ckpt else ''}"
            f"{'+g'+str(args_cli.group_size) if args_cli.ckpt else ''}"
            f"{'+chunkedce' if args_cli.chunked_ce else ''}")
    summary = {"condition": cond, "seq": args_cli.seq, "steps": args_cli.steps,
               "rank": args_cli.rank}
    if torch.cuda.is_available():
        torch.cuda.set_device(args_cli.device)   # init the ROCm context first
        torch.zeros(1, device=args_cli.device)
        torch.cuda.reset_peak_memory_stats(args_cli.device)
    if args_cli.snapshot:
        torch.cuda.memory._record_memory_history(max_entries=200_000)
    rss0 = host_rss_kib()
    t0 = time.time()
    try:
        q35.stage_train(args)
    except torch.OutOfMemoryError as e:
        summary.update({"oom": True, "error": str(e).splitlines()[0],
                        "peak_gib": round(torch.cuda.max_memory_allocated(
                            args_cli.device) / 2**30, 3) if torch.cuda.is_available() else None,
                        "wall_s": round(time.time() - t0, 1)})
        if args_cli.snapshot and torch.cuda.is_available():
            try:
                torch.cuda.memory._record_memory_history()  # stop recording
                torch.cuda.memory._dump_snapshot(args_cli.snapshot)
                summary["snapshot"] = args_cli.snapshot
            except Exception as se:      # attribution must not mask the OOM
                summary["snapshot_error"] = repr(se)
        print("PROBE " + json.dumps(summary), flush=True)
        raise SystemExit(3)
    summary.update({
        "oom": False,
        "peak_gib": round(torch.cuda.max_memory_allocated(args_cli.device) / 2**30, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(args_cli.device) / 2**30, 3),
        "host_hwm_gib": round(host_rss_kib() / 2**20, 2),
        "host_delta_gib": round((host_rss_kib() - rss0) / 2**20, 2),
        "wall_s": round(time.time() - t0, 1),
    })
    print("PROBE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

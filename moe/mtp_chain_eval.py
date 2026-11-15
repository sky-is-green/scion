"""Chained draft acceptance for the released MTP head, on the GPU probe windows.

Measures what the llama.cpp draft-mtp lane will actually do: the head drafts a
token from (target h row, candidate token embedding), and after an acceptance
the chain continues from the *draft's own* fc3 state row, exactly like the
lane's `t_h_nextn` chain.

For each start position k in an eval window:
  state = h[k]                 (the target's head input at position k)
  cand  = ids[k+1]             (the real token that follows k)
  for j = 1..depth:
      logits, state_next = head(state, e(cand))
      draft = argmax(logits)
      accept if draft == ids[k+1+j]
      if reject: stop
      state, cand = state_next, draft

Reports per-depth acceptance, the empirical accepted-drafts-per-round, and the
ideal tokens per target step (1 + accepted drafts).

Usage:
  python moe/mtp_chain_eval.py --head v2/mtp-drafter-release/mtp-release-head-gpu.pt \
      --gguf models-p310/flashnext/Scion-FlashNext-176B-A6B-Q6_K.gguf \
      --probe-dir v2/traces/mtp-probe-gpu --depth 4 --n-windows 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtp_release_train import load_gguf_tensor, rms  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="release head checkpoint (fc1/fc2/fc3)")
    ap.add_argument("--gguf", required=True, help="the target GGUF (token_embd/output)")
    ap.add_argument("--probe-dir", required=True, help="win_XX_h.bin + tokens.bin")
    ap.add_argument("--gguf-py", default="")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n-windows", type=int, default=100,
                    help="number of eval windows (the trainer used the last 100)")
    ap.add_argument("--start-window", type=int, default=-1,
                    help="first window (default: the tail, so the eval matches "
                         "--n-train 1000 / --n-eval 100)")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp16", action="store_true", default=True)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.gguf_py:
        sys.path.insert(0, args.gguf_py)
    dev = args.device
    root = Path(args.probe_dir)

    sd = torch.load(args.head, map_location="cpu")
    fc1 = sd["_mtp_head.fc1.weight"].float().to(dev)
    fc2 = sd["_mtp_head.fc2.weight"].float().to(dev)
    fc3 = sd["_mtp_head.fc3.weight"].float().to(dev)

    dtype = torch.float16 if args.fp16 else torch.float32
    tok = load_gguf_tensor(args.gguf, "token_embd.weight").to(dev, dtype)
    out = load_gguf_tensor(args.gguf, "output.weight").to(dev, dtype)
    try:
        norm_w = load_gguf_tensor(args.gguf, "output_norm.weight").to(dev, dtype)
    except KeyError:
        norm_w = None
    fc1 = fc1.to(dtype)
    fc2 = fc2.to(dtype)
    fc3 = fc3.to(dtype)
    hidden = tok.shape[1]
    print(f"hidden {hidden}, vocab {tok.shape[0]}, norm {norm_w is not None}", flush=True)

    ids = np.fromfile(root / "tokens.bin", dtype=np.int32).reshape(-1, args.seq)
    n_all = ids.shape[0]
    n_eval = min(args.n_windows, n_all)
    start = args.start_window if args.start_window >= 0 else n_all - n_eval
    start = max(0, min(start, n_all - n_eval))
    win_list = list(range(start, start + n_eval))
    print(f"eval windows {start}..{start + n_eval - 1} of {n_all}", flush=True)
    hs = [np.memmap(root / f"h/win_{w:02d}_h.bin", dtype=np.float32, mode="r",
                    shape=(args.seq, hidden)) for w in range(n_all)]

    def head_forward(h, e):
        hf = h.to(dtype)
        x = torch.cat([rms(hf), rms(e)], dim=-1)
        t = F.gelu(fc1 @ x.T)
        z = fc2 @ t
        zt = z.T
        lg = (rms(zt, norm_w) @ out.T).float()
        st = (fc3 @ t).T.float()
        return lg, st

    depth = args.depth
    # counters per depth: accepted at this depth (given the previous accepted)
    acc = np.zeros(depth, dtype=np.float64)
    tot = np.zeros(depth, dtype=np.float64)
    rounds = 0
    accepted_total = 0

    with torch.no_grad():
        for w in win_list:
            hw = torch.from_numpy(np.asarray(hs[w])).to(dev)
            tw = torch.from_numpy(ids[w].astype(np.int64)).to(dev)
            # rows k where a full chain fits: need ids[k+1+depth]
            n_k = args.seq - 2 - depth
            if n_k <= 0:
                continue
            state = hw[:n_k]
            cand = tok[tw[1:1 + n_k]]
            alive = torch.ones(n_k, dtype=torch.bool, device=dev)
            hits = torch.zeros(n_k, dtype=torch.long, device=dev)
            for j in range(depth):
                lg, st_next = head_forward(state, cand)
                draft = lg.argmax(-1)
                target = tw[1 + j + 1:1 + j + 1 + n_k]
                hit = (draft == target)
                new = alive & hit
                acc[j] += new.sum().item()
                tot[j] += alive.sum().item()
                hits += new.long()
                alive = new
                if not alive.any():
                    break
                state = st_next
                cand = tok[draft]
            rounds += n_k
            accepted_total += hits.sum().item()

    print(f"windows {n_eval}, starts/round {rounds}, depth {depth}", flush=True)
    for j in range(depth):
        p = acc[j] / tot[j] if tot[j] else float("nan")
        print(f"  P(accept @ draft {j + 1} | previous accepted) = {p:.4f}  "
              f"(n={int(tot[j])})", flush=True)
    # tail probability: product of conditionals = P(at least j accepted)
    tail = []
    p = 1.0
    for j in range(depth):
        p *= acc[j] / tot[j] if tot[j] else 0.0
        tail.append(p)
    print("  P(at least j drafts accepted): " + " ".join(f"{x:.4f}" for x in tail), flush=True)
    mean_acc = accepted_total / rounds if rounds else float("nan")
    print(f"  mean accepted drafts per round = {mean_acc:.4f}", flush=True)
    print(f"  ideal tokens per target step   = {1.0 + mean_acc:.4f}", flush=True)
    print(f"  draft-steps per accepted token = {1.0 / (1.0 + mean_acc):.4f}", flush=True)


if __name__ == "__main__":
    main()

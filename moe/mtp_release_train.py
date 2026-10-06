"""Train the release drafter (MTP head) on probe-collected release hiddens.

The prefix-trained head does not transfer to the released model (different
hidden geometry: 0.157 acceptance on the release vs 0.561 on the 4-layer
prefix), so the drafter is trained here directly on the release's own
post-norm hiddens, collected by the fork's `test-mtp-probe` tool:

  inputs : win_XX_h.bin   fp32 [seq, n_embd]   post-norm hidden per position
           tokens.bin     int32 [n_windows, seq] (train windows first, eval last)
  targets: win_XX_argmax.bin int32 [seq]       the main model's greedy token
  frozen : the release's own token_embd / output_norm / output (from the GGUF)
  trained: the head (`fc1`/`gelu`/`fc2`), self-distill to the main's choices.

Usage:
  python moe/mtp_release_train.py --probe-dir $MOE/qwen35/mtp-release \
    --gguf /path/to/qwen35-release.gguf --n-train 1024 --n-eval 16 \
    --steps 4096 --out $MOE/qwen35/mtp-release-head.pt --report ...json

CPU-testable helpers (`head_forward`, `acceptance`) have no model dependency.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scion_paths import GGUF_PY


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True,
                    help="dir with win_XX_h.bin / win_XX_argmax.bin / tokens.bin")
    ap.add_argument("--gguf", required=True,
                    help="the released model (token_embd/output_norm/output)")
    ap.add_argument("--gguf-py", default=str(GGUF_PY),
                    help="path to the fork's gguf-py (for GGUF dequant)")
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--layers", type=int, choices=(1, 2), default=2)
    ap.add_argument("--width-mult", type=float, default=1.0,
                    help="hidden width of the fc1/fc2 intermediate, as a multiple "
                         "of 2H (capacity knob; 1.0 = the prefix head's width)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=4096)
    ap.add_argument("--batch-windows", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-train-n", type=int, default=0,
                    help="also report acceptance on the first N *train* windows "
                         "(diagnoses overfitting; 0 = off)")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; evaluate --init on the eval (+ train) "
                         "windows and exit")
    ap.add_argument("--n-eval-fineweb", type=int, default=0,
                    help="of the last --n-eval windows, this many are held-out "
                         "train-distribution (fineweb) windows, reported "
                         "separately from the wikitext eval")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--init", default="", help="warm-start head checkpoint")
    ap.add_argument("--out", default="",
                    help="head checkpoint to write (required for training)")
    ap.add_argument("--report", default="")
    return ap


# ---------------------------------------------------------------- gguf io ---

def load_gguf_tensor(path: str, name: str) -> torch.Tensor:
    """Dequantized fp32 tensor [out, in] from a GGUF (F32/Q8_0/... via gguf-py)."""
    try:
        from gguf import GGUFReader
        from gguf.quants import dequantize
    except ModuleNotFoundError as e:
        raise SystemExit(
            "gguf-py is not importable; pass --gguf-py <fork>/gguf-py "
            f"(original error: {e})")

    r = GGUFReader(path)
    for t in r.tensors:
        if t.name == name:
            if t.tensor_type.name == "F32":
                return torch.from_numpy(np.asarray(t.data).copy()).float()
            d = dequantize(np.asarray(t.data), t.tensor_type)
            return torch.from_numpy(np.asarray(d).copy()).float()
    raise KeyError(name)


# ------------------------------------------------------------- head math ----

def rms(x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    return y if w is None else y * w


def head_forward(fc1: torch.Tensor, fc2: torch.Tensor, norm_w: torch.Tensor,
                 output: torch.Tensor, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Draft logits for hidden ``h`` [N, H] and next-token embedding ``e`` [N, H]."""
    x = torch.cat([rms(h), rms(e)], dim=-1)
    z = fc2 @ F.gelu(fc1 @ x.T)                 # [H, N]
    return (rms(z.T, norm_w) @ output.T)        # [N, vocab]


def acceptance(draft_logits: torch.Tensor, main_argmax: torch.Tensor,
               topk: int = 1) -> float:
    """Fraction of positions where the main token is in the draft's top-k."""
    if topk <= 1:
        hit = draft_logits.argmax(-1) == main_argmax
    else:
        hit = (draft_logits.topk(topk, dim=-1).indices
               == main_argmax.unsqueeze(-1)).any(-1)
    return float(hit.float().mean())


# ------------------------------------------------------------------ main ----

def main() -> None:
    args = build_parser().parse_args()
    if args.gguf_py:
        sys.path.insert(0, args.gguf_py)
    dev = args.device
    root = Path(args.probe_dir)

    print("loading release tensors ...", flush=True)
    tok_embd = load_gguf_tensor(args.gguf, "token_embd.weight")   # [vocab, H]
    output = load_gguf_tensor(args.gguf, "output.weight")         # [vocab, H]
    norm_w = load_gguf_tensor(args.gguf, "output_norm.weight")    # [H]
    hidden = tok_embd.shape[1]
    vocab = tok_embd.shape[0]
    print(f"hidden {hidden}, vocab {vocab}", flush=True)

    ids = np.fromfile(root / "tokens.bin", dtype=np.int32).reshape(-1, args.seq)
    n_all = ids.shape[0]
    if args.n_train + args.n_eval > n_all:
        raise SystemExit(f"tokens.bin has {n_all} windows, need "
                         f"{args.n_train + args.n_eval}")
    hs = [np.memmap(root / f"h/win_{w:02d}_h.bin", dtype=np.float32, mode="r",
                    shape=(args.seq, hidden)) for w in range(n_all)]
    am = [np.fromfile(root / f"h/win_{w:02d}_argmax.bin", dtype=np.int32)
          for w in range(n_all)]

    torch.manual_seed(0)
    w1 = int(round(args.width_mult * 2 * hidden))
    fc1 = torch.empty(w1, 2 * hidden).normal_(0, 0.02)
    fc2 = torch.empty(hidden, w1).normal_(0, 0.02)
    if args.layers != 2:
        raise SystemExit("only --layers 2 is implemented for the release trainer")
    if args.init:
        sd = torch.load(args.init, map_location="cpu")
        fc1 = sd["_mtp_head.fc1.weight"].float()
        fc2 = sd["_mtp_head.fc2.weight"].float()
        print(f"warm start from {args.init}", flush=True)
    fc1 = torch.nn.Parameter(fc1.to(dev))
    fc2 = torch.nn.Parameter(fc2.to(dev))
    opt = torch.optim.AdamW([fc1, fc2], lr=args.lr,
                            weight_decay=args.weight_decay)

    tok_gpu = tok_embd.to(dev)
    out_gpu = output.to(dev)
    norm_gpu = norm_w.to(dev)

    def batch(windows):
        h = torch.from_numpy(np.concatenate(
            [np.asarray(hs[w])[:-2] for w in windows])).to(dev)
        e = tok_gpu[torch.from_numpy(np.concatenate(
            [ids[w][1:-1].astype(np.int64) for w in windows])).to(dev)]
        tgt = torch.from_numpy(np.concatenate(
            [am[w][1:-1].astype(np.int64) for w in windows])).to(dev)
        return h, e, tgt

    def loss_and_acc(h, e, tgt, want_acc=False):
        lg = head_forward(fc1, fc2, norm_gpu, out_gpu, h, e)
        n = lg.shape[0]
        loss = 0.0
        for i in range(0, n, args.chunk):
            loss = loss + F.cross_entropy(lg[i:i + args.chunk], tgt[i:i + args.chunk],
                                          reduction="sum")
        loss = loss / n
        if not want_acc:
            return loss, None
        return loss, acceptance(lg, tgt, topk=1)

    def run_eval(win_list):
        accs = []
        with torch.no_grad():
            for w in win_list:
                h, e, tgt = batch([w])
                _, acc = loss_and_acc(h, e, tgt, want_acc=True)
                accs.append(acc)
        return sum(accs) / len(accs)

    def eval_split():
        """(wikitext eval, held-out fineweb) acceptance over the eval range."""
        k = args.n_eval_fineweb
        fw = run_eval(range(args.n_train, args.n_train + k)) if k else None
        wt = run_eval(range(args.n_train + k, args.n_train + args.n_eval))
        return wt, fw

    if args.eval_only:
        if not args.init:
            raise SystemExit("--eval-only needs --init <head checkpoint>")
        a, af = eval_split()
        at = run_eval(range(min(args.eval_train_n, args.n_train))) \
            if args.eval_train_n else float("nan")
        print(f"eval-only: eval acceptance {a:.4f}"
              + ("" if af is None else f" / fineweb-heldout {af:.4f}")
              + (f" / train acceptance {at:.4f}" if at == at else ""), flush=True)
        if args.report:
            Path(args.report).write_text(json.dumps(
                {"eval_only": True, "init": args.init, "eval_accept": round(a, 4),
                 "fineweb_accept": None if af is None else round(af, 4),
                 "train_accept": None if at != at else round(at, 4)}, indent=2) + "\n")
        return

    t0 = time.time()
    curve = []
    step = 0
    if not args.out:
        raise SystemExit("--out is required for training")
    while step < args.steps:
        wl = [(step * args.batch_windows + j) % args.n_train
              for j in range(args.batch_windows)]
        h, e, tgt = batch(wl)
        loss, _ = loss_and_acc(h, e, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        if args.eval_every and step % args.eval_every == 0:
            a, af = eval_split()
            at = run_eval(range(min(args.eval_train_n, args.n_train))) \
                if args.eval_train_n else None
            curve.append({"step": step, "accept": round(a, 4),
                          "accept_fineweb": None if af is None else round(af, 4),
                          "accept_train": None if at is None else round(at, 4)})
            print(f"  [eval] step {step} release acceptance {a:.4f}"
                  + ("" if af is None else f" (fineweb-heldout {af:.4f})")
                  + ("" if at is None else f" (train {at:.4f})"), flush=True)

    final, final_fw = eval_split()

    sd = {"_mtp_head.fc1.weight": fc1.detach().cpu().float(),
          "_mtp_head.fc2.weight": fc2.detach().cpu().float()}
    torch.save(sd, args.out)
    report = {"out": args.out, "gguf": args.gguf, "probe_dir": str(root),
              "n_train": args.n_train, "n_eval": args.n_eval,
              "n_eval_fineweb": args.n_eval_fineweb,
              "steps": args.steps, "lr": args.lr, "layers": args.layers,
              "width_mult": args.width_mult, "weight_decay": args.weight_decay,
              "final_accept": round(final, 4),
              "final_fineweb": None if final_fw is None else round(final_fw, 4),
              "curve": curve,
              "wall_s": round(time.time() - t0, 1)}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(f"final release acceptance {final:.4f}; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

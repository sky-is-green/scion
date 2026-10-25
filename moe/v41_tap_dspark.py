#!/usr/bin/env python3
"""Train a V4.1 DSpark head on full 3 x 5120 target taps.

This is the tap-projection variant of `mtp_dspark.py`: the probe stores all
three target-layer hiddens, a trainable linear maps 3H->H, and the existing
GRU/DSpark head consumes the projected hidden.  The target output head stays
frozen, exactly as in the k=1/DSpark trainers.

Probe layout:
  tokens.bin                    int32 [windows, seq]
  h/win_XXXXX_h.bin             float32 [seq, 3*5120]
  h/win_XXXXX_argmax.bin        int32 [seq]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtp_dspark import DSparkHead, ce_chunked, head_logits, ideal_tps  # noqa: E402
from mtp_release_train import load_gguf_tensor  # noqa: E402


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--probe-dir", required=True)
        p.add_argument("--gguf", required=True)
        p.add_argument("--gguf-py", default="")
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--seq", type=int, default=128)
        p.add_argument("--tap-dim", type=int, default=3 * 5120)
        p.add_argument("--n-train", type=int, default=0)
        p.add_argument("--n-eval", type=int, default=16)
        p.add_argument("--state-dim", type=int, default=0)
        p.add_argument("--init", default="")
    t = sub.choices["train"]
    t.add_argument("--k", type=int, default=3)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--batch-windows", type=int, default=1)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--out", default="")
    t.add_argument("--report", default="")
    e = sub.choices["eval"]
    e.add_argument("--head", required=True)
    e.add_argument("--k", type=int, default=3)
    return ap


def load_am(root: Path, w: int) -> np.ndarray:
    return np.fromfile(root / "h" / f"win_{w:05d}_argmax.bin", dtype=np.int32)


def make_tap_proj(tap_dim: int, hidden: int, device: str, init: str = "") -> nn.Linear:
    p = nn.Linear(tap_dim, hidden, bias=False).to(device)
    if init:
        p.load_state_dict(torch.load(init, map_location="cpu")["tap_proj"])
        return p
    with torch.no_grad():
        eye = torch.eye(hidden, dtype=p.weight.dtype, device=p.weight.device)
        p.weight.copy_(torch.cat([eye / 3.0] * (tap_dim // hidden), dim=1))
    return p


def cmd_train(a) -> None:
    dev = a.device
    root = Path(a.probe_dir)
    tok = load_gguf_tensor(a.gguf, "token_embd.weight").to(dev)
    output = load_gguf_tensor(a.gguf, "output.weight").to(dev)
    norm = load_gguf_tensor(a.gguf, "output_norm.weight").to(dev)
    H = tok.shape[1]
    tap_dim = a.tap_dim
    ids = np.fromfile(root / "tokens.bin", dtype=np.int32).reshape(-1, a.seq)
    n_all = len(ids)
    n_train = a.n_train or (n_all - a.n_eval)
    if n_train <= 0 or n_train + a.n_eval > n_all:
        raise SystemExit(f"bad split: n_all={n_all}, n_train={n_train}, n_eval={a.n_eval}")
    hs = [np.memmap(root / "h" / f"win_{w:05d}_h.bin", dtype=np.float32, mode="r",
                    shape=(a.seq, tap_dim)) for w in range(n_all)]

    torch.manual_seed(0)
    head = DSparkHead(H, k=a.k, state_dim=a.state_dim or None).to(dev)
    tap = make_tap_proj(tap_dim, H, dev)
    n_par = sum(p.numel() for p in head.parameters()) + sum(p.numel() for p in tap.parameters())
    print(f"tap head hidden {H} taps {tap_dim} k {a.k} "
          f"params {n_par / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(list(head.parameters()) + list(tap.parameters()),
                            lr=a.lr, weight_decay=a.weight_decay)
    rows = a.seq - a.k

    def batch(wl):
        h = torch.from_numpy(np.concatenate([np.asarray(hs[w])[:rows] for w in wl])).to(dev)
        tin = [torch.from_numpy(np.concatenate(
            [ids[w][i:i + rows].astype(np.int64) for w in wl])).to(dev)
            for i in range(1, a.k + 1)]
        tgt = [torch.from_numpy(np.concatenate(
            [load_am(root, w)[i:i + rows].astype(np.int64) for w in wl])).to(dev)
            for i in range(1, a.k + 1)]
        return h, tin, tgt

    def step_loss(h, tin, tgt):
        hr = head.h_rms(tap(h))
        state = head.init_state(h.shape[0], h)
        loss = h.new_zeros(())
        for i in range(a.k):
            state, out = head.step(hr, tok[tin[i]], state, i)
            loss = loss + ce_chunked(head_logits(out, norm, output), tgt[i])
        return loss / a.k

    def eval_windows(wl):
        accs = [[] for _ in range(a.k)]
        with torch.no_grad():
            for w in wl:
                h = torch.from_numpy(np.asarray(hs[w])[:rows]).to(dev)
                hr = head.h_rms(tap(h))
                state = head.init_state(h.shape[0], h)
                prev = torch.from_numpy(ids[w][1:rows + 1].astype(np.int64)).to(dev)
                for i in range(a.k):
                    state, out = head.step(hr, tok[prev], state, i)
                    draft = head_logits(out, norm, output).argmax(-1)
                    tgt = torch.from_numpy(load_am(root, w)[i + 1:i + 1 + rows].astype(np.int64)).to(dev)
                    accs[i].append(float((draft == tgt).float().mean()))
                    prev = draft
        return [float(np.mean(x)) for x in accs]

    t0 = time.time()
    curve = []
    for s in range(1, a.steps + 1):
        wl = [(s * a.batch_windows + j) % n_train for j in range(a.batch_windows)]
        h, tin, tgt = batch(wl)
        loss = step_loss(h, tin, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s % 100 == 0:
            print(f"step {s} loss {loss.item():.4f} ({(time.time()-t0)/s:.2f}s/step)", flush=True)
        if a.eval_every and s % a.eval_every == 0:
            ch = eval_windows(range(n_train, n_train + a.n_eval))
            tp = ideal_tps(ch)
            curve.append({"step": s, "chained": [round(x, 4) for x in ch],
                          "ideal_tps": round(tp, 3)})
            print(f"  [eval] step {s} chained {[round(x,4) for x in ch]} ideal {tp:.3f} t/s",
                  flush=True)
    ch = eval_windows(range(n_train, n_train + a.n_eval))
    tp = ideal_tps(ch)
    sd = {k: v.detach().cpu().float() for k, v in head.state_dict().items()}
    sd["tap_proj.weight"] = tap.weight.detach().cpu().float()
    sd["meta"] = {"hidden": H, "k": a.k, "state_dim": head.state_dim,
                  "tap_dim": tap_dim, "n_train": n_train, "n_eval": a.n_eval}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, a.out)
    rep = {"out": a.out, "k": a.k, "steps": a.steps, "n_train": n_train,
           "n_eval": a.n_eval, "tap_dim": tap_dim, "params": n_par,
           "chained_final": [round(x, 4) for x in ch],
           "ideal_tps_final": round(tp, 3), "wall_s": round(time.time() - t0, 1),
           "curve": curve}
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=2) + "\n")
    print(f"final chained {[round(x,4) for x in ch]} ideal {tp:.3f} t/s; wrote {a.out}", flush=True)


def cmd_eval(a) -> None:
    dev = a.device
    root = Path(a.probe_dir)
    sd = torch.load(a.head, map_location="cpu")
    meta = sd["meta"]
    tok = load_gguf_tensor(a.gguf, "token_embd.weight").to(dev)
    output = load_gguf_tensor(a.gguf, "output.weight").to(dev)
    norm = load_gguf_tensor(a.gguf, "output_norm.weight").to(dev)
    H = tok.shape[1]
    head = DSparkHead(H, k=meta["k"], state_dim=meta["state_dim"]).to(dev)
    tap = make_tap_proj(meta["tap_dim"], H, dev)
    head.load_state_dict({k: v for k, v in sd.items() if k.startswith(("pos", "gru", "proj", "fc"))})
    tap.weight.copy_(sd["tap_proj.weight"].to(dev))
    ids = np.fromfile(root / "tokens.bin", dtype=np.int32).reshape(-1, a.seq)
    hs = [np.memmap(root / "h" / f"win_{w:05d}_h.bin", dtype=np.float32, mode="r",
                    shape=(a.seq, meta["tap_dim"])) for w in range(len(ids))]
    rows = a.seq - meta["k"]
    accs = [[] for _ in range(meta["k"])]
    with torch.no_grad():
        for w in range(len(ids)):
            h = torch.from_numpy(np.asarray(hs[w])[:rows]).to(dev)
            hr = head.h_rms(tap(h))
            state = head.init_state(h.shape[0], h)
            prev = torch.from_numpy(ids[w][1:rows + 1].astype(np.int64)).to(dev)
            for i in range(meta["k"]):
                state, out = head.step(hr, tok[prev], state, i)
                draft = head_logits(out, norm, output).argmax(-1)
                tgt = torch.from_numpy(load_am(root, w)[i + 1:i + 1 + rows].astype(np.int64)).to(dev)
                accs[i].append(float((draft == tgt).float().mean()))
                prev = draft
    per = [float(np.mean(x)) for x in accs]
    print(json.dumps({"head": a.head, "k": meta["k"], "per_position": [round(x, 4) for x in per],
                      "ideal_tokens_per_step": round(ideal_tps(per), 3)}, indent=2))


def main() -> None:
    a = parser().parse_args()
    if a.gguf_py:
        sys.path.insert(0, a.gguf_py)
    (cmd_train if a.cmd == "train" else cmd_eval)(a)


if __name__ == "__main__":
    main()

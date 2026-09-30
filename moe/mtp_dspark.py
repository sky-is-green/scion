"""DSpark-shaped multi-token MTP head (per-position learned recurrent state).

The naive chain reuses the k=1 head with a stale h_t and collapses at position
2+ (measured 0.42/0.14/0.06).  DSpark's insight is that each drafted position
needs its own state, not a re-feed of the same hidden.  This head carries a
GRU state across the drafted block (a learned state per position), conditioned
on the frozen target hidden h_t and the previous token's embedding, and reuses
the target's own output_norm/output (so it adds no vocab projection).

Train (self-distilled to the release's own greedy tokens); eval reports
teacher-forced and chained per-position acceptance + ideal tokens/step.

  python moe/mtp_dspark.py train --probe-dir DIR --gguf REL.gguf --gguf-py GP \
      --n-train 3056 --n-eval 32 --k 3 --steps 2048 --out head.pt --report r.json
  python moe/mtp_dspark.py eval  --probe-dir DIR --gguf REL.gguf --gguf-py GP \
      --head head.pt --k 3
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
from mtp_release_train import load_gguf_tensor, rms  # noqa: E402


# ------------------------------------------------------------------- head ---

class DSparkHead(nn.Module):
    """GRU-carried multi-token head: one learned state per drafted position."""

    def __init__(self, hidden: int, k: int = 3, state_dim: int | None = None,
                 init_std: float = 0.02):
        super().__init__()
        self.hidden = hidden
        self.k = k
        self.state_dim = state_dim or hidden
        self.pos = nn.Parameter(torch.zeros(k, 2 * hidden))
        self.gru = nn.GRUCell(2 * hidden, self.state_dim)
        self.proj = nn.Linear(self.state_dim, hidden, bias=False)
        # residual direct branch (the k=1 MLP form) so position 1 keeps the
        # direct head's capacity and the state only adds the lookahead signal
        self.fc1 = nn.Linear(2 * hidden, 2 * hidden, bias=False)
        self.fc2 = nn.Linear(2 * hidden, hidden, bias=False)
        for w in (self.proj.weight, self.pos, self.fc1.weight, self.fc2.weight):
            nn.init.normal_(w, std=init_std)

    def h_rms(self, h: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(h.float(), (self.hidden,), eps=1e-6)

    def init_state(self, n: int, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros(n, self.state_dim, device=ref.device, dtype=torch.float32)

    def step(self, h_rms: torch.Tensor, e: torch.Tensor, state: torch.Tensor,
             i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([h_rms, F.rms_norm(e.float(), (self.hidden,), eps=1e-6)], -1)
        direct = self.fc2(F.gelu(self.fc1(x)))
        state = self.gru(x + self.pos[i], state)
        return state, self.proj(state) + direct   # [N, H]


# --------------------------------------------------------------- logits io --

def logits_chunks(out: torch.Tensor, norm_w: torch.Tensor, output: torch.Tensor,
                  chunk: int = 64):
    """Yield (row_slice, logits) for out [N,H] through the target norm+output."""
    n = out.shape[0]
    for s in range(0, n, chunk):
        z = out[s:s + chunk].float()
        yield slice(s, s + chunk), rms(z, norm_w) @ output.T


def head_logits(out: torch.Tensor, norm_w, output) -> torch.Tensor:
    """Full-width projection (one GEMM); the k=1 trainer does the same."""
    return rms(out.float(), norm_w) @ output.T


def ce_chunked(lg: torch.Tensor, tgt: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    total = lg.new_zeros(())
    for s in range(0, lg.shape[0], chunk):
        total = total + F.cross_entropy(lg[s:s + chunk], tgt[s:s + chunk], reduction="sum")
    return total / lg.shape[0]


# ---------------------------------------------------------------- loading ---

def ideal_tps(per) -> float:
    tp = 1.0
    cum = 1.0
    for p in per:
        cum *= p
        tp += cum
    return tp


def load_taps(probe: Path, w: int, seq: int, hidden: int) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(np.memmap(probe / f"h/win_{w:02d}_h.bin", dtype=np.float32,
                             mode="r", shape=(seq, hidden)))
    am = np.fromfile(probe / f"h/win_{w:02d}_argmax.bin", dtype=np.int32)
    return h, am


# ----------------------------------------------------------------- train ----

def cmd_train(a):
    dev = a.device
    tok = load_gguf_tensor(a.gguf, "token_embd.weight").to(dev)
    output = load_gguf_tensor(a.gguf, "output.weight").to(dev)
    norm = load_gguf_tensor(a.gguf, "output_norm.weight").to(dev)
    H = tok.shape[1]
    ids = np.fromfile(Path(a.probe_dir) / "tokens.bin", dtype=np.int32).reshape(-1, a.seq)
    n_all = ids.shape[0]
    root = Path(a.probe_dir)
    hs = [np.memmap(root / f"h/win_{w:02d}_h.bin", dtype=np.float32, mode="r",
                    shape=(a.seq, H)) for w in range(n_all)]

    torch.manual_seed(0)
    head = DSparkHead(H, k=a.k, state_dim=a.state_dim).to(dev)
    n_par = sum(p.numel() for p in head.parameters())
    print(f"DSparkHead hidden {H} k {a.k} state {head.state_dim} params {n_par/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    rows = a.seq - a.k            # h_t rows available for a k-position block

    def batch(wl):
        h = torch.from_numpy(np.concatenate([np.asarray(hs[w])[:rows] for w in wl])).to(dev)
        # input token for position i: ids[t+i]; target: am[t+i]
        tin = [torch.from_numpy(np.concatenate(
            [ids[w][i:i + rows].astype(np.int64) for w in wl])).to(dev)
            for i in range(1, a.k + 1)]
        tgt = [torch.from_numpy(np.concatenate(
            [np.fromfile(root / f"h/win_{w:02d}_argmax.bin", dtype=np.int32)[i:i + rows].astype(np.int64)
             for w in wl])).to(dev) for i in range(1, a.k + 1)]
        return h, tin, tgt

    def step_loss(h, tin, tgt, train=True):
        # teacher forcing: position i input is the real token x_{t+i}
        hr = head.h_rms(h)
        state = head.init_state(h.shape[0], h)
        loss = h.new_zeros(())
        for i in range(a.k):
            e = tok[tin[i]]
            state, out = head.step(hr, e, state, i)
            lg = head_logits(out, norm, output)
            loss = loss + ce_chunked(lg, tgt[i])
            del lg
        return loss / a.k

    def eval_windows(win_list):
        """Chained per-position acceptance: each position consumes the prior draft."""
        accs = [[] for _ in range(a.k)]
        with torch.no_grad():
            for w in win_list:
                h = torch.from_numpy(np.asarray(hs[w])[:rows]).to(dev)
                am = np.fromfile(root / f"h/win_{w:02d}_argmax.bin", dtype=np.int32)
                hr = head.h_rms(h)
                state = head.init_state(h.shape[0], h)
                prev = torch.from_numpy(ids[w][1:rows + 1].astype(np.int64)).to(dev)
                for i in range(a.k):
                    state, out = head.step(hr, tok[prev], state, i)
                    lg = head_logits(out, norm, output)
                    draft = lg.argmax(-1)
                    tgt = torch.from_numpy(am[i + 1:i + 1 + rows].astype(np.int64)).to(dev)
                    accs[i].append(float((draft == tgt).float().mean()))
                    prev = draft
        return [float(np.mean(x)) for x in accs]

    t0 = time.time()
    curve = []
    for s in range(1, a.steps + 1):
        wl = [(s * a.batch_windows + j) % a.n_train for j in range(a.batch_windows)]
        h, tin, tgt = batch(wl)
        loss = step_loss(h, tin, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s % 100 == 0:
            print(f"step {s} loss {loss.item():.4f} ({(time.time()-t0)/s:.2f}s/step)", flush=True)
        if a.eval_every and s % a.eval_every == 0:
            ch = eval_windows(range(a.n_train, a.n_train + a.n_eval))
            tp = ideal_tps(ch)
            curve.append({"step": s, "chained": [round(x, 4) for x in ch],
                          "ideal_tps": round(tp, 3)})
            print(f"  [eval] step {s} chained {[round(x,4) for x in ch]} ideal {tp:.3f} t/s", flush=True)

    ch = eval_windows(range(a.n_train, a.n_train + a.n_eval))
    tp = ideal_tps(ch)
    torch.save({"dspark.pos": head.pos.detach().cpu().float(),
                "dspark.gru.weight_ih": head.gru.weight_ih.detach().cpu().float(),
                "dspark.gru.weight_hh": head.gru.weight_hh.detach().cpu().float(),
                "dspark.gru.bias_ih": head.gru.bias_ih.detach().cpu().float(),
                "dspark.gru.bias_hh": head.gru.bias_hh.detach().cpu().float(),
                "dspark.proj.weight": head.proj.weight.detach().cpu().float(),
                "dspark.fc1.weight": head.fc1.weight.detach().cpu().float(),
                "dspark.fc2.weight": head.fc2.weight.detach().cpu().float(),
                "meta": {"hidden": H, "k": a.k, "state_dim": head.state_dim}}, a.out)
    rep = {"out": a.out, "k": a.k, "steps": a.steps, "n_train": a.n_train,
           "chained_final": [round(x, 4) for x in ch],
           "ideal_tps_final": round(tp, 3), "curve": curve,
           "wall_s": round(time.time() - t0, 1), "params": n_par}
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=2) + "\n")
    print(f"final chained {[round(x,4) for x in ch]} ideal {tp:.3f} t/s; wrote {a.out}", flush=True)


def cmd_eval(a):
    dev = a.device
    tok = load_gguf_tensor(a.gguf, "token_embd.weight").to(dev)
    output = load_gguf_tensor(a.gguf, "output.weight").to(dev)
    norm = load_gguf_tensor(a.gguf, "output_norm.weight").to(dev)
    sd = torch.load(a.head, map_location="cpu")
    meta = sd["meta"]
    head = DSparkHead(meta["hidden"], k=meta["k"], state_dim=meta["state_dim"]).to(dev)
    with torch.no_grad():
        head.pos.copy_(sd["dspark.pos"]); head.gru.weight_ih.copy_(sd["dspark.gru.weight_ih"])
        head.gru.weight_hh.copy_(sd["dspark.gru.weight_hh"]); head.gru.bias_ih.copy_(sd["dspark.gru.bias_ih"])
        head.gru.bias_hh.copy_(sd["dspark.gru.bias_hh"]); head.proj.weight.copy_(sd["dspark.proj.weight"])
        if "dspark.fc1.weight" in sd:
            head.fc1.weight.copy_(sd["dspark.fc1.weight"]); head.fc2.weight.copy_(sd["dspark.fc2.weight"])
    a.k = meta["k"]; a.seq = a.seq; a.n_eval = a.n_eval
    # reuse cmd_train's eval by faking the window range
    ids = np.fromfile(Path(a.probe_dir) / "tokens.bin", dtype=np.int32).reshape(-1, a.seq)
    root = Path(a.probe_dir); rows = a.seq - a.k
    ch_acc = [[] for _ in range(a.k)]
    with torch.no_grad():
        for w in range(a.n_train, a.n_train + a.n_eval):
            h = torch.from_numpy(np.asarray(np.memmap(root / f"h/win_{w:02d}_h.bin",
                dtype=np.float32, mode="r", shape=(a.seq, meta["hidden"])))[:rows]).to(dev)
            am = np.fromfile(root / f"h/win_{w:02d}_argmax.bin", dtype=np.int32)
            hr = head.h_rms(h); state = head.init_state(h.shape[0], h)
            prev = torch.from_numpy(ids[w][1:rows + 1].astype(np.int64)).to(dev)
            for i in range(a.k):
                state, out = head.step(hr, tok[prev], state, i)
                lg = head_logits(out, norm, output)
                draft = lg.argmax(-1)
                tgt = torch.from_numpy(am[i + 1:i + 1 + rows].astype(np.int64)).to(dev)
                ch_acc[i].append(float((draft == tgt).float().mean()))
                prev = draft
    per = [float(np.mean(x)) for x in ch_acc]
    tp = 1.0; cum = 1.0
    for p in per:
        cum *= p; tp += cum
    print(json.dumps({"head": a.head, "k": a.k, "per_position": [round(x, 4) for x in per],
                      "ideal_tokens_per_step": round(tp, 3)}, indent=2))


def build() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--probe-dir", required=True)
        p.add_argument("--gguf", required=True)
        p.add_argument("--gguf-py", default="/home/penis/llama.cpp/gguf-py")
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--seq", type=int, default=512)
        p.add_argument("--n-train", type=int, default=3056)
        p.add_argument("--n-eval", type=int, default=32)
        p.add_argument("--state-dim", type=int, default=0)
    t = sub.choices["train"]
    t.add_argument("--k", type=int, default=3)
    t.add_argument("--steps", type=int, default=2048)
    t.add_argument("--batch-windows", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--eval-every", type=int, default=256)
    t.add_argument("--out", default="")
    t.add_argument("--report", default="")
    e = sub.choices["eval"]
    e.add_argument("--head", required=True)
    e.add_argument("--k", type=int, default=3)
    return ap


def main():
    a = build().parse_args()
    if a.gguf_py:
        sys.path.insert(0, a.gguf_py)
    if a.state_dim == 0:
        a.state_dim = None
    (cmd_train if a.cmd == "train" else cmd_eval)(a)


if __name__ == "__main__":
    main()

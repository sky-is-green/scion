"""MoTE-style ternary-MoE proxy on Qwen3-1.7B (local canary base).

Architecture (MoTE, arXiv 2506.14435):
  - the pretrained dense FFN is kept as a FROZEN BF16 shared expert
  - E routed ternary experts (top-1), initialised from the dense FFN
  - router kept in BF16 (trainable)
  - only the ternary experts are trainable
  - Switch-style load-balance aux loss

Stages:
  upcycle : build the MoE from the dense checkpoint and save
  train   : STE QAT of the routed experts (optional KD from the dense teacher)
  eval    : held-out perplexity for dense / upcycled-RTN / trained

Ops: run under `systemd-run --user --scope -p MemoryMax=..` and pin one GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ART = Path(os.environ.get("MOE_ARTIFACTS", HERE / "artifacts"))
BASE = ART / "canary-hf"
OUT = ART / "proxy"


# ---------------------------------------------------------------- quantizer --

def ternary_absmean(w: torch.Tensor, group: int = 128) -> torch.Tensor:
    """Straight-through ternary with per-group absmean scales (Bonsai format)."""
    if group <= 0 or w.shape[-1] % group != 0:
        alpha = w.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        q = torch.clamp(torch.round(w / alpha), -1, 1)
        return q * alpha
    g = w.reshape(*w.shape[:-1], w.shape[-1] // group, group)
    alpha = g.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    q = torch.clamp(torch.round(g / alpha), -1, 1) * alpha
    return q.reshape(w.shape)


def _lloyd_scale(g: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """Lloyd-Max / TWN fixed-point group scale, matching the Q1_0_g128 rule.

    Iterates ``a <- mean(|w| : |w| > a/2)`` from four starts and keeps the one
    with the best residual reduction ``s1^2/sw``.
    """
    best_a = mean.clone()
    best_obj = torch.full_like(mean, -1.0)
    for init in (0.5, 0.7, 0.9, 1.1):
        a = init * mean
        s1 = torch.zeros_like(mean)
        sw = torch.zeros_like(mean)
        for _ in range(8):
            mask = g.abs() > 0.5 * a.unsqueeze(-1)
            s1 = (g.abs() * mask).sum(-1)
            sw = mask.sum(-1).float()
            a = torch.where(sw > 0, s1 / sw, torch.zeros_like(a))
        obj = torch.where(sw > 0, s1 * s1 / sw.clamp_min(1e-9), torch.zeros_like(s1))
        take = obj > best_obj
        best_obj = torch.where(take, obj, best_obj)
        best_a = torch.where(take, a, best_a)
    return torch.where(best_a > 0, best_a, mean)


def ternary_lloyd(w: torch.Tensor, group: int = 128) -> torch.Tensor:
    """Ternary with Lloyd-refined per-group scales (the deployable rule).

    Same storage as ``ternary_absmean`` (2-bit codes + fp16 scale per group),
    but the scale is the fixed point the TAARDIS Q1_0_g128 quantizer uses
    (Lloyd-Max refinement; default-on in that fork).  The fp16 round-trip of
    the scale is reproduced here so training sees exactly the deployed scale.

    Independent torch reimplementation of the rule in the TAARDIS llama.cpp
    fork (MIT; see NOTICE), not a code copy.
    """
    if group <= 0 or w.shape[-1] % group != 0:
        group = w.shape[-1]
    g = w.float().reshape(*w.shape[:-1], w.shape[-1] // group, group)
    mean = g.abs().mean(-1)
    a = _lloyd_scale(g, mean)
    a = a.half().float()
    q = torch.clamp(torch.round(g / a.unsqueeze(-1).clamp_min(1e-12)), -1, 1)
    return (q * a.unsqueeze(-1)).reshape(w.shape).to(w.dtype)


class TernaryLinear(nn.Module):
    """Linear whose weight is ternarised on the fly (STE); weight is the master."""

    def __init__(self, out_features: int, in_features: int, group: int = 128,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        with torch.no_grad():
            wq = ternary_absmean(w, self.group)
        wq = wq + (w - w.detach())          # STE: only the master carries grad
        return F.linear(x, wq)


class ExpertMLP(nn.Module):
    def __init__(self, hidden: int, inter: int, act, group: int = 128,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.gate_proj = TernaryLinear(inter, hidden, group, dtype)
        self.up_proj = TernaryLinear(inter, hidden, group, dtype)
        self.down_proj = TernaryLinear(hidden, inter, group, dtype)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class MoEFFN(nn.Module):
    """Frozen BF16 shared expert + E ternary routed experts + BF16 router."""

    def __init__(self, shared: nn.Module, hidden: int, n_experts: int,
                 act, group: int = 128):
        super().__init__()
        self.shared = shared                      # frozen dense FFN
        for p in self.shared.parameters():
            p.requires_grad_(False)
        self.router = nn.Linear(hidden, n_experts, bias=False,
                                dtype=shared.gate_proj.weight.dtype)
        self.experts = nn.ModuleList()
        for _ in range(n_experts):
            e = ExpertMLP(hidden, shared.intermediate_size, act, group,
                          shared.gate_proj.weight.dtype)
            with torch.no_grad():
                e.gate_proj.weight.copy_(shared.gate_proj.weight)
                e.up_proj.weight.copy_(shared.up_proj.weight)
                e.down_proj.weight.copy_(shared.down_proj.weight)
            self.experts.append(e)
        with torch.no_grad():
            self.router.weight.normal_(0.0, 0.02)

    def forward(self, x: torch.Tensor):
        probs = F.softmax(self.router(x).float(), dim=-1)
        idx = probs.argmax(dim=-1)                        # top-1
        p = probs.gather(-1, idx.unsqueeze(-1)).to(x.dtype)
        y = self.shared(x)
        for e_i, expert in enumerate(self.experts):
            mask = (idx == e_i)                           # [B, T]
            if not bool(mask.any()):
                continue
            x_e = x[mask]                                  # only routed tokens
            y_e = expert(x_e)
            contrib = torch.zeros_like(x)
            contrib[mask] = p[mask] * y_e.to(contrib.dtype)
            y = y + contrib
        self._last_probs = probs
        return y


def load_dense(device: str = "cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                 device_map=device)
    return model, tok


def upcycle(n_experts: int, group: int, device: str):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(BASE)
    torch.manual_seed(0)  # deterministic router init across runs
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                 device_map=device)
    act = model.model.layers[0].mlp.act_fn
    hidden = cfg.hidden_size
    for layer in model.model.layers:
        dev = next(layer.mlp.parameters()).device
        layer.mlp = MoEFFN(layer.mlp, hidden, n_experts, act, group).to(dev)
    # freeze everything except the ternary experts and the router
    for name, p in model.named_parameters():
        p.requires_grad_((".experts." in name) or (".router." in name))
    return model


# ------------------------------------------------------------------- corpus --

def windows(tok, n: int, seq: int, seed: int, split: str):
    from datasets import load_dataset
    if split == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    else:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train",
                          streaming=True)
        buf = []
        for row in ds:
            buf.append(row["text"])
            if sum(len(t) for t in buf) > 10_000_000:
                break
        text = "\n\n".join(buf)
    ids = tok(text, return_tensors="pt").input_ids[0]
    rng = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, max(len(ids) - seq - 1, 1), (n,), generator=rng)
    return torch.stack([ids[s:s + seq] for s in starts])


# ---------------------------------------------------------------- training ---

def train(args):
    model = upcycle(args.experts, args.group, args.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f"trainable: {n_tr/1e9:.3f}B params (routed experts only)")

    teacher = None
    if args.kd_weight > 0:
        teacher, tok = load_dense(args.teacher_device)
        for p in teacher.parameters():
            p.requires_grad_(False)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE)

    data = windows(tok, args.windows, args.seq, args.seed, args.corpus)
    opt = torch.optim.Adafactor(trainable, lr=args.lr, weight_decay=0.0)
    model.train()
    step = 0
    OUT.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        for i in range(0, len(data)):
            ids = data[i:i + 1].to(args.device)
            out = model(ids, output_hidden_states=False)
            logits = out.logits
            lm = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                ids[:, 1:].reshape(-1))
            loss = lm
            kd_val = 0.0
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(ids).logits
                # KD on the teacher's top-50 logits
                k = min(50, t_logits.shape[-1])
                t_top = t_logits[:, :-1].topk(k, dim=-1)
                s_sel = logits[:, :-1].gather(-1, t_top.indices).reshape(-1, k)
                t_val = t_top.values.reshape(-1, k)
                kd = F.kl_div(F.log_softmax(s_sel.float() / args.temp, dim=-1),
                              F.log_softmax(t_val.float() / args.temp, dim=-1),
                              log_target=True, reduction="batchmean") * (args.temp ** 2)
                loss = loss + args.kd_weight * kd
                kd_val = float(kd)
            # Switch-style load-balance aux (MoTE gamma=0.01)
            probs = torch.stack([layer.mlp._last_probs for layer in model.model.layers])
            idx_all = probs.argmax(-1)                                  # [L, T]
            share = torch.nn.functional.one_hot(
                idx_all, num_classes=args.experts).float().mean(dim=1)  # [L, E]
            mean_p = probs.mean(dim=1)                                  # [L, E]
            aux = (share * mean_p).sum(-1).mean() * args.experts
            loss = loss + args.aux_weight * aux
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                print(f"step {step} lm {lm.item():.4f} kd {kd_val:.4f} "
                      f"total {loss.item():.4f}", flush=True)
            if args.steps and step >= args.steps:
                break
            if step % args.ckpt_every == 0:
                save(model, args, step)
        if args.steps and step >= args.steps:
            break
    save(model, args, step)
    print("training done")


def save(model, args, step):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"moe-e{args.experts}-g{args.group}-step{step}.pt"
    sd = {k: v for k, v in model.state_dict().items() if ".experts." in k or ".router." in k}
    torch.save(sd, p)
    print(f"saved {p}")


@torch.no_grad()
def evaluate(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    data = windows(tok, args.eval_windows, args.seq, args.eval_seed, "wikitext")

    results = {}

    def ppl(model, tag):
        model.eval()
        total, ntok = 0.0, 0
        for i in range(len(data)):
            ids = data[i:i + 1].to(args.device)
            logits = model(ids).logits[:, :-1]
            tgt = ids[:, 1:]
            total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                     tgt.reshape(-1), reduction="sum").item()
            ntok += tgt.numel()
        results[tag] = math.exp(total / ntok)
        print(f"{tag}: ppl {results[tag]:.4f}")

    dense, _ = load_dense(args.device)
    ppl(dense, "dense_fp")
    del dense
    torch.cuda.empty_cache()

    moe = upcycle(args.experts, args.group, args.device)
    ppl(moe, "upcycled_masters_fp")
    # RTN (ternarised, untrained)
    with torch.no_grad():
        for layer in moe.model.layers:
            for e in layer.mlp.experts:
                for lin in (e.gate_proj, e.up_proj, e.down_proj):
                    lin.weight.copy_(ternary_absmean(lin.weight, args.group))
    ppl(moe, "upcycled_rtn")

    if args.load:
        sd = torch.load(args.load, map_location="cpu")
        missing = moe.load_state_dict(sd, strict=False)
        print(f"loaded {args.load}: missing={len(missing.missing_keys)}")
        ppl(moe, "trained_ternary")

    (OUT / "eval.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["upcycle", "train", "eval"])
    ap.add_argument("--experts", type=int, default=2)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--teacher-device", default="cuda:0")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--kd-weight", type=float, default=0.0)
    ap.add_argument("--aux-weight", type=float, default=0.01)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--corpus", default="fineweb")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--eval-seed", type=int, default=999)
    ap.add_argument("--load", default="")
    args = ap.parse_args()

    if args.stage == "upcycle":
        model = upcycle(args.experts, args.group, args.device)
        save(model, args, 0)
    elif args.stage == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()

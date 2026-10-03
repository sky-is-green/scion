"""ToMoE-style learned channel masks inside the ternary experts (OLMoE).

Body: expert banks RTN-quantised once in place, frozen (same setting as the
other arms).  Masks: per-expert binary masks over the intermediate dimension,
learned jointly with the routers under output KD -- the MLP-expert form of
ToMoE (arXiv 2501.15316).  Mask = top-k of sigmoid(logits) with a straight-
through estimator; k = --keep-frac * intermediate.

Arms:
  --branch none      masks only (the new inside-expert structural correction)
  --branch residual  masks + the published residual-stream branch placement
                     (olmoe_corrections.MoEWithCorrection, rank from --rank)

Usage:
  HIP_VISIBLE_DEVICES=1 python olmoe_masks.py train --device cuda:0 \
      --branch none --keep-frac 0.5 --steps 1024
  HIP_VISIBLE_DEVICES=1 python olmoe_masks.py eval --device cuda:0 \
      --branch none --keep-frac 0.5 --load <ckpt>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from olmoe_corrections import MoEWithCorrection, quantize_bank_inplace  # noqa: E402
from olmoe_proxy import CACHE, MODEL, OUT, gate_hook, load_model, windows  # noqa: E402


class ExpertMask(nn.Module):
    """Per-expert top-k channel mask with a straight-through estimator."""

    def __init__(self, num_experts: int, dim: int, keep_frac: float,
                 seed: int = 0):
        super().__init__()
        if not 0.0 < keep_frac <= 1.0:
            raise ValueError("keep_frac must be in (0, 1]")
        gen = torch.Generator().manual_seed(seed)
        self.logits = nn.Parameter(torch.randn(num_experts, dim, generator=gen) * 0.01)
        self.k = max(1, int(round(dim * keep_frac)))

    def forward(self) -> torch.Tensor:
        p = torch.sigmoid(self.logits)
        thresh = p.topk(self.k, dim=-1).values[..., -1:]
        hard = (p >= thresh).to(p.dtype)
        return hard + p - p.detach()          # values are hard; grads to logits

    @torch.no_grad()
    def keep_counts(self) -> torch.Tensor:
        return (self.forward() > 0.5).sum(dim=-1)


def masked_expert_forward(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply an expert mask to the intermediate state (the testable core)."""
    return h * mask


def patch_experts_masked(model, keep_frac: float, mask_seed: int = 0) -> None:
    """Attach a channel mask to every OlmoeExperts and route forward through it."""
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

    def forward(self, hidden_states, top_k_index, top_k_weights):
        gu, dn = self.gate_up_proj, self.down_proj
        mask = self.channel_mask().to(hidden_states.dtype)
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index,
                                                      num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in expert_hit:
            e = e[0]
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            state = hidden_states[token_idx]
            gate, up = F.linear(state, gu[e]).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = masked_expert_forward(h, mask[e])
            h = F.linear(h, dn[e])
            h = h * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))
        return final_hidden_states

    for m in model.modules():
        if not isinstance(m, OlmoeExperts):
            continue
        dev = next(m.parameters()).device
        m.channel_mask = ExpertMask(m.num_experts, m.intermediate_dim,
                                    keep_frac, mask_seed).to(dev)
        m.forward = forward.__get__(m, type(m))


def build(args):
    mm = {0: "14GiB", 1: "19GiB"} if args.device_map == "auto" else None
    model, tok = load_model(args.device_map, mm)
    for layer in model.model.layers:
        quantize_bank_inplace(layer.mlp.experts.gate_up_proj, args.group, kind=args.quant)
        quantize_bank_inplace(layer.mlp.experts.down_proj, args.group, kind=args.quant)
    patch_experts_masked(model, args.keep_frac, args.mask_seed)
    if args.branch == "residual":
        hidden = model.config.hidden_size
        for layer in model.model.layers:
            dev = next(layer.mlp.parameters()).device
            layer.mlp = MoEWithCorrection(layer.mlp, hidden, args.rank,
                                          args.branch_quant, args.quant).to(dev)
    for name, p in model.named_parameters():
        p.requires_grad_((".channel_mask." in name) or (".gate." in name)
                         or (".branch." in name))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_mask = sum(p.numel() for n, p in model.named_parameters() if ".channel_mask." in n)
    print(f"trainable {n_tr/1e6:.2f}M ({n_mask/1e6:.2f}M mask logits) "
          f"branch={args.branch} keep_frac={args.keep_frac}", flush=True)
    return model, tok


def save(model, args, step):
    sd = {k: v for k, v in model.state_dict().items()
          if ".channel_mask." in k or ".gate." in k or ".branch." in k}
    p = OUT / f"olmoe-mask-k{args.keep_frac:g}-{args.branch}-step{step}.pt"
    torch.save(sd, p)
    print(f"saved {p}", flush=True)


@torch.no_grad()
def quick_eval(model, data, args, ref=None):
    model.eval()
    total, ntok, agree = 0.0, 0, []
    for i in range(len(data)):
        ids = data[i:i + 1].to(args.device)
        store = {}
        hs = [layer.mlp.gate.register_forward_hook(gate_hook(store, j))
              for j, layer in enumerate(model.model.layers)]
        logits = model(ids).logits
        for h in hs:
            h.remove()
        total += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                 ids[:, 1:].reshape(-1), reduction="sum").item()
        ntok += ids[:, 1:].numel()
        if ref is not None:
            for j in store:
                a = ref[i][j].to(store[j][2].device)
                b = store[j][2]
                agree.append(float((a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean()))
    model.train()
    ppl = math.exp(total / ntok)
    return ppl, (sum(agree) / len(agree) if agree else None)


def train(args):
    model, tok = build(args)
    cache = torch.load(CACHE, map_location="cpu")
    data = windows(tok, args.windows, args.seq, args.seed)

    ref = None
    ev = None
    if args.eval_every:
        mm = {0: "14GiB", 1: "19GiB"} if args.device_map == "auto" else None
        teacher, _ = load_model(args.device_map, mm)
        ev = windows(tok, 2, args.seq, 999, "wikitext")
        ref = {}
        for i in range(len(ev)):
            ids = ev[i:i + 1].to(args.device)
            store = {}
            hs = [layer.mlp.gate.register_forward_hook(gate_hook(store, j))
                  for j, layer in enumerate(teacher.model.layers)]
            teacher(ids)
            for h in hs:
                h.remove()
            ref[i] = {j: store[j][2].cpu() for j in store}
        del teacher
        torch.cuda.empty_cache()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adafactor(params, lr=args.lr, weight_decay=0.0)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for rec in cache:
            ids = data[step % len(data):step % len(data) + 1].to(args.device)
            store = {}
            hs = [layer.mlp.gate.register_forward_hook(gate_hook(store, j))
                  for j, layer in enumerate(model.model.layers)]
            logits = model(ids).logits
            for h in hs:
                h.remove()
            lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                 ids[:, 1:].reshape(-1))
            ti = rec["idx"].to(logits.device)
            tv = rec["val"].to(logits.device).float()
            s_sel = logits[:, :-1].gather(-1, ti).reshape(-1, ti.shape[-1])
            kd = F.kl_div(F.log_softmax(s_sel.float() / args.temp, dim=-1),
                          F.log_softmax(tv.reshape(-1, tv.shape[-1]) / args.temp, dim=-1),
                          log_target=True, reduction="batchmean") * (args.temp ** 2)
            rkd = torch.zeros((), device=args.device)
            for i, layer in enumerate(model.model.layers):
                s_logits, _, _ = store[i]
                dev = s_logits.device
                t_idx = rec["router"][i][0].to(dev).long()
                t_p = rec["router"][i][1].to(dev).float()
                s_p = F.softmax(s_logits.float(), dim=-1).gather(-1, t_idx)
                s_p = s_p / s_p.sum(-1, keepdim=True).clamp_min(1e-9)
                t_p = t_p / t_p.sum(-1, keepdim=True).clamp_min(1e-9)
                rkd = rkd + F.kl_div(s_p.clamp_min(1e-9).log(), t_p,
                                     reduction="batchmean").to(rkd.device)
            rkd = rkd / len(model.model.layers)
            loss = lm + args.kd_weight * kd + args.router_weight * rkd
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                print(f"step {step} lm {lm.item():.4f} kd {kd.item():.4f} "
                      f"rkd {float(rkd):.4f} total {loss.item():.4f}", flush=True)
            if args.eval_every and step % args.eval_every == 0 and ev is not None:
                ppl, ag = quick_eval(model, ev, args, ref)
                print(f"  [eval] step {step} ppl {ppl:.2f} router_agree {ag:.4f}",
                      flush=True)
            if step % args.ckpt_every == 0:
                save(model, args, step)
            if args.steps and step >= args.steps:
                break
        if args.steps and step >= args.steps:
            break
    save(model, args, step)
    print("training done", flush=True)


@torch.no_grad()
def evaluate(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = windows(tok, args.eval_windows, args.seq, 999, "wikitext")

    teacher, _ = load_model(args.device)
    ref, t_total, t_ntok = {}, 0.0, 0
    for i in range(len(data)):
        ids = data[i:i + 1].to(args.device)
        store = {}
        hs = [layer.mlp.gate.register_forward_hook(gate_hook(store, j))
              for j, layer in enumerate(teacher.model.layers)]
        logits = teacher(ids).logits
        for h in hs:
            h.remove()
        ref[i] = {j: store[j][2].cpu() for j in store}
        t_total += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                   ids[:, 1:].reshape(-1), reduction="sum").item()
        t_ntok += ids[:, 1:].numel()
    print(f"teacher ppl {math.exp(t_total/t_ntok):.4f}", flush=True)
    del teacher
    torch.cuda.empty_cache()

    model, _ = build(args)
    model.eval()

    def run(tag):
        total, ntok, agree = 0.0, 0, []
        for i in range(len(data)):
            ids = data[i:i + 1].to(args.device)
            store = {}
            hs = [layer.mlp.gate.register_forward_hook(gate_hook(store, j))
                  for j, layer in enumerate(model.model.layers)]
            logits = model(ids).logits
            for h in hs:
                h.remove()
            total += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                     ids[:, 1:].reshape(-1), reduction="sum").item()
            ntok += ids[:, 1:].numel()
            for j in store:
                a = ref[i][j].to(store[j][2].device)
                b = store[j][2]
                agree.append(float((a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean()))
        ppl = math.exp(total / ntok)
        print(f"{tag}: ppl {ppl:.4f} router_agree {sum(agree)/len(agree):.4f}", flush=True)
        return {"ppl": round(ppl, 4), "router_agree": round(sum(agree) / len(agree), 4)}

    res = {"rtn_no_masks": run("rtn_no_masks")}
    if args.load:
        sd = torch.load(args.load, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        res["trained"] = run("trained")
    (OUT / "masks-eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "eval"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--windows", type=int, default=512)
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--quant", choices=["absmean", "lloyd"], default="absmean")
    ap.add_argument("--keep-frac", type=float, default=0.5)
    ap.add_argument("--mask-seed", type=int, default=0)
    ap.add_argument("--branch", choices=["none", "residual"], default="none")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--branch-quant", choices=["fp32", "g128", "rank"], default="fp32")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--router-weight", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--load", default="")
    args = ap.parse_args()
    if args.stage == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()

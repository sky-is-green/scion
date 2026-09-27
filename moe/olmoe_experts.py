"""TAARDIS-style per-expert correction branches for a ternary MoE (OLMoE).

Body: expert banks RTN-quantised once in place, frozen.
Branches: per-expert low-rank corrections inside each expert, applied to the
intermediate state (after gate/up) and to the expert output (after down).
Routers stay trainable.  Loss = LM + output KD + router KD.

This is the TAARDIS placement (per-matmul branches) rather than one branch per
layer output.

Usage:
  HIP_VISIBLE_DEVICES=0,1 python olmoe_experts.py train \
      --device-map auto --steps 2000 --rank 8
  ... eval --load <ckpt>
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
from olmoe_corrections import quantize_bank_inplace  # noqa: E402
from olmoe_proxy import CACHE, MODEL, OUT, gate_hook, load_model, windows  # noqa: E402


class ExpertBranch(nn.Module):
    """Low-rank correction, fp32 masters, zero-init on the output side."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x.float())).to(x.dtype)


def patch_experts_corrected(model, rank: int) -> None:
    """Attach per-expert branches and route the expert forward through them."""
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

    def forward(self, hidden_states, top_k_index, top_k_weights):
        gu = self.gate_up_proj
        dn = self.down_proj
        bu = self.branches_up
        bd = self.branches_down
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, gu[expert_idx]).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = h + bu[expert_idx](h)
            h = F.linear(h, dn[expert_idx])
            h = h + bd[expert_idx](h)
            h = h * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))
        return final_hidden_states

    for m in model.modules():
        if not isinstance(m, OlmoeExperts):
            continue
        dev = next(m.parameters()).device
        m.branches_up = nn.ModuleList(
            [ExpertBranch(m.intermediate_dim, rank) for _ in range(m.num_experts)]).to(dev)
        m.branches_down = nn.ModuleList(
            [ExpertBranch(m.hidden_dim, rank) for _ in range(m.num_experts)]).to(dev)
        m.forward = forward.__get__(m, type(m))


def build(args):
    mm = {0: "14GiB", 1: "19GiB"} if args.device_map == "auto" else None
    model, tok = load_model(args.device_map, mm)
    for layer in model.model.layers:
        quantize_bank_inplace(layer.mlp.experts.gate_up_proj, args.group)
        quantize_bank_inplace(layer.mlp.experts.down_proj, args.group)
    patch_experts_corrected(model, args.rank)
    for name, p in model.named_parameters():
        p.requires_grad_((".branches_" in name) or (".gate." in name))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_tr/1e6:.2f}M (per-expert branches + routers)", flush=True)
    return model, tok


def save(model, args, step):
    sd = {k: v for k, v in model.state_dict().items()
          if ".branches_" in k or ".gate." in k}
    p = OUT / f"olmoe-exp-r{args.rank}-step{step}.pt"
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
                print(f"  [eval] step {step} ppl {ppl:.2f} router_agree {ag:.4f}", flush=True)
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

    res = {"rtn_no_branches": run("rtn_no_branches")}
    if args.load:
        sd = torch.load(args.load, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        res["trained_branches"] = run("trained_branches")
    (OUT / "experts-eval.json").write_text(json.dumps(res, indent=2))
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
    ap.add_argument("--rank", type=int, default=8)
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

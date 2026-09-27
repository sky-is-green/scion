"""Proxy B — ternary QAT of an existing pretrained MoE (OLMoE-1B-7B) with
router-aware knowledge distillation.

Stages:
  cache : FP teacher pass -> per-window top-50 output logits + per-layer
          router top-8 indices/probs
  train : ternary STE on the fused expert banks (gate_up_proj/down_proj),
          router kept BF16 and trainable; loss = LM + output KD + router KD
  eval  : held-out PPL and per-layer router top-8 agreement for
          teacher / RTN / trained

Teacher and student are the same checkpoint; the teacher is the FP model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moe_proxy import ternary_absmean, ternary_lloyd  # noqa: E402

ART = Path(os.environ.get("MOE_ARTIFACTS", HERE / "artifacts"))
MODEL = ART / "olmoe-hf"
OUT = ART / "olmoe"
CACHE = OUT / "teacher-cache.pt"


def ternary_ste(w: torch.Tensor, group: int = 128, kind: str = "absmean") -> torch.Tensor:
    """Straight-through ternary with the selected per-group scale rule."""
    fn = ternary_lloyd if kind == "lloyd" else ternary_absmean
    with torch.no_grad():
        wq = fn(w, group)
    return wq + (w - w.detach())


def patch_experts(group: int, model=None):
    """Make OlmoeExperts ternarise its fused banks on the fly (STE).

    Only modules with ``_ternary=True`` are affected; the expert math runs in
    the master dtype (fp32 masters supported) and is cast back at the end.

    Multi-GPU dispatch (device_map="auto") binds the pre-patch forward onto
    each experts *instance*, shadowing the class patch.  Pass the loaded
    ``model`` to rebind those instances too.
    """
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

    def ternary_forward(self, hidden_states, top_k_index, top_k_weights):
        gu = ternary_ste(self.gate_up_proj, group)
        dn = ternary_ste(self.down_proj, group)
        dt = gu.dtype
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
            current_state = hidden_states[token_idx].to(dt)
            gate, up = F.linear(current_state, gu[expert_idx]).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = F.linear(h, dn[expert_idx])
            h = h * top_k_weights[token_idx, top_k_pos, None].to(dt)
            final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))
        return final_hidden_states

    original = OlmoeExperts.forward

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if not getattr(self, "_ternary", False):
            return original(self, hidden_states, top_k_index, top_k_weights)
        return ternary_forward(self, hidden_states, top_k_index, top_k_weights)

    OlmoeExperts.forward = forward

    if model is not None:
        for m in model.modules():
            if not isinstance(m, OlmoeExperts):
                continue
            inner = m.__dict__.get("forward")

            def make(inner):
                def f(self, *a, **k):
                    if not getattr(self, "_ternary", False):
                        return inner(*a, **k)
                    return ternary_forward(self, *a, **k)
                return f

            m.forward = make(inner).__get__(m, type(m)) if inner is not None else forward.__get__(m, type(m))


def load_model(device_map="cuda:0", max_memory=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map=device_map,
                                                 max_memory=max_memory)
    return model, tok


def windows(tok, n, seq, seed, split="fineweb", max_chars=10_000_000):
    from datasets import load_dataset
    if split == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    else:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        buf = []
        for row in ds:
            buf.append(row["text"])
            if sum(len(t) for t in buf) > max_chars:
                break
        text = "\n\n".join(buf)
    ids = tok(text, return_tensors="pt").input_ids[0]
    rng = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, max(len(ids) - seq - 1, 1), (n,), generator=rng)
    return torch.stack([ids[s:s + seq] for s in starts])


def parse_layers(spec: str, n_layers: int) -> set:
    if not spec or spec == "all":
        return set(range(n_layers))
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def gate_hook(store, i):
    def hook(mod, inp, out):
        store[i] = out                      # (logits, scores, indices)
    return hook


# ------------------------------------------------------------------- cache ---

FEAT_STRIDE = 4   # token stride for cached hidden-state distillation targets


@torch.no_grad()
def stage_cache(args):
    model, tok = load_model(args.device)
    data = windows(tok, args.windows, args.seq, args.seed,
                   max_chars=args.corpus_chars)
    feat_states = bool(getattr(args, "feat_states", False))
    recs = []
    for w in range(len(data)):
        ids = data[w:w + 1].to(args.device)
        store = {}
        hs = []
        for i, layer in enumerate(model.model.layers):
            hs.append(layer.mlp.gate.register_forward_hook(gate_hook(store, i)))
        fstore = {}
        fh = None
        if feat_states:
            fh = model.model.norm.register_forward_hook(
                lambda m, inp, out: fstore.__setitem__("feat", out))
        logits = model(ids).logits
        for h in hs:
            h.remove()
        if fh is not None:
            fh.remove()
        t_top = logits[:, :-1].topk(args.top_logits, dim=-1)
        router = {i: (store[i][2].cpu().to(torch.int16),
                      store[i][1].cpu().to(torch.float16))
                  for i in store}
        rec = {"idx": t_top.indices.cpu().to(torch.int32),
               "val": t_top.values.cpu().to(torch.float16),
               "router": router}
        if feat_states:
            rec["feat"] = fstore["feat"][0][::FEAT_STRIDE].cpu().to(torch.float16)
        recs.append(rec)
        if w % 100 == 0:
            print(f"cached {w}/{len(data)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if getattr(args, "out", "") else CACHE
    torch.save(recs, path)
    print(f"wrote {path} ({path.stat().st_size/1e9:.2f} GB)")


# ------------------------------------------------------------------- train ---

def stage_train(args):
    patch_experts(args.group)
    mm = {0: "14GiB", 1: "19GiB"} if args.device_map == "auto" else None
    model, tok = load_model(args.device_map, mm)
    sel = parse_layers(args.train_layers, len(model.model.layers))
    for i, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        experts._ternary = i in sel
        if args.master_dtype == "fp32" and i in sel:
            with torch.no_grad():
                experts.gate_up_proj.data = experts.gate_up_proj.data.float()
                experts.down_proj.data = experts.down_proj.data.float()
    for name, p in model.named_parameters():
        if "experts." in name:
            layer_i = int(name.split("layers.")[1].split(".")[0])
            p.requires_grad_(layer_i in sel)
        elif ".gate." in name:
            layer_i = int(name.split("layers.")[1].split(".")[0])
            p.requires_grad_(layer_i in sel)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_tr/1e9:.3f}B over layers {sorted(sel)} "
          f"(masters {args.master_dtype})")
    cache = torch.load(CACHE, map_location="cpu")
    opt = torch.optim.Adafactor([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, weight_decay=0.0)
    model.train()
    OUT.mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(args.epochs):
        for rec in cache:
            ids = DATA[step % len(DATA):step % len(DATA) + 1].to(args.device)
            store = {}
            hs = []
            for i, layer in enumerate(model.model.layers):
                hs.append(layer.mlp.gate.register_forward_hook(gate_hook(store, i)))
            logits = model(ids).logits
            for h in hs:
                h.remove()
            lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                 ids[:, 1:].reshape(-1))
            # output KD on teacher top-k
            ti = rec["idx"].to(logits.device)
            tv = rec["val"].to(logits.device).float()
            s_sel = logits[:, :-1].gather(-1, ti).reshape(-1, ti.shape[-1])
            kd = F.kl_div(F.log_softmax(s_sel.float() / args.temp, dim=-1),
                          F.log_softmax(tv.reshape(-1, tv.shape[-1]) / args.temp, dim=-1),
                          log_target=True, reduction="batchmean") * (args.temp ** 2)
            # router KD: student probs over teacher's top-8, renormalised
            rkd = torch.tensor(0.0, device=args.device)
            for i, layer in enumerate(model.model.layers):
                s_logits, _, _ = store[i]
                dev = s_logits.device
                t_idx = rec["router"][i][0].to(dev).long()              # [T, 8]
                t_p = rec["router"][i][1].to(dev).float()               # [T, 8]
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
            if step % args.ckpt_every == 0:
                save(model, args, step)
            if args.steps and step >= args.steps:
                break
        if args.steps and step >= args.steps:
            break
    save(model, args, step)
    print("training done")


DATA = None


def save(model, args, step):
    sd = {k: v for k, v in model.state_dict().items()
          if "experts." in k or ".gate." in k}
    p = OUT / f"olmoe-e-g{args.group}-step{step}.pt"
    torch.save(sd, p)
    print(f"saved {p}")


# -------------------------------------------------------------------- eval ---

@torch.no_grad()
def stage_eval(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = windows(tok, args.eval_windows, args.seq, 999, "wikitext")

    def run(model, tag, ref_router=None):
        total, ntok = 0.0, 0
        agree = {i: [] for i in range(len(model.model.layers))}
        for i in range(len(data)):
            ids = data[i:i + 1].to(args.device)
            store = {}
            hs = []
            for j, layer in enumerate(model.model.layers):
                hs.append(layer.mlp.gate.register_forward_hook(gate_hook(store, j)))
            logits = model(ids).logits
            for h in hs:
                h.remove()
            total += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                     ids[:, 1:].reshape(-1), reduction="sum").item()
            ntok += ids[:, 1:].numel()
            if ref_router is not None:
                for j in store:
                    a = ref_router[i][j].to(store[j][2].device)
                    b = store[j][2]
                    inter = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean().item()
                    agree[j].append(inter)
        ppl = math.exp(total / ntok)
        out = {"ppl": round(ppl, 4)}
        if ref_router is not None:
            out["router_agree_mean"] = round(
                sum(sum(v) / len(v) for v in agree.values()) / len(agree), 4)
        print(tag, out, flush=True)
        return out

    # teacher
    teacher, _ = load_model(args.device)
    ref = {}
    for i in range(len(data)):
        ids = data[i:i + 1].to(args.device)
        store = {}
        hs = [layer.mlp.gate.register_forward_hook(gate_hook(store, j))
              for j, layer in enumerate(teacher.model.layers)]
        teacher(ids)
        for h in hs:
            h.remove()
        ref[i] = {j: store[j][2].cpu() for j in store}
    results = {"teacher": run(teacher, "teacher_fp", ref)}
    del teacher
    torch.cuda.empty_cache()

    patch_experts(args.group)
    student, _ = load_model(args.device)
    sel = parse_layers(args.train_layers, len(student.model.layers))
    for i, layer in enumerate(student.model.layers):
        layer.mlp.experts._ternary = i in sel
    results["rtn"] = run(student, "rtn_untrained", ref)
    if args.load:
        sd = torch.load(args.load, map_location="cpu")
        student.load_state_dict(sd, strict=False)
        results["trained"] = run(student, "trained", ref)
    (OUT / "eval.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cache", "train", "eval"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--windows", type=int, default=1024)
    ap.add_argument("--corpus-chars", type=int, default=10_000_000,
                    help="fineweb character buffer to draw windows from")
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--train-layers", default="all")
    ap.add_argument("--master-dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--top-logits", type=int, default=50)
    ap.add_argument("--feat-states", action="store_true",
                    help="also cache the final-norm hidden states (token stride "
                         f"{FEAT_STRIDE}) for feature distillation")
    ap.add_argument("--out", default="",
                    help="cache output path (default: $MOE_ARTIFACTS/olmoe/teacher-cache.pt)")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--router-weight", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--load", default="")
    args = ap.parse_args()

    if args.stage == "cache":
        stage_cache(args)
    elif args.stage == "train":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        DATA = windows(tok, args.windows, args.seq, args.seed)
        stage_train(args)
    else:
        stage_eval(args)


if __name__ == "__main__":
    main()

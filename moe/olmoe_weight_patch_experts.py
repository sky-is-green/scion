"""Per-expert weight patching inside the sensitive early layers.

Layer-level patching showed that restoring one early layer's expert banks
recovers most of the ternary PPL gap.  This screen asks whether the damage is
concentrated in a few experts: rank experts by weight error, patch groups of
8/16/32, and compare with the bottom-8 control.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from olmoe_proxy import MODEL, OUT, gate_hook, load_model, windows  # noqa: E402
from olmoe_weight_patch import eval_model, quantize_experts  # noqa: E402


def expert_error(texp, sexp, device):
    """Per-expert relative squared error, gate_up and down combined."""
    errs = {}
    for name in ("gate_up_proj", "down_proj"):
        t = getattr(texp, name).detach().float().to(device)
        s = getattr(sexp, name).detach().float()
        num = (t - s).flatten(1).pow(2).sum(1)
        den = t.flatten(1).pow(2).sum(1).clamp_min(1e-12)
        errs[name] = (num / den).cpu()
        del t, s, num, den
    return errs["gate_up_proj"], errs["down_proj"]


@torch.no_grad()
def expert_load(model, data, device, n_layers):
    counts = [torch.zeros(64, dtype=torch.long) for _ in range(n_layers)]
    for i in range(len(data)):
        ids = data[i:i + 1].to(device)
        store = {}
        hs = [layer.mlp.gate.register_forward_hook(gate_hook(store, j))
              for j, layer in enumerate(model.model.layers)]
        model(ids)
        for h in hs:
            h.remove()
        for j, (_, _, idx) in store.items():
            counts[j] += torch.bincount(idx.reshape(-1).cpu(), minlength=64)
    return [c.float() / c.sum().clamp_min(1) for c in counts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-device", default="cuda:0")
    ap.add_argument("--student-device", default="cuda:1")
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--layers", default="0,1,2,3,8")
    ap.add_argument("--groups", default="8,16,32")
    ap.add_argument("--rank", choices=["error", "error_x_load", "load"],
                    default="error")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = windows(tok, args.eval_windows, args.seq, 999, "wikitext")

    teacher, _ = load_model(args.teacher_device)
    ref, t_total, t_ntok = {}, 0.0, 0
    for i in range(len(data)):
        ids = data[i:i + 1].to(args.teacher_device)
        store = {}
        hs = [layer.mlp.gate.register_forward_hook(gate_hook(store, j))
              for j, layer in enumerate(teacher.model.layers)]
        logits = teacher(ids).logits
        for h in hs:
            h.remove()
        ref[i] = {j: store[j][2].cpu() for j in store}
        t_total += torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            ids[:, 1:].reshape(-1), reduction="sum").item()
        t_ntok += ids[:, 1:].numel()
    teacher_ppl = math.exp(t_total / t_ntok)
    print(f"teacher ppl {teacher_ppl:.4f}", flush=True)

    student, _ = load_model(args.student_device)
    quantize_experts(student, args.group)
    base_ppl, base_agree = eval_model(student, data, args.student_device, ref)
    print(f"ternary student ppl {base_ppl:.4f} agree {base_agree:.4f}", flush=True)
    loads = expert_load(student, data, args.student_device,
                        len(student.model.layers))

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    groups = [int(x) for x in args.groups.split(",") if x.strip()]
    out = {"teacher_ppl": round(teacher_ppl, 4),
           "ternary_ppl": round(base_ppl, 4),
           "ternary_agree": round(base_agree, 4),
           "layers": {}}

    for i in layers:
        texp = teacher.model.layers[i].mlp.experts
        sexp = student.model.layers[i].mlp.experts
        gu_err, dn_err = expert_error(texp, sexp, args.student_device)
        err = gu_err + dn_err
        if args.rank == "load":
            score = loads[i]
        elif args.rank == "error_x_load":
            score = err * loads[i]
        else:
            score = err
        order = torch.argsort(score, descending=True).tolist()
        rec = {"rank_mode": args.rank,
               "expert_load": [round(float(x), 6) for x in loads[i]],
               "expert_error_gate_up": [round(float(x), 6) for x in gu_err],
               "expert_error_down": [round(float(x), 6) for x in dn_err],
               "rank": order, "patches": []}
        for k in groups:
            for label, sel in (("top", order[:k]),
                               ("bottom", order[-k:])):
                saved = {}
                with torch.no_grad():
                    for e in sel:
                        saved[e] = (sexp.gate_up_proj[e].detach().clone(),
                                    sexp.down_proj[e].detach().clone())
                        sexp.gate_up_proj[e].copy_(
                            texp.gate_up_proj[e].to(args.student_device))
                        sexp.down_proj[e].copy_(
                            texp.down_proj[e].to(args.student_device))
                ppl, agree = eval_model(student, data, args.student_device, ref)
                recovery = (base_ppl - ppl) / max(base_ppl - teacher_ppl, 1e-9)
                rec["patches"].append({"group": f"{label}-{k}",
                                       "ppl": round(ppl, 4),
                                       "recovery": round(recovery, 5),
                                       "router_agree": round(agree, 4)})
                print(f"  L{i:02d} {label}-{k}: ppl {ppl:9.3f} "
                      f"recovery {recovery:7.4f} agree {agree:.4f}", flush=True)
                with torch.no_grad():
                    for e, (g, d) in saved.items():
                        sexp.gate_up_proj[e].copy_(g)
                        sexp.down_proj[e].copy_(d)
        out["layers"][str(i)] = rec

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"weight-patch-expert-groups-{args.rank}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "layers"}, indent=2))
    print(f"wrote {OUT/f'weight-patch-expert-groups-{args.rank}.json'}")


if __name__ == "__main__":
    main()

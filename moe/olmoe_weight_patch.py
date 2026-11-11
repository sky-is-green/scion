"""Layer-level weight patching on OLMoE: where does FP restoration recover PPL?

The arXiv Weight Patching method replaces parameter slices from a specialized
model into a base model and measures recovery of a behavioral anchor.  Here the
pair is (FP teacher, ternary student) and the anchor is held-out PPL / router
agreement.  This is the coarse layer-level screen; a head/neuron version can
follow if the map is concentrated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moe_proxy import ternary_lloyd  # noqa: E402
from olmoe_proxy import MODEL, OUT, gate_hook, load_model, windows  # noqa: E402


@torch.no_grad()
def quantize_experts(model, group: int = 128) -> None:
    for layer in model.model.layers:
        exp = layer.mlp.experts
        for name in ("gate_up_proj", "down_proj"):
            w = getattr(exp, name)
            w.copy_(ternary_lloyd(w, group))


@torch.no_grad()
def eval_model(model, data, device, ref=None):
    total, ntok, agree = 0.0, 0, []
    for i in range(len(data)):
        ids = data[i:i + 1].to(device)
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
    return math.exp(total / ntok), (sum(agree) / len(agree) if agree else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-device", default="cuda:0")
    ap.add_argument("--student-device", default="cuda:1")
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--targets", default="both",
                    help="comma list of expert banks to patch: "
                         "gate_up,down,both")
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
        t_total += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                   ids[:, 1:].reshape(-1), reduction="sum").item()
        t_ntok += ids[:, 1:].numel()
    teacher_ppl = math.exp(t_total / t_ntok)
    print(f"teacher ppl {teacher_ppl:.4f}", flush=True)

    student, _ = load_model(args.student_device)
    quantize_experts(student, args.group)
    base_ppl, base_agree = eval_model(student, data, args.student_device, ref)
    print(f"ternary student ppl {base_ppl:.4f} agree {base_agree:.4f}", flush=True)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    results = {}
    for target in targets:
        rows = []
        for i, layer in enumerate(student.model.layers):
            sexp = layer.mlp.experts
            texp = teacher.model.layers[i].mlp.experts
            names = (["gate_up_proj", "down_proj"] if target == "both"
                     else [{"gate_up": "gate_up_proj",
                            "down": "down_proj"}[target]])
            saved = {n: getattr(sexp, n).detach().clone() for n in names}
            with torch.no_grad():
                for n in names:
                    getattr(sexp, n).copy_(getattr(texp, n).to(args.student_device))
            ppl, agree = eval_model(student, data, args.student_device, ref)
            rec = (base_ppl - ppl) / max(base_ppl - teacher_ppl, 1e-9)
            rows.append({"layer": i, "ppl": round(ppl, 4),
                         "recovery": round(rec, 5),
                         "router_agree": round(agree, 4)})
            print(f"  [{target}] patch L{i:02d}: ppl {ppl:9.3f} "
                  f"recovery {rec:7.4f} agree {agree:.4f}", flush=True)
            with torch.no_grad():
                for n in names:
                    getattr(sexp, n).copy_(saved[n])
        results[target] = rows

    out = {"teacher_ppl": round(teacher_ppl, 4),
           "ternary_ppl": round(base_ppl, 4),
           "ternary_agree": round(base_agree, 4),
           "targets": results}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weight-patch-experts.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "targets"}, indent=2))
    print(f"wrote {OUT/'weight-patch-experts.json'}")


if __name__ == "__main__":
    main()

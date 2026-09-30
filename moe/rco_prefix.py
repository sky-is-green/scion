"""RCO allocation on the 4-layer qwen35 prefix (session 8 build).

Body-only allocator -- the surface RCO is meant to decide (handoff Sec. 7.2):

  groups    = the fused expert banks (per layer: gate_up_proj, down_proj)
  options   = {2 (deployable Lloyd g128 ternary), 4, 6, 8} bits/param with a
              fp16 scale per 128-weight group
  objective = the tc3/cur05 recipe's body-only loss on the cache:
              lm + kd_weight * KL(top-512, T) + tail_weight * marginal
              + tailcond_weight * D_KL2
  budget    = the hand allocation (all-ternary) bytes x --budget-pct/100

The loop is the ported RCO machinery (``rco_alloc.py``): Gumbel-perturbed
logits -> exact budget DP (hard forward) -> straight-through gradient
``dL/dp_k = <dL/dW, W_k>`` from per-parameter grad hooks -> tangent projection
-> Adam -> retraction -> vector transport.  Baselines measured at the same
budget: the hand map (all ternary, i.e. the floor), a sensitivity-greedy rule
(groups ranked by ternary reconstruction error) and a uniform rule.

Writes ``$MOE/qwen35/rco-alloc-<tag>.json``.

Lock discipline mirrors ``phase1-w1.sh``: refuses while a live stage holds
``$MOE/.stage-lock``, clears a stale one, removes its own on exit.
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kd_loss import residual_mass_kl, support_mass_lse, tail_conditional_piece  # noqa: E402
from qwen35_moe_proxy import (OUT, _corpus_windows, load_prefix,  # noqa: E402
                              model_logits, text_layers)
from rco_alloc import (SCALE_OVERHEAD, TERNARY_BPW, BudgetAllocator,  # noqa: E402
                       assignment_bits, dequant_option, dot_grad_option,
                       greedy_assignment, iter_dequant, quantize_tensor)

MOE = OUT  # artifacts/ternary/moe/qwen35


# ------------------------------------------------------------------ lock ---
def acquire_lock(lock: Path) -> None:
    if lock.exists():
        pid = None
        try:
            pid = int((lock / "pid").read_text().strip())
        except Exception:
            pass
        if pid and Path(f"/proc/{pid}").exists():
            raise SystemExit(f"REFUSING: live stage lock {lock} (pid {pid}) — "
                             f"another heavy stage holds the box")
        print(f"clearing stale lock {lock}", flush=True)
        shutil.rmtree(lock, ignore_errors=True)
    lock.mkdir(parents=True, exist_ok=False)
    (lock / "pid").write_text(str(os.getpid()))
    (lock / "tag").write_text("rco-alloc")
    atexit.register(lambda: shutil.rmtree(lock, ignore_errors=True))


# ------------------------------------------------------------- objective ---
def loss_terms(model, ids, rec, args, vocab: int):
    """The recipe's body-only loss on one cached window (tc3/cur05 terms)."""
    logits = model_logits(model, ids)
    lm = F.cross_entropy(logits[:, :-1].reshape(-1, vocab).float(),
                         ids[:, 1:].reshape(-1))
    ti = rec["idx"].to(logits.device)
    tv = rec["val"].to(logits.device).float()
    s_sel = logits[:, :-1].gather(-1, ti).reshape(-1, ti.shape[-1])
    kd = F.kl_div(F.log_softmax(s_sel.float() / args.temp, dim=-1),
                  F.log_softmax(tv.reshape(-1, ti.shape[-1]) / args.temp, dim=-1),
                  log_target=True, reduction="batchmean") * (args.temp ** 2)
    w_t = rec["w"].to(logits.device).float().reshape(-1)
    w_s, lse_s = support_mass_lse(logits[:, :-1].reshape(-1, vocab),
                                  ti.reshape(-1, ti.shape[-1]))
    tail = residual_mass_kl(w_s, w_t).mean()
    m = rec["tidx"].shape[-1]
    tcond = tail_conditional_piece(
        logits[:, :-1].reshape(-1, vocab),
        rec["tidx"].to(logits.device).reshape(-1, m),
        rec["tlp"].to(logits.device).float().reshape(-1, m),
        ti.reshape(-1, ti.shape[-1]), w_t,
        student_mass=w_s, student_lse=lse_s).mean()
    loss = (lm + args.kd_weight * kd + args.tail_weight * tail
            + args.tcond_weight * tcond)
    return loss, {"lm": float(lm), "kd": float(kd), "tail": float(tail),
                  "tcond": float(tcond)}


@torch.no_grad()
def eval_loss(model, data, cache, idxs, args, vocab: int) -> float:
    total = 0.0
    for w in idxs:
        ids = data[w:w + 1].to(args.device)
        loss, _ = loss_terms(model, ids, cache[w], args, vocab)
        total += float(loss)
    return total / max(len(idxs), 1)


# ---------------------------------------------------------- search setup ---
def searched_banks(model):
    out = []
    for i, layer in enumerate(text_layers(model)):
        e = layer.mlp.experts
        out.append((f"blk.{i}.ffn_gate_up_exps", e.gate_up_proj))
        out.append((f"blk.{i}.ffn_down_exps", e.down_proj))
    return out


def to_dev(opt, device):
    """Shallow copy of an option dict with the packed codes/scales on device."""
    return {**opt, "packed": opt["packed"].to(device, non_blocking=True),
            "scales": opt["scales"].to(device, non_blocking=True)}


def apply_assignment(tensors, opts, assignment, device) -> None:
    with torch.no_grad():
        for i, (_, p) in enumerate(tensors):
            o = to_dev(opts[i][int(assignment[i])], device)
            flat = p.data.reshape(-1, p.shape[-1])
            assert flat.data_ptr() == p.data.data_ptr(), "non-contiguous param"
            for (r0, r1), blk in iter_dequant(o, p.dtype):
                flat[r0:r1].copy_(blk)
            del o


def ternary_rel_error(p, opt, device) -> float:
    o = to_dev(opt, device)
    num = den = 0.0
    flat = p.detach().reshape(-1, p.shape[-1])
    for (r0, r1), blk in iter_dequant(o, torch.float32):
        d = flat[r0:r1].float() - blk
        num += float((d * d).sum())
        den += float((flat[r0:r1].float() ** 2).sum())
    del o
    return math.sqrt(num / max(den, 1e-12))


def avg_bpw(assignment, options, costs, weights) -> float:
    l2c = {int(b): float(c) for b, c in zip(options, costs.tolist())}
    return sum(float(weights[i].item()) * l2c[int(b)]
               for i, b in enumerate(assignment)) / float(weights.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", default=str(MOE / "prefix-top512-tail64-cur05.pt"))
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--options", default="2,4",
                    help="container bit levels (2 = ternary).  6/8 are supported "
                         "but their codes add ~2.4/3.2 GB of device memory for "
                         "the 4-layer banks — use them only on a clear card")
    ap.add_argument("--budget-pct", type=float, default=125.0,
                    help="percent of the all-ternary (hand) total bits")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tau-init", type=float, default=1.0)
    ap.add_argument("--tau-min", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--corpus-windows", type=int, default=4096)
    ap.add_argument("--corpus-chars", type=int, default=50_000_000)
    ap.add_argument("--corpus-file", default=str(MOE / "curric-combo.jsonl"))
    ap.add_argument("--agentic-frac", type=float, default=0.05)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--windows-search", type=int, default=32)
    ap.add_argument("--windows-holdout", type=int, default=8)
    ap.add_argument("--holdout-offset", type=int, default=2048)
    ap.add_argument("--kd-weight", type=float, default=2.0)
    ap.add_argument("--tail-weight", type=float, default=2.0)
    ap.add_argument("--tcond-weight", type=float, default=3.0)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not args.no_lock:
        acquire_lock(MOE / ".stage-lock")

    t0 = time.time()
    options = sorted(int(b) for b in args.options.split(","))
    assert options[0] == 2, "the hand floor (ternary, 2) must be an option"

    model, tok, _, _ = load_prefix(args.layers, args.device)
    vocab = int(model.config.vocab_size if hasattr(model.config, "vocab_size")
                else model.lm_head.out_features)

    # corpus windows exactly as the cache stage built them (same flags)
    cwin = _corpus_windows(tok, SimpleNamespace(
        windows=args.corpus_windows, seq=args.seq, seed=args.seed,
        corpus_chars=args.corpus_chars, corpus_file=args.corpus_file or None,
        agentic_frac=args.agentic_frac))
    cache = torch.load(args.cache_file, map_location="cpu", mmap=True)
    print(f"cache: {len(cache)} windows ({args.cache_file})", flush=True)

    search = list(range(args.windows_search))
    holdout = list(range(args.holdout_offset,
                         args.holdout_offset + args.windows_holdout))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(search)

    tensors = searched_banks(model)
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    for _, p in tensors:
        p.requires_grad_(True)

    # pre-quantise every option (packed on the device); free FP afterwards is
    # not needed -- the FP values stay in the params until the first apply.
    opts, errs, sizes = [], [], []
    for name, p in tensors:
        per = {b: quantize_tensor(p.detach(), b, args.group) for b in options}
        per = {b: {**o, "packed": o["packed"].cpu(), "scales": o["scales"].cpu()}
               for b, o in per.items()}
        opts.append(per)
        errs.append(ternary_rel_error(p, per[options[0]], args.device))
        sizes.append(int(p.numel()))
        print(f"  {name}: {tuple(p.shape)} ternary rel err {errs[-1]:.4f}",
              flush=True)
    costs = torch.tensor([b + SCALE_OVERHEAD for b in options],
                         device=args.device)
    weights = torch.tensor(sizes, dtype=torch.float32, device=args.device)
    target_bits = TERNARY_BPW * args.budget_pct / 100.0
    hand_bits = TERNARY_BPW * 1.0
    print(f"groups {len(tensors)} options {options} costs {costs.tolist()}", flush=True)
    print(f"params {weights.sum().item()/1e9:.3f}B  hand {hand_bits:.4f} bpw  "
          f"target {target_bits:.4f} bpw ({args.budget_pct:.1f}% of hand)",
          flush=True)

    results = {"config": vars(args) | {"options": options}}
    baselines = {}

    def measure(label, assignment):
        apply_assignment(tensors, opts, assignment, args.device)
        loss = eval_loss(model, cwin, cache, holdout, args, vocab)
        bpw = avg_bpw(assignment, options, costs, weights)
        baselines[label] = {"loss": loss, "bpw": bpw,
                            "assignment": [int(b) for b in assignment]}
        print(f"[baseline] {label:10s} loss {loss:.4f} bpw {bpw:.4f}", flush=True)
        return loss

    # hand = all ternary (the floor).  Held-out bookkeeping: this is the
    # reference the RCO arm must beat at equal bytes.
    measure("hand", [options[0]] * len(tensors))
    order_sens = list(np.argsort(errs)[::-1])
    measure("sens", [greedy_assignment(options, costs.cpu(), weights.cpu(),
                                       target_bits, order_sens)[i]
                     for i in range(len(tensors))])
    measure("uniform", [greedy_assignment(options, costs.cpu(), weights.cpu(),
                                          target_bits, list(range(len(tensors))))[i]
                        for i in range(len(tensors))])
    if options[-1] > options[0]:
        measure(f"all{options[-1]}", [options[-1]] * len(tensors))

    if args.steps <= 0:
        results["baselines"] = baselines
        out = Path(args.out) if args.out else MOE / f"rco-alloc-{args.tag or 'smoke'}.json"
        out.write_text(json.dumps(results, indent=1))
        names = [n for n, _ in tensors]
        flat = {n: int(options[0]) for n in names}
        map_path = out.with_name(out.stem + ".map.json")
        map_path.write_text(json.dumps(flat, indent=1))
        print(f"wrote {out} and {map_path}")
        return

    alloc = BudgetAllocator(costs, weights, target_bits, lr=args.lr,
                            device=args.device)
    print(f"init: E[bits] {alloc.expected_bits():.4f}", flush=True)

    state = {"dl_dp": torch.zeros(len(tensors), len(options),
                                  device=args.device)}

    def compute_dl_dp():
        """dL/dp_k = <dL/dW, W_k> per group, after a full backward.

        Deliberately outside autograd's accumulation: reading the bank grads
        inside post-accumulate hooks races with the per-expert
        index-accumulation on the 3-D banks (non-finite reads observed), and
        hooks cannot safely free grads mid-accumulation.  Peak grad memory is
        bounded because each param's grad is consumed and freed here.
        """
        state["dl_dp"].zero_()
        for i, (_, p) in enumerate(tensors):
            g = p.grad
            if g is None:
                continue
            with torch.no_grad():
                for k, b in enumerate(options):
                    o = to_dev(opts[i][b], args.device)
                    state["dl_dp"][i, k] = dot_grad_option(g, o)
                    del o
            p.grad = None
            del g
        return state["dl_dp"]

    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    model.train()
    history, best = [], None
    for step in range(args.steps):
        frac = step / max(args.steps - 1, 1)
        tau = max(args.tau_min, args.tau_init * (args.tau_min / args.tau_init) ** frac)
        hard, soft = alloc.sample(tau, gen)
        hard_labels = [options[int(k)] for k in hard.tolist()]
        apply_assignment(tensors, opts, hard_labels, args.device)

        w = search[step % len(search)]
        ids = cwin[w:w + 1].to(args.device)
        loss, parts = loss_terms(model, ids, cache[w], args, vocab)
        loss.backward()
        dl_dp = compute_dl_dp()
        if step < 2:
            print(f"  [dbg] dl_dp finite {bool(torch.isfinite(dl_dp).all())} "
                  f"absmax {float(dl_dp.abs().max()):.3e} | "
                  f"alpha finite {bool(torch.isfinite(alloc.alpha).all())}",
                  flush=True)
        diag = alloc.step(dl_dp, soft)
        history.append({"step": step, "tau": tau, "loss": float(loss),
                        "hard_bpw": avg_bpw(hard_labels, options, costs, weights),
                        "hard_assignment": hard_labels,
                        **parts, **diag})
        if step % 10 == 0 or step == args.steps - 1:
            print(f"[rco {step:4d}] loss {float(loss):.4f} "
                  f"bits {diag['budget']:.4f} tau {tau:.3f} "
                  f"ent {diag['alpha_entropy']:.2f} decided {diag['decided']}",
                  flush=True)
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            cur = eval_loss(model, cwin, cache, holdout, args, vocab)
            if best is None or cur < best["loss"]:
                best = {"loss": cur, "step": step, "assignment": hard_labels,
                        "bpw": avg_bpw(hard_labels, options, costs, weights)}
                print(f"  [eval] step {step} holdout {cur:.4f} (new best)",
                      flush=True)
            else:
                print(f"  [eval] step {step} holdout {cur:.4f}", flush=True)

    # argmax assignment (unconstrained by the budget; diagnostic only)
    with torch.no_grad():
        argmax_idx = alloc.probs().argmax(-1).tolist()
    argmax = [options[int(k)] for k in argmax_idx]
    apply_assignment(tensors, opts, argmax, args.device)
    argmax_loss = eval_loss(model, cwin, cache, holdout, args, vocab)

    results.update({
        "baselines": baselines,
        "history": history,
        "best": best,
        "argmax": {"loss": argmax_loss, "bpw": avg_bpw(argmax, options, costs, weights),
                   "assignment": [int(b) for b in argmax]},
        "final_alpha_bits": alloc.expected_bits(),
        "elapsed_s": round(time.time() - t0, 1),
    })
    out = Path(args.out) if args.out else MOE / f"rco-alloc-{args.tag or 'run'}.json"
    out.write_text(json.dumps(results, indent=1))
    names = [n for n, _ in tensors]
    best_assignment = (best or {}).get("assignment") or [options[0]] * len(tensors)
    flat = {n: int(b) for n, b in zip(names, best_assignment)}
    map_path = out.with_name(out.stem + ".map.json")
    map_path.write_text(json.dumps(flat, indent=1))
    print(f"wrote {out} and {map_path}", flush=True)

    hand = baselines.get("hand")
    if best and hand:
        d = (best["loss"] - hand["loss"]) / max(hand["loss"], 1e-9)
        print(f"read: RCO best {best['loss']:.4f} @ {best['bpw']:.4f} bpw vs hand "
              f"{hand['loss']:.4f} @ {hand['bpw']:.4f} ({100*d:+.2f}%)", flush=True)


if __name__ == "__main__":
    main()

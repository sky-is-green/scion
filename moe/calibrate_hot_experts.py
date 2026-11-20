"""Calibrate the hot-expert set from router load on the Scion prompt corpus.

Loads the first N layers of the qwen3.5-MoE checkpoint (the same prefix the
correction trainer uses), runs the rollout prompts, and counts the routed
top-k assignments per expert per layer.  Experts are ranked by token load; the
top ``--hot-fraction`` per layer become the Q4_0_G128 set, the rest stay
TQ1_0_G128.  The output map is consumed by the hot/cold body exporter and by
``--alloc-file``-style tooling.

Local validation uses ``--layers 4``.  The full 40-layer map needs the full
checkpoint and is a pod step: the 4-layer prefix here is a ranking/codec test,
not the deployed map.

The local 7900 XT/PyTorch path can fault after a few dozen forwards, so the
script supports ``--start/--end`` shards plus ``--counts-out``; merge the
per-shard ``.npz`` files with ``--merge`` (a glob) without loading the model.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_router import load_prefix  # noqa: E402

DEFAULT_PROMPTS = HERE.parent / "v2" / "corpus" / "rollout-prompts.jsonl"
DEFAULT_OUT = HERE.parent / "v2" / "hotcold" / "hotmap.json"


def load_prompts(path: Path, n: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def encode_prompt(tok, row: dict, max_len: int) -> torch.Tensor:
    msgs = [{"role": "user", "content": row["prompt"]}]
    tools = row.get("tools") or None
    try:
        if not getattr(tok, "chat_template", None):
            raise ValueError("no chat template on this prefix tokenizer")
        text = tok.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=True)
    except (TypeError, ValueError):
        # The prefix tokenizer in the local 4-layer dir carries no template;
        # raw-prompt routing is sufficient for load ranking.  The pod's full
        # checkpoint tokenizer has the Qwen template and will use it.
        text = row["prompt"]
    return tok(text, return_tensors="pt", truncation=True,
               max_length=max_len).input_ids


def calib_error(layer, group: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Per-expert relative L2 error of the deployable 2-bit Lloyd rule."""
    from rco_alloc import quantize_grouped

    out = []
    for attr in ("gate_up_proj", "down_proj"):
        w = getattr(layer.mlp.experts, attr).detach()
        e = w.shape[0]
        err = torch.empty(e, dtype=torch.float32)
        chunk = 8
        for s in range(0, e, chunk):
            part = w[s:s + chunk].to(device=device, dtype=torch.float32)
            q = quantize_grouped(part, 2, group)
            num = (q - part).pow(2).sum(dim=(-2, -1))
            den = part.pow(2).sum(dim=(-2, -1)).clamp_min(1e-12)
            err[s:s + chunk] = (num / den).sqrt().cpu()
            del part, q
        out.append(err.numpy())
    return out[0], out[1]


def run_shard(args) -> None:
    rows_all = load_prompts(Path(args.prompts), args.max_prompts)
    end = len(rows_all) if args.end < 0 else min(args.end, len(rows_all))
    rows = rows_all[args.start:end]
    if not rows:
        raise SystemExit("empty shard")
    print(f"shard {args.start}:{end} ({len(rows)} prompts) from {args.prompts}",
          flush=True)

    model, tok, missing, unexpected = load_prefix(
        Path(args.model_dir), args.layers, args.device)
    print(f"prefix: {args.layers} layers; missing={len(missing)} "
          f"unexpected={len(unexpected)}", flush=True)

    layers = list(model.layers)
    e_count = int(layers[0].mlp.experts.num_experts)
    counts = torch.zeros(len(layers), e_count, dtype=torch.float64, device="cpu")
    n_tokens = 0
    done_ids = []

    def make_hook(i):
        def hook(mod, inputs):
            idx = inputs[1]
            c = torch.bincount(idx.reshape(-1).to(torch.int64),
                               minlength=e_count).to(torch.float64)
            counts[i] += c.cpu()
        return hook

    handles = [layer.mlp.experts.register_forward_pre_hook(make_hook(i))
               for i, layer in enumerate(layers)]
    model.eval()
    with torch.no_grad():
        for n, row in enumerate(rows):
            ids = encode_prompt(tok, row, args.seq_len).to(args.device)
            n_tokens += int(ids.shape[1])
            try:
                model(input_ids=ids, use_cache=False)
            except TypeError:
                model(ids)
            done_ids.append(row.get("id", str(args.start + n)))
            if (n + 1) % 10 == 0 or n + 1 == len(rows):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print(f"  {n+1}/{len(rows)} prompts, {n_tokens} tokens", flush=True)
            del ids
    for h in handles:
        h.remove()

    out = Path(args.counts_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, counts=counts.numpy(), tokens=n_tokens,
             ids=np.array(done_ids), start=args.start, end=end,
             layers=len(layers), experts=e_count, seq_len=args.seq_len)
    print(f"wrote {out}", flush=True)


def build_report(counts: np.ndarray, args, n_tokens: int, n_ids: int,
                 layers=None, meta_extra: dict | None = None) -> dict:
    n_layers, e_count = counts.shape
    if args.hot_per_layer:
        hot_n = int(args.hot_per_layer)
    else:
        hot_n = max(1, int(round(args.hot_fraction * e_count)))
    hot_n = min(hot_n, e_count)

    report = {
        "meta": {
            "model_dir": str(args.model_dir),
            "layers": n_layers,
            "experts": e_count,
            "prompts": n_ids,
            "prompts_file": str(args.prompts),
            "seq_len": args.seq_len,
            "tokens": n_tokens,
            "hot_per_layer": hot_n,
            "hot_fraction": hot_n / e_count,
            "group": args.group,
            "ranking": "load_x_error" if args.error_weight else "load",
            "note": "4-layer local map is a ranking/codec test; the deployed "
                    "map must be regenerated on the full checkpoint (--layers 40)",
        },
        "layers": {},
    }
    if meta_extra:
        report["meta"].update(meta_extra)

    total_hot_mass = 0.0
    total_mass = 0.0
    for i in range(n_layers):
        load = counts[i].astype(np.float64)
        total = load.sum()
        total_mass += total
        score = load.copy()
        err_gu = err_dn = None
        if args.error_weight:
            if layers is None:
                raise SystemExit("--error-weight needs the model; run on a shard "
                                 "or without --merge")
            err_gu, err_dn = calib_error(layers[i], args.group,
                                         torch.device(args.device))
            score = load * (1.0 + 0.5 * (err_gu + err_dn))
        order = np.argsort(-score, kind="stable")
        hot = np.sort(order[:hot_n]).tolist()
        cold = np.sort(order[hot_n:]).tolist()
        hot_mass = float(load[hot].sum())
        total_hot_mass += hot_mass
        entry = {
            "tokens": int(total),
            "hot": hot,
            "cold": cold,
            "load": [int(x) for x in load.tolist()],
            "hot_mass": hot_mass,
            "hot_share_of_routed": hot_mass / max(total, 1.0),
            "dead": int((load == 0).sum()),
            "load_min": float(load.min()),
            "load_median": float(np.median(load)),
            "load_max": float(load.max()),
        }
        if err_gu is not None:
            entry["err_q2_gate_up"] = [float(x) for x in err_gu]
            entry["err_q2_down"] = [float(x) for x in err_dn]
        report["layers"][str(i)] = entry
        print(f"L{i:02d}: hot {hot_n}/{e_count} -> "
              f"{100.0 * hot_mass / max(total, 1.0):.1f}% of routed tokens; "
              f"dead {entry['dead']}; load med {entry['load_median']:.0f} "
              f"max {entry['load_max']:.0f}", flush=True)

    report["meta"]["hot_share_of_routed"] = total_hot_mass / max(total_mass, 1.0)
    return report


def run_merge(args) -> None:
    paths = sorted(glob.glob(args.merge))
    if not paths:
        raise SystemExit(f"no counts files match {args.merge}")
    counts = None
    n_tokens = 0
    n_ids = 0
    for p in paths:
        z = np.load(p, allow_pickle=True)
        c = z["counts"]
        counts = c if counts is None else counts + c
        n_tokens += int(z["tokens"])
        n_ids += len(z["ids"])
        print(f"merged {p}: {c.shape}, {int(z['tokens'])} tokens", flush=True)
    report = build_report(counts, args, n_tokens, n_ids,
                          meta_extra={"shards": len(paths)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"hot coverage: {100.0 * report['meta']['hot_share_of_routed']:.1f}% "
          f"of all routed tokens", flush=True)
    print(f"wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--max-prompts", type=int, default=360)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--counts-out", default="",
                    help="save this shard's counts and exit (no map)")
    ap.add_argument("--merge", default="",
                    help="glob of per-shard .npz files; build the map and exit")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--hot-fraction", type=float, default=0.16)
    ap.add_argument("--hot-per-layer", type=int, default=0,
                    help="explicit hot count; 0 uses round(hot_fraction*E)")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--error-weight", action="store_true",
                    help="rank by load * (1 + 2-bit relative error)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if args.merge:
        run_merge(args)
        return
    if args.counts_out:
        run_shard(args)
        return

    # Single-process path (small layer counts / stable GPUs).
    rows = load_prompts(Path(args.prompts), args.max_prompts)
    print(f"prompts: {len(rows)} from {args.prompts}", flush=True)
    model, tok, missing, unexpected = load_prefix(
        Path(args.model_dir), args.layers, args.device)
    print(f"prefix: {args.layers} layers; missing={len(missing)} "
          f"unexpected={len(unexpected)}", flush=True)
    layers = list(model.layers)
    e_count = int(layers[0].mlp.experts.num_experts)
    counts = torch.zeros(len(layers), e_count, dtype=torch.float64, device="cpu")
    n_tokens = 0

    def make_hook(i):
        def hook(mod, inputs):
            idx = inputs[1]
            c = torch.bincount(idx.reshape(-1).to(torch.int64),
                               minlength=e_count).to(torch.float64)
            counts[i] += c.cpu()
        return hook

    handles = [layer.mlp.experts.register_forward_pre_hook(make_hook(i))
               for i, layer in enumerate(layers)]
    model.eval()
    with torch.no_grad():
        for n, row in enumerate(rows):
            ids = encode_prompt(tok, row, args.seq_len).to(args.device)
            n_tokens += int(ids.shape[1])
            try:
                model(input_ids=ids, use_cache=False)
            except TypeError:
                model(ids)
            if (n + 1) % 25 == 0 or n + 1 == len(rows):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print(f"  {n+1}/{len(rows)} prompts, {n_tokens} tokens", flush=True)
            del ids
    for h in handles:
        h.remove()

    report = build_report(counts.numpy(), args, n_tokens, len(rows), layers=layers)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"hot coverage: {100.0 * report['meta']['hot_share_of_routed']:.1f}% "
          f"of all routed tokens", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

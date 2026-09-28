"""DeltaLoss steering ranker (SignRoundV2) over the Qwen3.5-MoE prefix.

W3 diagnostic.  AUTOGRID classifies every tensor FREE / TERNARY / STEER and
refuses the STEER ones because they need a steered pipeline.  ``steer.py`` in
the AUTOGRID fork gives a per-tensor sensitivity -- ``DeltaLoss =
|| g_w * (W_q - W) ||_1`` with the gradient taken from an output-reconstruction
loss on the *deployed* quantizer.  This runner walks the prefix, captures each
linear's real calibration input, and writes the ranking so it can be compared
against the routing-drift placement findings (the E1 rule).

Scope note: the sweep is over ``nn.Linear`` modules, which in this architecture
is exactly the set the recipe leaves in FP -- ``self_attn.{q,k,v,o}_proj``,
``linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}`` and
``mlp.shared_expert.{gate,up,down}_proj``.  The fused expert banks
(``mlp.experts.{gate_up,down}_proj``) and the router ``mlp.gate`` are 3-D
parameters, not Linear, and are the ternary *target* rather than STEER
candidates; they are deliberately out of scope here.

Example:

    PYTHONPATH=~/Desktop/work/autogrid HIP_VISIBLE_DEVICES=1 \\
    python moe/steer_probe.py --prefix-layers 4 --device cuda:0 \\
        --quantizer lloyd --group 128 --out $MOE/steer-rank.json

The pure pieces (row subsampling, role annotation, ranking/normalisation) have
no transformers or autogrid import, so they are unit-tested on CPU in
``moe/tests/test_steer_probe.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ART = Path(os.environ.get("MOE_ARTIFACTS", HERE / "artifacts"))
OUT = ART / "qwen35"
AUTOGRID = Path(os.environ.get("AUTOGRID_REPO", Path.home() / "Desktop/work/autogrid"))


# ------------------------------------------------------------------ pure -----

def add_autogrid(path: Path = AUTOGRID) -> None:
    """Put the AUTOGRID fork on ``sys.path`` so ``autogrid_ext`` imports."""
    p = str(Path(path).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def quantizer_for(name: str, group: int = 128):
    """The deployed container quantizer, as a ``Tensor -> Tensor`` callable.

    ``lloyd`` is the fixed point the TAARDIS Q1_0_g128 container actually uses,
    so it is the default: DeltaLoss has to be measured against what ships.
    """
    from autogrid_ext.containers import ternary_absmean, ternary_lloyd
    if name == "lloyd":
        return lambda w: ternary_lloyd(w, group)
    if name == "absmean":
        return lambda w: ternary_absmean(w, group)
    raise KeyError(name)


def capture_hook(store: dict, name: str, max_rows: int, seed: int):
    """Forward *pre*-hook that keeps a bounded sample of the module's input.

    ``deltaloss_linear`` wants the linear's input, not its output, so this is a
    pre-hook.  Rows are flattened to ``[tokens, in_features]`` and subsampled
    once per module with a fixed generator, so repeated runs on the same windows
    rank identically.
    """
    def hook(mod, args):
        if name in store:
            return
        x = args[0].detach()
        x = x.reshape(-1, x.shape[-1]).float()
        if max_rows and x.shape[0] > max_rows:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(x.shape[0], generator=g)[:max_rows]
            x = x[idx]
        store[name] = x.cpu()
    return hook


_ROLE_PATTERNS = (
    ("attn", re.compile(r"self_attn\.(q_proj|k_proj|v_proj|o_proj)")),
    ("gdn", re.compile(r"linear_attn\.(in_proj_\w+|out_proj)")),
    ("shared_expert", re.compile(r"mlp\.shared_expert\.(gate_proj|up_proj|down_proj)")),
    # the shared-expert gate is a separate weight from mlp.gate (the router);
    # it is a real nn.Linear, so it is swept, and it is not "other"
    ("shared_expert_gate", re.compile(r"mlp\.shared_expert_gate$")),
    ("expert_bank", re.compile(r"experts\.(gate_up_proj|down_proj)")),
    ("router", re.compile(r"mlp\.gate")),
)


def role_of(name: str) -> str:
    """Coarse role label, so the ranking can be read per placement site."""
    for role, pat in _ROLE_PATTERNS:
        if pat.search(name):
            return role
    return "other"


def layer_of(name: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def rank_entries(entries: list[dict], key: str = "delta_loss") -> list[dict]:
    """Sort by ``key`` descending and add a share-of-total + normalised score.

    DeltaLoss is an L1 sum, so it scales with tensor size; ``share`` (fraction
    of the summed total) and ``norm`` (value / max) are what make tensors of
    different shapes comparable in one list.
    """
    out = [dict(e) for e in entries]
    out.sort(key=lambda e: e[key], reverse=True)
    total = sum(e[key] for e in out) or 1.0
    top = out[0][key] or 1.0
    for rank, e in enumerate(out, 1):
        e["rank"] = rank
        e["share"] = e[key] / total
        e["norm"] = e[key] / top
    return out


# -------------------------------------------------------------------- run ----

def linear_targets(model) -> list[tuple[str, torch.nn.Linear]]:
    """``(name, module)`` for every Linear in the decoder layers, in order."""
    from qwen35_moe_proxy import text_layers
    out = []
    for i, layer in enumerate(text_layers(model)):
        for name, mod in layer.named_modules():
            if isinstance(mod, torch.nn.Linear):
                out.append((f"layers.{i}.{name}", mod))
    return out


@torch.no_grad()
def collect(model, data, device, max_rows: int, seed: int) -> dict:
    """Run the calibration windows and park one input sample per linear."""
    from qwen35_moe_proxy import model_logits
    store: dict[str, torch.Tensor] = {}
    targets = linear_targets(model)
    handles = [mod.register_forward_pre_hook(capture_hook(store, name, max_rows, seed))
               for name, mod in targets]
    try:
        for i in range(len(data)):
            model_logits(model, data[i:i + 1].to(device))
            print(f"  capture {i+1}/{len(data)} ({len(store)} linears)", flush=True)
    finally:
        for h in handles:
            h.remove()
    return store


def score_all(model, store: dict, quantizer, skip_shapes: bool = True) -> list[dict]:
    """DeltaLoss per captured linear, against the deployed quantizer.

    Deliberately NOT under ``torch.no_grad``: DeltaLoss is defined in terms of
    the gradient of the reconstruction loss, so grad mode has to stay on for
    ``deltaloss_linear``.  Only the weight copies are taken under no_grad.
    """
    from autogrid_ext.steer import deltaloss_linear
    weights = dict(linear_targets(model))
    entries = []
    for name, x in store.items():
        mod = weights[name]
        # deltaloss_linear builds an autograd graph, so both operands have to sit
        # on the same device; the captured inputs are parked on the CPU.  The
        # weight itself is only read, so it never needs to be a leaf of the graph.
        with torch.no_grad():
            w = mod.weight.detach().to(x.device)
        if skip_shapes and (w.shape[-1] % 128):
            # a g128 container cannot represent this row length at all
            entries.append({"name": name, "role": role_of(name),
                            "layer": layer_of(name),
                            "delta_loss": 0.0,
                            "skipped": f"in_features {w.shape[-1]} not a multiple of 128"})
            continue
        val = deltaloss_linear(w, x, quantizer)
        entries.append({"name": name, "role": role_of(name), "layer": layer_of(name),
                        "shape": list(w.shape), "rows": int(x.shape[0]),
                        "in_features": int(w.shape[-1]),
                        "delta_loss": val})
        print(f"  {name}: {val:.6g}", flush=True)
        del w
    return rank_entries(entries)


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as a function so tests can check flags without loading a model."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix-layers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--windows", type=int, default=2,
                    help="calibration windows (seed 0, fineweb -- training-corpus input)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--corpus-chars", type=int, default=50_000_000)
    ap.add_argument("--max-rows", type=int, default=512,
                    help="token rows kept per linear (0 = keep all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quantizer", choices=["lloyd", "absmean"], default="lloyd")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--autogrid", default=str(AUTOGRID), help="AUTOGRID fork checkout")
    ap.add_argument("--model-dir", default="", help="override $MOE_ARTIFACTS/empero-hf")
    ap.add_argument("--out", default="", help="write the ranked JSON here")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    add_autogrid(args.autogrid)
    try:
        import autogrid_ext.steer  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"cannot import autogrid_ext from {args.autogrid}: {exc}\n"
                         f"pass --autogrid or set AUTOGRID_REPO")

    from transformers import AutoTokenizer

    from olmoe_proxy import windows
    from qwen35_moe_proxy import MODEL, load_prefix

    model_dir = Path(args.model_dir) if args.model_dir else MODEL
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"no checkpoint at {model_dir}; pass --model-dir or set "
                         f"MOE_ARTIFACTS (the FP empero-hf shards are needed)")
    tok = AutoTokenizer.from_pretrained(model_dir)
    data = windows(tok, args.windows, args.seq, args.seed, "fineweb",
                   max_chars=args.corpus_chars)

    print(f"steer probe: {args.prefix_layers}-layer prefix, {args.quantizer} "
          f"g{args.group}, {len(data)} windows, max_rows {args.max_rows}", flush=True)
    model, _, _, _ = load_prefix(args.prefix_layers, args.device, model_dir=model_dir)
    store = collect(model, data, args.device, args.max_rows, args.seed)
    print(f"captured {len(store)} linears", flush=True)

    entries = score_all(model, store, quantizer_for(args.quantizer, args.group))
    res = {"prefix_layers": args.prefix_layers, "quantizer": args.quantizer,
           "group": args.group, "windows": args.windows, "seq": args.seq,
           "seed": args.seed, "max_rows": args.max_rows, "ranked": entries}

    by_role: dict[str, list[float]] = {}
    for e in entries:
        by_role.setdefault(e["role"], []).append(e.get("norm", 0.0))
    res["role_summary"] = {r: {"n": len(v), "mean_norm": sum(v) / len(v)}
                          for r, v in sorted(by_role.items())}
    print(json.dumps(res["role_summary"], indent=2), flush=True)

    out = Path(args.out) if args.out else OUT / "steer-rank.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

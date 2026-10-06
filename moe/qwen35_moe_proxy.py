"""Qwen3.5-MoE (35B-A3B) ternary proxy: fused expert banks + correction branches.

Port of ``olmoe_proxy.py`` / ``olmoe_corrections.py`` to the ``qwen3_5_moe``
architecture:

  - experts are fused parameters (``experts.gate_up_proj`` /
    ``experts.down_proj``), ternarised in place (STE), 256 experts top-8;
  - the router is ``layer.mlp.gate`` (not an ``nn.Linear``); the shared expert
    and the attention/GDN projections stay FP for the first cut;
  - 40 layers (30 Gated-DeltaNet, 10 full attention), hidden 2048;
  - corrections are residual-stream branches on each ``layer.mlp`` output plus
    trainable routers; loss = LM + output KD (router KD optional, default 0).

Stages:
  smoke : local prefix check (embedding + N layers from a partial checkpoint)
          of the patched forward and a few correction steps; no full model.
  cache : FP teacher pass -> per-window top-50 logits + per-layer router top-8.
  train : frozen ternary body + correction branches + routers.
  eval  : held-out PPL and per-layer router agreement.

Quant-methods options (all default to the frozen v1 recipe): ``--quant catq``
(CAT-Q body reconstruction), ``--kd-filter-frac`` (SignRoundV2 loss
filtering), ``--corpus-file``/``--agentic-frac`` (AYOT reasoning-trace
mixing), ``--top-logits`` (tail-plan step 1).

Ops: training needs the FP teacher and ternary student resident together, so it
needs a card that holds both (or two cards via ``--device-map auto`` with
``--max-memory``); single-GPU stages should pin a non-display GPU (for example
``HIP_VISIBLE_DEVICES=1``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moe_proxy import ternary_absmean  # noqa: E402
from olmoe_corrections import (CorrectionBranch, MoEWithCorrection,  # noqa: E402
                               gate_stats, load_branch_state, moe_block,
                               quantize_bank_inplace)
from olmoe_proxy import gate_hook, ternary_ste, windows  # noqa: E402
from ayot import load_traces, mix_windows, windows_from_texts  # noqa: E402
from kd_loss import (kd_filtered, residual_mass_kl, sample_tail_tokens,  # noqa: E402
                     support_mass, support_mass_lse, tail_conditional_piece)
from mtp import (MTPHead, chunked_ce, chunked_kl, draft_acceptance,  # noqa: E402
                 freeze_except_head, mtp_logits, mtp_targets,
                 self_target_main)
from router_bias import (balance_update, balance_z_loss,  # noqa: E402
                         patch_router_balance)

ART = Path(os.environ.get("MOE_ARTIFACTS", HERE / "artifacts"))
MODEL = ART / "empero-hf"          # full or partial qwen3_5_moe checkpoint
OUT = ART / "qwen35"
CACHE = OUT / "teacher-cache.pt"


# ------------------------------------------------------------------- patch ---

def patch_experts(group: int, model=None):
    """Make ``Qwen3_5MoeExperts`` ternarise its fused banks on the fly (STE).

    ``device_map="auto"`` binds the pre-patch forward onto each experts
    instance, shadowing the class patch; pass the loaded ``model`` to rebind
    those instances too.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

    def ternary_forward(self, hidden_states, top_k_index, top_k_weights):
        gu = ternary_ste(self.gate_up_proj, group)
        dn = ternary_ste(self.down_proj, group)
        dt = gu.dtype
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index,
                                                      num_classes=self.num_experts)
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

    original = Qwen3_5MoeExperts.forward

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if not getattr(self, "_ternary", False):
            return original(self, hidden_states, top_k_index, top_k_weights)
        return ternary_forward(self, hidden_states, top_k_index, top_k_weights)

    Qwen3_5MoeExperts.forward = forward

    if model is not None:
        for m in model.modules():
            if not isinstance(m, Qwen3_5MoeExperts):
                continue
            inner = m.__dict__.get("forward")

            def make(inner):
                def f(self, *a, **k):
                    if not getattr(self, "_ternary", False):
                        return inner(*a, **k)
                    return ternary_forward(self, *a, **k)
                return f

            m.forward = (make(inner).__get__(m, type(m)) if inner is not None
                         else forward.__get__(m, type(m)))


def text_layers(model):
    """Return the decoder-layer list for text-model and conditional wrappers."""
    root = model
    if hasattr(root, "language_model"):
        return root.language_model.layers
    if hasattr(root, "layers"):
        return root.layers
    return root.model.language_model.layers


def model_logits(model, ids):
    """Logits for the conditional wrapper (.logits) or the bare text prefix."""
    out = model(input_ids=ids, use_cache=False)
    logits = getattr(out, "logits", None)
    if logits is None:
        logits = model.lm_head(out.last_hidden_state)
    return logits


def model_hidden_logits(model, ids):
    """Hidden states + logits in one forward (the MTP head needs the hidden states).

    Text-prefix only: the conditional wrapper does not expose
    ``last_hidden_state`` without ``output_hidden_states``, and the MTP
    experiment runs on the prefix.
    """
    out = model(input_ids=ids, use_cache=False)
    h = getattr(out, "last_hidden_state", None)
    if h is None:
        raise AttributeError("model_hidden_logits needs a text model "
                             "(no last_hidden_state on this output)")
    logits = getattr(out, "logits", None)
    if logits is None:
        logits = model.lm_head(h)
    return h, logits


def _corpus_windows(tok, args):
    """Training windows; ``--corpus-file`` mixes in AYOT agentic traces."""
    data = windows(tok, args.windows, args.seq, args.seed, max_chars=args.corpus_chars)
    if args.corpus_file:
        agentic = windows_from_texts(tok, load_traces(args.corpus_file), args.windows,
                                     args.seq, args.seed)
        data = mix_windows(data, agentic, args.agentic_frac, args.seed)
        print(f"corpus: {int(round(args.windows * args.agentic_frac))}/{args.windows} "
              f"windows from {args.corpus_file}", flush=True)
    return data


def _quant_kwargs(args):
    """CAT-Q knobs for ``quantize_bank_inplace`` when ``--quant catq``."""
    if args.quant != "catq":
        return {}
    return {"catq_kw": {"steps": args.catq_steps, "lr": args.catq_lr,
                        "gamma": args.catq_gamma, "s0": args.catq_s0}}


@torch.no_grad()
def mtp_acceptance(model, head, data, args, topk: int = 1) -> float:
    """Greedy draft acceptance of the MTP head on held-out windows.

    ``main[:, t]`` predicts token ``t+1`` and the head at ``t`` predicts
    ``t+2``; acceptance is the fraction where the head's argmax matches the main
    model's argmax for the same token (``draft_acceptance``).
    """
    model.eval()
    emb = model.get_input_embeddings()
    accs = []
    for i in range(len(data)):
        ids = data[i:i + 1].to(args.device)
        h, logits = model_hidden_logits(model, ids)
        h_in, e_in, _ = mtp_targets(h, ids)
        mlogits = mtp_logits(model, head, h_in, emb(e_in))
        accs.append(draft_acceptance(logits, mlogits, topk))
    model.train()
    return sum(accs) / len(accs)


@torch.no_grad()
def quick_eval(model, data, args, ref=None):
    """In-run eval (qwen35-aware: text prefix or full conditional wrapper)."""
    model.eval()
    total, ntok, agree = 0.0, 0, []
    layers = text_layers(model)
    for i in range(len(data)):
        ids = data[i:i + 1].to(args.device)
        store = {}
        hs = [moe_block(layer).gate.register_forward_hook(gate_hook(store, j))
              for j, layer in enumerate(layers)]
        logits = model_logits(model, ids)
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


# ---------------------------------------------------------------- prefix -----

def load_prefix(n_layers: int, device: str = "cuda:0", dtype=torch.bfloat16,
                model_dir: Path | None = None):
    """Load embedding + N decoder layers (+ norm/lm_head when present).

    Enough for the smoke test; reads only the local shards the index maps.
    """
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel
    from safetensors import safe_open

    model_dir = Path(model_dir or MODEL)
    cfg = AutoConfig.from_pretrained(model_dir)
    tcfg = cfg.text_config
    tcfg.num_hidden_layers = n_layers
    tcfg.layer_types = list(tcfg.layer_types)[:n_layers]
    if hasattr(tcfg, "mtp_num_hidden_layers"):
        tcfg.mtp_num_hidden_layers = 0
    # Construct in bf16, not the default fp32: the fp32 construction doubles
    # the host peak (17.5 GB vs 8.8 GB for the 4-layer prefix) and that peak is
    # the binding host constraint for every prefix stage (kld_eval's memory
    # guard reserves for it).  The GPU model is bf16 either way -- loading bf16
    # into fp32 and converting back is an identity round-trip -- so the weights
    # are bit-identical to the old path.
    _prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen3_5MoeTextModel(tcfg)
    finally:
        torch.set_default_dtype(_prev_dtype)
    tok = AutoTokenizer.from_pretrained(model_dir)

    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    handles, state, skipped = {}, {}, set()
    for key, shard in idx.items():
        if key.startswith("model.language_model."):
            stripped = key[len("model.language_model."):]
            if stripped.startswith("layers."):
                if int(stripped.split(".")[1]) >= n_layers:
                    continue
            elif stripped not in ("embed_tokens.weight", "norm.weight"):
                continue
        elif key == "lm_head.weight":
            stripped = "lm_head.weight"
        else:
            continue
        path = model_dir / shard
        if not path.exists():
            skipped.add(shard)
            continue
        h = handles.get(shard)
        if h is None:
            h = handles[shard] = safe_open(path, framework="pt", device="cpu")
        state[stripped] = h.get_tensor(key)
    if skipped:
        print(f"warning: {len(skipped)} shard(s) not on disk; skipped "
              f"({len(state)} tensors loaded)", flush=True)
    if "lm_head.weight" in state:
        model.lm_head = torch.nn.Linear(tcfg.hidden_size,
                                        state["lm_head.weight"].shape[0], bias=False)
    # assign=True takes the loaded tensors directly instead of copying them
    # into the constructed params: one resident copy instead of two (the
    # 4-layer prefix holds 8.8 GB instead of ~17.5 GB through the load).
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    del state
    # A skipped shard can leave the final norm at its zero init; make it
    # identity so the smoke's hidden states are meaningful.  Replace the
    # parameter rather than filling it in place: with assign=True the loaded
    # tensors may be backed by the safetensors mmap, and an in-place write
    # would touch the checkpoint file.
    final_norm = getattr(model, "norm", None)
    if final_norm is not None and final_norm.weight.abs().sum().item() == 0:
        final_norm.weight = torch.nn.Parameter(
            torch.ones_like(final_norm.weight), requires_grad=False)
    model.to(dtype=dtype, device=device)
    model.eval()
    return model, tok, missing, unexpected


# ------------------------------------------------------------------- smoke ---

def smoke(args):
    patch_experts(args.group)
    model, tok, missing, unexpected = load_prefix(args.layers, args.device)
    print(f"prefix: {args.layers} layers; missing={len(missing)} "
          f"unexpected={len(unexpected)}", flush=True)
    if not hasattr(model, "lm_head"):
        raise SystemExit("prefix has no lm_head; cannot run the LM-loss smoke")

    data = windows(tok, 2, args.seq, 999, "wikitext")
    ids = data[0:1].to(args.device)

    def capture():
        routes = {}
        handles = [layer.mlp.gate.register_forward_hook(gate_hook(routes, i))
                   for i, layer in enumerate(model.layers)]
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        for h in handles:
            h.remove()
        return out, routes

    experts = [layer.mlp.experts for layer in model.layers]
    for e in experts:
        e._ternary = False
    fp_out, fp_route = capture()
    for e in experts:
        e._ternary = True
    tern_out, tern_route = capture()
    drift = float((tern_out.last_hidden_state - fp_out.last_hidden_state).norm()
                  / (fp_out.last_hidden_state.norm() + 1e-12))
    agree = float(torch.stack([
        (fp_route[i][2].unsqueeze(-1) == tern_route[i][2].unsqueeze(-2)).any(-1).float().mean()
        for i in fp_route]).mean())
    print(f"FP vs ternary: hidden drift {drift:.4f}, router top-8 agreement {agree:.4f}",
          flush=True)

    # correction branches on each MoE block output + trainable routers
    hidden = model.config.hidden_size
    for layer in model.layers:
        layer.mlp = MoEWithCorrection(layer.mlp, hidden, args.rank, args.branch_quant).to(args.device)
    for name, p in model.named_parameters():
        p.requires_grad_((".branch." in name) or ("mlp.gate." in name))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_tr/1e6:.2f}M (branches + routers)", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adafactor(params, lr=args.lr, weight_decay=0.0)
    model.train()
    for step in range(1, args.steps + 1):
        hidden = model(input_ids=ids, use_cache=False).last_hidden_state
        logits = model.lm_head(hidden)
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                             ids[:, 1:].reshape(-1))
        lm.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 5 == 0 or step == 1:
            print(f"smoke step {step}: lm {lm.item():.4f}", flush=True)
    print("smoke ok", flush=True)


# ------------------------------------------------------------------- cache ---

@torch.no_grad()
def stage_cache(args):
    model, tok = load_full(args)
    data = _corpus_windows(tok, args)
    recs = []
    for w in range(len(data)):
        ids = data[w:w + 1].to(args.device)
        store = {}
        handles = [layer.mlp.gate.register_forward_hook(gate_hook(store, i))
                   for i, layer in enumerate(text_layers(model))]
        logits = model_logits(model, ids)
        for h in handles:
            h.remove()
        t_top = logits[:, :-1].topk(args.top_logits, dim=-1)
        router = {i: (store[i][2].cpu().to(torch.int16),
                      store[i][1].cpu().to(torch.float16)) for i in store}
        # wmass: the teacher's own mass on the support it was cached at.  The
        # top-k KD term renormalises over the support and is blind to this
        # number, which is the whole point of --kd-tail-weight (tail plan 2a).
        # Recorded here because the full logsumexp is already in hand -- the
        # topk above consumed the same full-width logits -- so it is free, and
        # the tail term comes out exact instead of sampled.
        wmass = support_mass(logits[:, :-1], t_top.indices)
        rec = {"idx": t_top.indices.cpu().to(torch.int32),
               "val": t_top.values.cpu().to(torch.float16),
               "w": wmass.cpu().to(torch.float16),
               "router": router}
        if args.tail_logits:
            # TAD's D_KL2 needs probabilities a top-k cache cannot hold, so the
            # teacher's tail conditional is sampled here (Sparse Logit
            # Sampling): m tokens per position + their conditional log-probs.
            # The generator is seeded per window so a cache rebuilt with the
            # same flags samples the same tokens -- same discipline as the
            # window corpus itself.
            gen = torch.Generator(device=logits.device).manual_seed(
                int(args.seed) * 1_000_003 + w)
            tidx, tlp = sample_tail_tokens(logits[:, :-1], t_top.indices,
                                           args.tail_logits, generator=gen)
            rec["tidx"] = tidx.cpu().to(torch.int32)
            rec["tlp"] = tlp.cpu().to(torch.float16)
        recs.append(rec)
        if w % 100 == 0:
            print(f"cached {w}/{len(data)} (teacher support mass "
                  f"{float(wmass.mean()):.4f})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = Path(args.cache_file) if args.cache_file else CACHE
    torch.save(recs, path)
    print(f"wrote {path} ({path.stat().st_size/1e9:.2f} GB)", flush=True)


@torch.no_grad()
def build_ref(args):
    """Precompute teacher router refs for the in-run eval windows."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = windows(tok, 2, args.seq, 999, "wikitext")
    model, _ = load_full(args)
    ref = {}
    for i in range(len(data)):
        ids = data[i:i + 1].to(args.device)
        store = {}
        hs = [layer.mlp.gate.register_forward_hook(gate_hook(store, j))
              for j, layer in enumerate(text_layers(model))]
        model(ids)
        for h in hs:
            h.remove()
        ref[i] = {j: store[j][2].cpu() for j in store}
    torch.save(ref, args.ref_file)
    print(f"wrote {args.ref_file} ({len(ref)} windows)", flush=True)


# ------------------------------------------------------------------- train ---

def load_full(args):
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if getattr(args, "prefix_layers", 0):
        # local dry runs: an N-layer prefix loads from the shards on disk
        model, _, _, _ = load_prefix(args.prefix_layers, args.device, model_dir=MODEL)
        return model, tok
    mm = None
    if args.max_memory:
        mm = {int(k): v for k, v in (part.split(":") for part in args.max_memory.split(","))}
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=args.device_map, max_memory=mm)
    return model, tok


def ternarize_banks(model, args) -> None:
    """Freeze the expert banks in place under the selected scale rule.

    ``--alloc-file`` (the RCO port) replaces the all-ternary hand
    map with a per-bank bit assignment: 2 = the deployed Lloyd ternary rule,
    4/6/8 = symmetric g128 integer codes with fp16 scales.  Default off, so
    the frozen v1 path is untouched.
    """
    alloc = {}
    if getattr(args, "alloc_file", ""):
        from rco_alloc import quantize_grouped
        alloc = json.loads(Path(args.alloc_file).read_text())
        print(f"alloc: {len(alloc)} entries from {args.alloc_file}", flush=True)
    with torch.no_grad():
        for i, layer in enumerate(text_layers(model)):
            for name, proj in (("gate_up", layer.mlp.experts.gate_up_proj),
                               ("down", layer.mlp.experts.down_proj)):
                key = f"blk.{i}.ffn_{name}_exps"
                bits = int(alloc.get(key, 2))
                if bits == 2:
                    quantize_bank_inplace(proj, args.group, kind=args.quant,
                                          **_quant_kwargs(args))
                else:
                    proj.copy_(quantize_grouped(proj, bits, args.group))
                if alloc:
                    print(f"  {key}: {bits}-bit", flush=True)
            layer.mlp.experts._ternary = False          # banks already quantised


def attach_branches(model, args) -> int:
    """Wrap the MoE / attention outputs in correction branches.  Returns n_trainable.

    Placement map (unchanged since v1): ``moe_out`` wraps ``layer.mlp``;
    ``attn_out`` wraps the GDN ``linear_attn.out_proj`` (ssm_out in GGUF) or the
    full-attention ``self_attn.o_proj`` (attn_output).  Both project
    value_dim -> hidden, so the branch takes explicit in/out dims.
    """
    layers = text_layers(model)
    hidden = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    target = args.branch_target
    gate = getattr(args, "branch_gate", "none")
    for layer in layers:
        dev = next(layer.mlp.parameters()).device
        if target in ("moe_out", "both"):
            layer.mlp = MoEWithCorrection(layer.mlp, hidden, args.rank,
                                          args.branch_quant, args.quant,
                                          gate=gate).to(dev)
        if target in ("attn_out", "both"):
            if getattr(layer, "layer_type", "") == "linear_attention":
                proj = layer.linear_attn.out_proj
                layer.linear_attn.out_proj = MoEWithCorrection(
                    proj, proj.in_features, args.rank, args.branch_quant,
                    args.quant, out_dim=proj.out_features, gate=gate).to(dev)
            else:
                proj = layer.self_attn.o_proj
                layer.self_attn.o_proj = MoEWithCorrection(
                    proj, proj.in_features, args.rank, args.branch_quant, args.quant,
                    out_dim=proj.out_features, gate=gate).to(dev)
    if gate != "none":
        # Phase C build: gates are created 1+tanh(0) == 1, so the arm starts
        # bit-identical to the ungated branch and the comparison is one-variable
        # by construction.
        print(f"branch gates: {gate} (identity at init)", flush=True)
    for name, p in model.named_parameters():
        p.requires_grad_(".branch." in name or ".gate." in name)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_student(model, args) -> int:
    """The deployable student: ternarised banks + correction branches.

    Shared by the train/eval stages and ``kld_eval`` so the W1 gate instrument
    cannot drift from the stage it is measuring.
    """
    ternarize_banks(model, args)
    return attach_branches(model, args)


def stage_train(args):
    patch_experts(args.group)
    model, tok = load_full(args)
    n_tr = build_student(model, args)
    print(f"trainable {n_tr/1e6:.2f}M (branches + routers) target {args.branch_target}",
          flush=True)
    if args.balance != "none":
        # Phase B: bias-based balancing patches the gate forward and keeps a
        # per-expert bias buffer; the stock v1 path (none) is untouched.
        n_gates = patch_router_balance(model, args.balance,
                                       args.balance_cb_eta, args.balance_qb_damp)
        print(f"router balance: {args.balance} on {n_gates} gates "
              f"(delta {args.balance_delta}, z {args.balance_z_coeff})", flush=True)
    head = None
    if args.mtp_weight > 0:
        tcfg = getattr(model.config, "text_config", model.config)
        head = MTPHead(int(tcfg.hidden_size), layers=args.mtp_head_layers).to(args.device)
        model._mtp_head = head          # nn.Module attribute -> in model.parameters()
        print(f"MTP head: {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M "
              f"params, layers {args.mtp_head_layers}, weight {args.mtp_weight}, "
              f"target {args.mtp_target}", flush=True)
        if args.mtp_only:
            # Frozen-body drafter: the correction recipe is final, the head
            # trains alone (no perturbation of the main gate) and only its
            # loss backprops -- see the mtp block for the loss override.
            n_head = freeze_except_head(model, head)
            print(f"MTP-only: body frozen, {n_head/1e6:.1f}M head params train",
                  flush=True)

    cache_path = Path(args.cache_file) if args.cache_file else CACHE
    cache = torch.load(cache_path, map_location="cpu")
    data = _corpus_windows(tok, args)

    ref = None
    ev = None
    if args.eval_every:
        if not args.ref_file:
            raise SystemExit("--eval-every needs --ref-file (build with the 'ref' stage)")
        ev = windows(tok, 2, args.seq, 999, "wikitext")
        ref = torch.load(args.ref_file, map_location="cpu")
        print(f"loaded eval refs from {args.ref_file}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    else:
        opt = torch.optim.Adafactor(params, lr=args.lr, weight_decay=0.0)
    model.train()
    step = 0
    if args.resume:
        missing, unexpected = load_branch_state(model, args.resume)
        m = re.search(r"step(\d+)", args.resume)
        step = int(m.group(1)) if m else 0
        print(f"resumed {args.resume} at step {step}: missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
    for epoch in range(args.epochs):
        for rec in cache:
            bal, zl, mtp = None, None, None
            ids = data[step % len(data):step % len(data) + 1].to(args.device)
            if head is not None:
                h, logits = model_hidden_logits(model, ids)
            else:
                logits = model_logits(model, ids)
            lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                 ids[:, 1:].reshape(-1))
            ti = rec["idx"].to(logits.device)
            tv = rec["val"].to(logits.device).float()
            s_sel = logits[:, :-1].gather(-1, ti).reshape(-1, ti.shape[-1])

            # the teacher's own support mass (cache 'w') is needed by the tail
            # terms and by the exact D_KL1 support weighting; an older cache has
            # no such key and the honest response is to refuse, not to substitute
            # a constant (the per-token spread is 4x at the p01 tail, which is
            # exactly where the gate's worst tokens live).
            w_t = None
            if args.kd_support_w == "wt" or args.kd_tail_weight > 0 or args.kd_tailcond_weight > 0:
                if "w" not in rec:
                    raise SystemExit(
                        f"--kd-support-w/--kd-tail-* need a cache with per-token "
                        f"teacher support mass, and {cache_path} has none. Rebuild it: "
                        f"moe/phase1-w1.sh cache c   (the 'w' field is free to "
                        f"record -- the cache stage already holds the full "
                        f"logits it topk'd)")
                w_t = rec["w"].to(logits.device).float().reshape(-1)

            if args.kd_filter_frac > 0 or args.kd_support_w == "wt":
                kd = kd_filtered(s_sel, tv.reshape(-1, tv.shape[-1]),
                                 args.temp, args.kd_filter_frac,
                                 token_weight=w_t if args.kd_support_w == "wt" else None)
            else:
                kd = F.kl_div(F.log_softmax(s_sel.float() / args.temp, dim=-1),
                              F.log_softmax(tv.reshape(-1, tv.shape[-1]) / args.temp, dim=-1),
                              log_target=True, reduction="batchmean") * (args.temp ** 2)
            loss = lm + args.kd_weight * kd
            tail, w_s, tcond = None, None, None
            if args.kd_tail_weight > 0 or args.kd_tailcond_weight > 0:
                # One full-vocab pass serves both tail terms: the marginal needs
                # the mass, the tail-conditional also needs the normaliser.
                w_s, lse_s = support_mass_lse(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    ti.reshape(-1, ti.shape[-1]))
            if args.kd_tail_weight > 0:
                # The marginal piece of the full-vocab KL that the top-k KD term
                # cannot see.
                tail = residual_mass_kl(w_s, w_t).mean()
                loss = loss + args.kd_tail_weight * tail
            if args.kd_tailcond_weight > 0:
                # TAD's D_KL2: the tail-conditional piece, estimated from the
                # sampled teacher-tail tokens (Sparse Logit Sampling).  An old
                # cache cannot supply it, so refuse rather than skip a term the
                # run's whole purpose is to measure.
                if "tidx" not in rec or "tlp" not in rec:
                    raise SystemExit(
                        f"--kd-tailcond-weight needs a cache with sampled tail "
                        f"tokens, and {cache_path} has none. Rebuild it: "
                        f"moe/phase1-w1.sh cache-tail   (64 teacher-tail "
                        f"samples per position, ~12% extra cache)")
                m = rec["tidx"].shape[-1]
                tcond = tail_conditional_piece(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    rec["tidx"].to(logits.device).reshape(-1, m),
                    rec["tlp"].to(logits.device).float().reshape(-1, m),
                    ti.reshape(-1, ti.shape[-1]), w_t,
                    student_mass=w_s, student_lse=lse_s).mean()
                loss = loss + args.kd_tailcond_weight * tcond
            if args.balance == "zloss":
                # OLMoE control arm: a differentiable scale penalty on the raw
                # router logits, added to the loss like any other term.
                zl, bal = balance_z_loss(model, args.balance_z_coeff)
                loss = loss + zl
            if head is not None:
                # t+2 drafter training: the head sees h_t and emb(x_{t+1}) and
                # predicts x_{t+2}, through the frozen norm + lm_head.  Target:
                # the corpus token (the v1 auxiliary form) or the main model's
                # own choice at the same position (self / self-soft) -- the
                # quantity acceptance actually scores.
                h_in, e_in, tgt = mtp_targets(h, ids)
                if args.mtp_only:
                    h_in = h_in.detach()   # frozen body: no backward into it
                mlogits = mtp_logits(model, head, h_in,
                                     model.get_input_embeddings()(e_in))
                if args.mtp_target == "corpus":
                    mtp = chunked_ce(mlogits, tgt)
                elif args.mtp_target == "self":
                    mtp = chunked_ce(mlogits, self_target_main(logits))
                else:  # self-soft: imitate the main model's own distribution
                    mtp = chunked_kl(mlogits, logits[:, 1:-1].detach(),
                                     args.mtp_self_temp)
                if args.mtp_only:
                    # Only the drafter trains; the main-model terms built into
                    # `loss` above are diagnostics in this mode (their params
                    # are frozen, so adding them would only pay a wasted
                    # backward pass).
                    loss = args.mtp_weight * mtp
                else:
                    loss = loss + args.mtp_weight * mtp
            if args.log_entropy:
                # The sharpening signature the Phase 1 KLD gate found: entropy
                # and peak top-1 mass, read straight off the training logits so a
                # --kd-weight sweep is self-diagnosing and does not need a KLD run
                # per checkpoint. Diagnostic only -- costs a no-grad pass.
                with torch.no_grad():
                    flat = logits[:, :-1].reshape(-1, logits.shape[-1]).float()
                    lp = F.log_softmax(flat, dim=-1)
                    ent = float(-(lp.exp() * lp).sum(-1).mean())
                    peak = float(lp.exp().max(-1).values.mean())
                    del lp, flat
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if args.balance in ("bias", "quantile", "cbqb", "cb"):
                # ALF-LB / K3: one update per optimizer step, from the loads the
                # forward just accumulated.  Returns the diagnostics before reset.
                # ``cb`` has no buffer update; it rides along for the diagnostics
                # (loadH + per-sequence load variance, its own axis).
                bal = balance_update(model, args.balance, args.balance_delta)
            step += 1
            if (args.lr_half_every and step >= args.lr_decay_start
                    and step % args.lr_half_every == 0):
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                print(f"step {step}: lr -> {opt.param_groups[0]['lr']:.3e}", flush=True)
            if step % args.log_every == 0:
                extra = ""
                if args.log_entropy:
                    extra = f" H {ent:.3f} peak {peak:.4f}"
                # w_s vs w_t is the tail term's own diagnostic, and it is the
                # quantity the sharpening signature shows up in first: the
                # trained arms put far more mass on the teacher's top-512 than
                # the teacher does.  Logged whenever the term is on, for free.
                if tail is not None:
                    extra += (f" mass s {float(w_s.mean()):.4f}"
                              f" t {float(w_t.mean()):.4f}")
                if tcond is not None:
                    extra += f" tcond {float(tcond):.4f}"
                if zl is not None:
                    extra += f" zl {zl.item():.2e}"
                if mtp is not None:
                    extra += f" mtp {mtp.item():.4f}"
                if bal is not None:
                    extra += f" loadH {bal['mean_load_entropy']:.3f}"
                    seqv = bal.get("mean_seq_load_var", float("nan"))
                    if math.isfinite(seqv):
                        extra += f" seqvar {seqv:.5f}"
                print(f"step {step} lm {lm.item():.4f} kd {kd.item():.4f} "
                      f"tail {0.0 if tail is None else tail.item():.4f} "
                      f"total {loss.item():.4f}{extra}", flush=True)
            if args.eval_every and step % args.eval_every == 0 and ev is not None:
                ppl, ag = quick_eval(model, ev, args, ref)
                print(f"  [eval] step {step} ppl {ppl:.2f} router_agree {ag:.4f}",
                      flush=True)
                if head is not None:
                    acc = mtp_acceptance(model, head, ev, args)
                    print(f"  [eval] step {step} mtp_accept {acc:.4f}", flush=True)
            if args.ckpt_every and step % args.ckpt_every == 0:
                save(model, args, step)
            if args.steps and step >= args.steps:
                break
        if args.steps and step >= args.steps:
            break
    save(model, args, step)
    stats = gate_stats(model)
    if stats:
        # Phase C diagnostic: did the gates learn anything, and where?  A gate
        # unchanged from init plus a flat distribution means the mechanism is a
        # no-op on this arm -- a real answer, recorded as such.
        OUT.mkdir(parents=True, exist_ok=True)
        gpath = OUT / f"qwen35-branch-gates-{getattr(args, 'tag', '') or 'run'}.json"
        gpath.write_text(json.dumps(stats, indent=2))
        for r in stats:
            print("gate " + json.dumps(r), flush=True)
        print(f"wrote {gpath}", flush=True)
    print("training done", flush=True)


def save(model, args, step):
    """Write the branch state dict.

    ``--tag`` is part of the filename because nothing else in it distinguishes
    two arms: a top-50 and a top-512 run at the same rank/quant/step all want
    ``qwen35-corr-r512-g128-step4096.pt`` and silently overwrite each other.
    """
    sd = {k: v for k, v in model.state_dict().items()
          if ".branch." in k or ".gate." in k or k.startswith("_mtp_head.")}
    tag = "" if args.branch_quant == "fp32" else f"-{args.branch_quant}"
    name = f"qwen35-corr-r{args.rank}{tag}-step{step}"
    if getattr(args, "tag", ""):
        name += f"-{args.tag}"
    p = OUT / f"{name}.pt"
    torch.save(sd, p)
    print(f"saved {p}", flush=True)


# -------------------------------------------------------------------- eval ---

@torch.no_grad()
def stage_eval(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = windows(tok, args.eval_windows, args.seq, 999, "wikitext")

    patch_experts(args.group)
    model, _ = load_full(args)
    build_student(model, args)
    if args.balance != "none":
        n_gates = patch_router_balance(model, args.balance,
                                       args.balance_cb_eta, args.balance_qb_damp)
        print(f"router balance: {args.balance} on {n_gates} gates", flush=True)
    if args.load:
        missing, unexpected = load_branch_state(model, args.load)
        print(f"loaded {args.load}: missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
    model.eval()

    total, ntok = 0.0, 0
    for i in range(len(data)):
        ids = data[i:i + 1].to(args.device)
        logits = model_logits(model, ids)
        total += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                 ids[:, 1:].reshape(-1), reduction="sum").item()
        ntok += ids[:, 1:].numel()
    ppl = math.exp(total / ntok)
    res = {"ppl": round(ppl, 4), "checkpoint": args.load, "rank": args.rank,
           "branch_quant": args.branch_quant}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qwen35-eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as a function so tests can check flags without running a stage."""
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "cache", "ref", "train", "eval"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--max-memory", default="", help="e.g. '0:39GiB,1:39GiB'")
    ap.add_argument("--layers", type=int, default=4, help="smoke prefix depth")
    ap.add_argument("--prefix-layers", type=int, default=0,
                    help="run the full stages on an N-layer prefix (local dry runs only)")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--quant", choices=["absmean", "lloyd", "catq"], default="lloyd",
                    help="per-group scale rule for the frozen banks (lloyd = deployable; "
                         "catq = learned LM+ST reconstruction)")
    ap.add_argument("--alloc-file", default="",
                    help="RCO allocation map {tensor: bits} for the expert banks "
                         "(2 = lloyd ternary, 4/6/8 = g128 integer); default off")
    ap.add_argument("--catq-steps", type=int, default=200,
                    help="CAT-Q reconstruction steps when --quant catq")
    ap.add_argument("--catq-lr", type=float, default=0.05)
    ap.add_argument("--catq-gamma", type=float, default=0.8)
    ap.add_argument("--catq-s0", type=float, default=30.0)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--branch-quant", choices=["fp32", "g128", "rank"], default="fp32")
    ap.add_argument("--branch-target", choices=["moe_out", "attn_out", "both"], default="both",
                    help="moe_out (block output), attn_out (ssm_out/o_proj), or both")
    ap.add_argument("--branch-gate", choices=["none", "rw"], default="none",
                    help="Phase C residual-stream gates: none = frozen v1 (plain "
                         "additive branch); rw = per-channel read gate on the "
                         "branch input + per-channel write gate on its output, "
                         "1+tanh(r) with r=0 (identity at init, so the arm starts "
                         "bit-identical to v1). Diagonal, folds into the factors "
                         "at export -- no serving support needed")
    ap.add_argument("--windows", type=int, default=4096)
    ap.add_argument("--corpus-chars", type=int, default=50_000_000)
    ap.add_argument("--corpus-file", default="",
                    help="AYOT trace JSONL mixed into the training windows")
    ap.add_argument("--agentic-frac", type=float, default=0.1,
                    help="fraction of rows drawn from --corpus-file (AYOT: 0.1)")
    ap.add_argument("--cache-file", default="")
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-logits", type=int, default=50)
    ap.add_argument("--tail-logits", type=int, default=0,
                    help="cache stage: sampled teacher-tail tokens per position "
                         "for the D_KL2 term (0 = off, the frozen v1 cache; 64 "
                         "is the validated size, ~12%% extra cache)")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-half-every", type=int, default=0)
    ap.add_argument("--lr-decay-start", type=int, default=0)
    ap.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor")
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--kd-weight", type=float, default=1.0)
    ap.add_argument("--kd-filter-frac", type=float, default=0.0,
                    help="drop this fraction of the largest per-token KD losses "
                         "(SignRoundV2 uses 0.001; 0 = off)")
    ap.add_argument("--kd-support-w", choices=["one", "wt"], default="one",
                    help="per-token weight on the support-conditional KD term: "
                         "'one' (v1) leaves the top-k KL unweighted, 'wt' scales "
                         "each token by the teacher's own support mass w_t, which "
                         "is TAD's exact coarsened D_KL1 (marginal + w_t*conditional). "
                         "Needs a cache built with the 'w' field")
    ap.add_argument("--kd-tail-weight", type=float, default=0.0,
                    help="weight on the residual-mass (marginal) KL: match the "
                         "student's mass on the cached top-k support to the "
                         "teacher's, which the top-k KD term renormalises away. "
                         "Needs a cache built with the 'w' field. 0 = off, and "
                         "v1 stays off: this is a candidate, not a shipped default")
    ap.add_argument("--kd-tailcond-weight", type=float, default=0.0,
                    help="weight on TAD's D_KL2, the (1-w_t)-weighted "
                         "tail-conditional KL: the third chain-rule piece of the "
                         "full-vocab loss, estimated from the cache's sampled "
                         "tail tokens (--tail-logits). 1.0 adds the exact "
                         "measured piece; 0 = off (frozen v1)")
    ap.add_argument("--balance", choices=["none", "bias", "quantile", "zloss",
                                          "cb", "cbqb"],
                    default="none",
                    help="router balancing (Phase B): none = frozen v1 (no "
                         "balancing term); bias = DeepSeek aux-loss-free bias; "
                         "quantile = K3 Quantile Balancing; zloss = OLMoE router "
                         "z-loss; cb = causal per-sequence mass bias (the "
                         "routing-sweep follow-up); cbqb = cb + quantile. Bias "
                         "arms save the per-expert bias in the checkpoint, so "
                         "eval must pass the same --balance (and the same "
                         "--balance-cb-eta / --balance-qb-damp)")
    ap.add_argument("--balance-delta", type=float, default=1e-3,
                    help="ALF-LB step size u (bias arm)")
    ap.add_argument("--balance-cb-eta", type=float, default=0.05,
                    help="CB nudge scale: a hot expert at twice the uniform "
                         "score-mass rate is pushed down by eta (cb / cbqb)")
    ap.add_argument("--balance-qb-damp", type=float, default=1.0,
                    help="damping on the quantile coordinate step (quantile / "
                         "cbqb); the trained arm's biases reached +-2, so the "
                         "full step overshoots at a 511-token batch")
    ap.add_argument("--balance-z-coeff", type=float, default=1e-3,
                    help="coefficient on the router z-loss (zloss arm)")
    ap.add_argument("--mtp-weight", type=float, default=0.0,
                    help="weight on the t+2 multi-token-prediction loss (Phase E "
                         "item 11): trains a small head that shares the frozen "
                         "norm+lm_head and logs greedy draft acceptance at eval. "
                         "0 = off (frozen v1)")
    ap.add_argument("--mtp-target", choices=("corpus", "self", "self-soft"),
                    default="corpus",
                    help="drafter target: 'corpus' = the true t+2 token (the v1 "
                         "auxiliary form), 'self' = the main model's own greedy "
                         "choice at that position, 'self-soft' = its distribution "
                         "at --mtp-self-temp (the acceptance-oriented objective; "
                         "only meaningful with --mtp-only)")
    ap.add_argument("--mtp-self-temp", type=float, default=2.0,
                    help="temperature for --mtp-target self-soft (default 2.0)")
    ap.add_argument("--mtp-only", action="store_true",
                    help="frozen-body drafter protocol: freeze every non-head "
                         "parameter and backprop only the drafter loss (the main "
                         "model cannot be perturbed)")
    ap.add_argument("--mtp-head-layers", type=int, choices=(1, 2), default=1,
                    help="drafter capacity: 1 = single linear (v1), 2 = one GELU "
                         "hidden layer (default 1)")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--ref-file", default="",
                    help="precomputed teacher router refs for the in-run eval "
                         "(build with the 'ref' stage); avoids teacher residency")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--load", default="")
    ap.add_argument("--tag", default="",
                    help="suffix for checkpoint filenames; set it whenever two arms "
                         "share rank/branch-quant/steps, or they overwrite each other")
    ap.add_argument("--log-entropy", action="store_true",
                    help="also log training entropy and peak top-1 mass each "
                         "--log-every steps. Diagnostic for the --kd-weight sweep: "
                         "these are the quantities the KLD gate showed collapsing. "
                         "Costs a no-grad forward over the vocab; off by default")
    ap.add_argument("--resume", default="",
                    help="resume training from a branch checkpoint (step number from the filename)")
    return ap


def main():
    args = build_parser().parse_args()
    if args.stage == "smoke":
        smoke(args)
    elif args.stage == "cache":
        stage_cache(args)
    elif args.stage == "ref":
        build_ref(args)
    elif args.stage == "train":
        stage_train(args)
    else:
        stage_eval(args)


if __name__ == "__main__":
    main()

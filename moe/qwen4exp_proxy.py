"""Qwen3.8-Flash-Next (``qwen4_exp``) ternary proxy: fused expert banks +
correction branches + KD, ported from the 35B (``qwen35_moe_proxy``) harness.

Architecture map (official ``Qwen/Qwen3.8-Flash-Next-FP8`` config, 2026-09-30):

  - 48 layers, ``layer_types`` alternating 3x ``linear_attention``
    (GatedDeltaNet) + 1x ``full_attention`` (QSA-indexed attention);
    the checkpoint's ``full_attention`` entries are normalised to
    ``qwen_sparse_attention`` by the config;
  - ``hidden`` 2560, ``hc_count`` 4 residual streams; every block is wrapped in
    a ``Qwen4ExpTextGatedResidual`` hyper-connection ("GatedResidual");
  - ``mlp`` = ``Qwen4ExpTextSparseMoeBlock`` (router + fused expert banks +
    shared expert), banks have the **same layout as 35B**:
    ``gate_up_proj [E, 2ff, h]``, ``down_proj [E, h, ff]``, fused per-expert
    checkpoint tensors merged with ``MergeModulelist`` + ``Concatenate`` in the
    official loader (``core_model_loading``);
  - ``ple`` = ``Qwen4ExpTextPLELayer`` on ``ple_layer_ids=[2]`` (1-based), i.e.
    decoder layer index 1; its n-gram table is 320,001,536 x 160 (~95 GiB
    bf16), stored fp8 with a single per-tensor ``weight_scale`` in 128 running
    checkpoint shards (``ngram_embedding.shard_N.weight``);
  - official FP8 format: fine-grained 128x128 blocks + ``weight_scale_inv``.

Correction placement (P0 item 3 -- decision, see ``attach_branches``):

  ``Qwen4ExpTextDecoderLayer.forward`` is **not** plain-additive: it keeps a
  4-stream hyper state ``[B, S, 4*h]``; ``attn_hyper_connection`` /
  ``mlp_hyper_connection`` mix it to a single stream, run the block, then
  inject the block output back into all four streams:
  ``hidden = hyper_input + (block_out.unsqueeze(-2) * injection_weights).flatten(-2)``.
  The correction branches therefore attach at the **block output, pre-injection**
  -- wrapping ``layer.mlp`` (the sparse MoE block) and the attention output
  projection (``linear_attn.out_proj`` / ``self_attn.o_proj``) -- exactly the
  35B recipe's touch points.  The correction then rides the model's own
  hyper-connection injection with the same weights as the expert output; the
  GatedResidual topology itself is untouched.  The alternative (post-injection
  on the 4*h hyper state) would double the branch dimensions, break the 35B
  recipe equivalence, and re-scale a surface the model already gates -- rejected
  and recorded in HANDOFF-FLASH-NEXT.md.

Runtime port (P0 item 1): ``patch_indexer`` fixes a scatter-dtype bug in the
pinned transformers build (``Qwen4ExpTextQSAIndexer.forward`` scatters int32
indices, which torch rejects; cast to int64).  ``patch_experts`` is a near-copy
of the 35B fused-bank STE patch.  The cache record schema is identical to the
35B's (``idx/val/w/router/tidx/tlp``), built by ``make_record`` so the KLD
instrument and ``kd_loss`` are reused untouched.

FP8 prefix loader (P0 item 4): ``load_fp8_prefix`` reads only the shards the
first N layers need, dequantizes the fine-grained fp8 blocks with
``weight_scale_inv``, merges the per-expert tensors into the fused banks, and
-- when a PLE layer is in range -- builds a row-compact ``SparseNGramTable``
from the sharded table (only the rows the run's windows hash to).  This is the
local-verification path; on the FP8-capable pod the official
``from_pretrained`` route keeps fp8 native (see ``preflight``).

PLE precision A/B (P0 item 5): ``stage_ple_ab`` holds the FP prefix fixed and
re-quantizes only the gathered PLE rows (int8/int4/int2, symmetric per-group),
measuring full-vocab KLD against the fp8-dequantized teacher.  The release
container's n-gram precision is otherwise unpriced.

Stages: ``smoke``, ``cache``, ``ref``, ``train``, ``eval``, ``ple-ab``,
``preflight``, ``compact-check``.

Compact-bank training mode (P2, gate G2; ``--compact-banks``, default off): the
frozen expert banks are held in their *deployed* form -- 2-bit ternary codes
plus an fp16 scale per 128-group -- and the training forward decodes only the
expert slices a token hits (``compact_banks`` / ``CompactBank`` /
``compact_experts_forward``).  This is what makes the cur05-mirror fit one
141 GB H200: the resident bf16 student is ~240 GiB of experts alone, while the
codes+scales are ~32 GiB.  The decode is bit-equal to the in-place
``ternary_lloyd`` banker (``_verify_bank``), so the trained student is exactly
what ships.  Branches and router stay dense with gradients.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import re
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moe_proxy import _lloyd_scale, ternary_absmean, ternary_lloyd  # noqa: E402
from olmoe_corrections import (CorrectionBranch, MoEWithCorrection,  # noqa: E402
                               gate_stats, load_branch_state, moe_block,
                               quantize_bank_inplace)
from olmoe_proxy import gate_hook, ternary_ste, windows  # noqa: E402
from kd_loss import (kd_filtered, residual_mass_kl, sample_tail_tokens,  # noqa: E402
                     support_mass, support_mass_lse, tail_conditional_piece)
from router_bias import (balance_diagnostics, balance_update,  # noqa: E402
                         balance_z_loss, _new_seq, _new_stats,
                         _seq_load_update)
from router_balance import causal_mass_bias, margins, margins_from_scores  # noqa: E402

ART = Path(os.environ.get("MOE_ARTIFACTS", HERE / "artifacts"))
# Local mirror of the official FP8 checkpoint (config + tokenizer + index +
# `shards/`).  Override with Q4_MODEL or --model-dir.
MODEL = Path(os.environ.get(
    "Q4_MODEL", "/home/penis/Desktop/work/models/qwen38-flashnext-fp8"))
OUT = ART / "qwen4exp"
CACHE = OUT / "teacher-cache.pt"
FP8_BLOCK = 128               # fine-grained weight block [128, 128]
PLE_SHARD_DIR = "shards"      # where downloader puts model-*-of-*.safetensors


# ----------------------------------------------------------- runtime patches --

def _pure_causal(attention_mask) -> bool:
    """True for the no-cache, batch-1, unpadded causal mask the cache uses."""
    if attention_mask is None or attention_mask.dtype != torch.bool:
        return False
    if attention_mask.dim() != 4 or attention_mask.shape[0] != 1 \
            or attention_mask.shape[1] != 1:
        return False
    s = attention_mask.shape[-1]
    if attention_mask.shape[-2] != s:
        return False
    causal = torch.ones(s, s, dtype=torch.bool, device=attention_mask.device).tril()
    return bool(torch.equal(attention_mask[0, 0], causal))


@torch.no_grad()
def _vectorized_indexer_forward(self, hidden_states, position_embeddings,
                                attention_mask, past_key_values):
    """Batched equivalent of the QSA indexer selection (pure-causal, batch 1).

    The reference implementation loops over query positions in Python; at seq
    512 that is ~6k kernel launches and it dominates the per-window time.  This
    keeps the same selection semantics:

      - block keys are pooled once for every complete block and RoPE'd at the
        block start;
      - scores are ``relu(q . pooled)`` summed over heads / sqrt(head_dim);
      - a block is visible to query q iff ``b < (q + 1) // R``; the top
        ``min(block_topk, n_visible)`` blocks by score are selected;
      - the tail (tokens from the last incomplete block through the query) is
        always visible.

    Only valid for the frozen forward (no cache, batch 1, unpadded causal mask);
    the dispatcher falls back to the reference otherwise.
    """
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M

    batch, seq, _ = hidden_states.shape
    hd = self.index_head_dim
    ratio = self.compress_ratio
    device = hidden_states.device
    full_cos, full_sin = position_embeddings

    n_full = seq // ratio
    if self.block_topk >= n_full:
        # The official config gives block_topk = budget // ratio = 512 blocks
        # while a 512-token window has 128: every complete block is selected,
        # so selected = all complete blocks + tail = the plain causal mask.
        # (This is the pod's actual regime; the general path below covers
        # smaller budgets, where equal scores make topk's tie choice
        # unspecified -- the reference itself is then arbitrary.)
        return torch.ones(seq, seq, dtype=torch.bool, device=device).tril()[None, None]

    qk = self.index_qk_proj(hidden_states)
    q, token_k = torch.split(
        qk, [self.index_n_heads * hd, self.index_kv_heads * hd], dim=-1)
    q = q.reshape(batch, seq, self.index_n_heads, hd)
    keys = token_k.reshape(batch, seq, self.index_kv_heads, hd).squeeze(2)
    q = self.q_layernorm(q)
    q = M.apply_rotary_pos_emb(q, cos=full_cos[:, -seq:, :],
                               sin=full_sin[:, -seq:, :], unsqueeze_dim=2)

    idx = torch.arange(seq, device=device)
    n_blocks = (idx + 1) // ratio                       # complete blocks per query
    out = ((idx[None, :] >= n_blocks[:, None] * ratio)
           & (idx[None, :] <= idx[:, None]))            # tail mask [S, S]
    if n_full > 0:
        kg = keys[0, :n_full * ratio].reshape(n_full, ratio, hd)
        pooled = self.k_layernorm(kg.float().mean(dim=1).to(keys.dtype))
        starts = torch.arange(0, n_full * ratio, ratio, device=device)
        pooled = M.apply_rotary_pos_emb(
            pooled.unsqueeze(1),
            cos=full_cos[0].index_select(0, starts),
            sin=full_sin[0].index_select(0, starts)).squeeze(1)
        scores = torch.einsum("qhd,bd->qbh", q[0].float(), pooled.float())
        scores = torch.relu(scores).sum(-1) / math.sqrt(hd)      # [S, B]
        blocked = (torch.arange(n_full, device=device)[None, :]
                   >= n_blocks[:, None])
        scores = scores.masked_fill(blocked, float("-inf"))
        k_sel = min(self.block_topk, n_full)
        # stable sort: equal scores keep increasing block order, matching the
        # reference's per-query topk over the visible candidate list (relu
        # zeroes make ties common; topk's tie order is not specified)
        sel = torch.sort(scores, dim=-1, descending=True,
                         stable=True).indices[:, :k_sel]             # [S, k]
        valid = ~blocked.gather(1, sel)                          # [S, k]
        blk = (sel.unsqueeze(-1) * ratio
               + torch.arange(ratio, device=device)).reshape(
            seq, k_sel * ratio)                              # [S, k*R]
        valid = valid.unsqueeze(-1).expand(-1, -1, ratio).reshape(
            seq, k_sel * ratio)
        rows = torch.arange(seq, device=device).unsqueeze(1).expand_as(blk)
        out = out.clone()
        out[rows[valid], blk[valid]] = True                  # scatter-only True
    return out[None, None]                       # [batch=1, head=1, S, S]


def patch_indexer(fast: bool = False):
    """Fix the QSA indexer's scatter dtype (pinned transformers build).

    ``Qwen4ExpTextQSAIndexer.forward`` builds ``selected_token_indices`` as
    int32 and scatters them into a bool mask; ``Tensor.scatter`` requires int64
    indices, so **every** full-attention forward raises
    ``Expected dtype int64 for index`` on the pinned commit ``f339035b``.
    The fix is one cast; the rest of the reference body is a verbatim copy of
    the upstream function (re-diff it when the runtime pin moves).

    ``fast=True`` additionally routes the pure-causal no-cache batch-1 case
    through ``_vectorized_indexer_forward``; the flag is read per call, so it
    can be toggled after patching.
    """
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextQSAIndexer)

    Qwen4ExpTextQSAIndexer._q4exp_fast = bool(fast)
    if getattr(Qwen4ExpTextQSAIndexer, "_q4exp_scatter_fix", False):
        return

    def _reference_forward(self, hidden_states, position_embeddings,
                           attention_mask, past_key_values):
        from transformers.models.qwen4_exp import modeling_qwen4_exp as M

        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
        # The cos/sin here are the full positions for the keys, so we need to
        # slice to get only the current positions for the queries
        full_cos, full_sin = position_embeddings
        current_cos, current_sin = (full_cos[:, -seq_length:, :],
                                    full_sin[:, -seq_length:, :])

        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk,
            [self.index_n_heads * self.index_head_dim,
             self.index_kv_heads * self.index_head_dim],
            dim=-1,
        )
        q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
        q = self.q_layernorm(q)
        q = M.apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)

        if past_key_values is not None:
            raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)

        # Note that the mask is never None here as we only allow eager and sdpa,
        # and we do not allow sdpa's mask skip.  It's always 4D with either
        # bool (sdpa) or float (eager) and already gives us the valid indices
        visible_token_indices = (attention_mask if attention_mask.dtype == torch.bool
                                 else attention_mask == 0)

        selected_token_indices = torch.full(
            (batch_size, seq_length, self.token_budget + self.compress_ratio - 1),
            -1,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        for batch_idx in range(batch_size):
            for query_idx in range(seq_length):
                local_visible_indices = torch.nonzero(
                    visible_token_indices[batch_idx, 0, query_idx], as_tuple=False
                ).flatten()
                num_complete_blocks = local_visible_indices.shape[-1] // self.compress_ratio
                # Compute selected tokens
                if num_complete_blocks > 0:
                    block_token_indices = local_visible_indices[
                        : num_complete_blocks * self.compress_ratio].view(
                        num_complete_blocks, self.compress_ratio)

                    key_groups = raw_keys[batch_idx].index_select(
                        0, block_token_indices.flatten())
                    key_groups = key_groups.view(*block_token_indices.shape,
                                                 self.index_head_dim)
                    pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
                    pooled_keys = self.k_layernorm(pooled_keys)
                    group_starts = block_token_indices[:, 0]
                    block_key_states = M.apply_rotary_pos_emb(
                        pooled_keys.unsqueeze(1),
                        cos=full_cos[batch_idx].index_select(0, group_starts),
                        sin=full_sin[batch_idx].index_select(0, group_starts),
                    ).squeeze(1)

                    scores = torch.matmul(
                        q[batch_idx, query_idx].float(),
                        block_key_states.float().transpose(-1, -2)
                    ).transpose(-1, -2)
                    scores = torch.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)

                    selected_block_indices = scores.topk(
                        min(self.block_topk, num_complete_blocks), dim=0).indices
                    # Remap the indices of the blocks to the indices of individual tokens
                    selected_tokens = block_token_indices.index_select(
                        0, selected_block_indices).flatten()
                else:
                    selected_tokens = torch.tensor([], device=hidden_states.device)
                tail = local_visible_indices[num_complete_blocks * self.compress_ratio:]
                selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
                selected_token_indices[batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens

        # Create the additive mask to be added to the main causal mask
        kv_length = attention_mask.shape[-1]
        selected_token_mask = torch.zeros(
            (*selected_token_indices.shape[:-1], kv_length + 1),
            device=attention_mask.device, dtype=torch.bool)
        # We absorb all the -1 by scattering them to the last index that we will drop
        scatter_indices = torch.where(selected_token_indices >= 0,
                                      selected_token_indices, kv_length).long()
        selected_token_mask = selected_token_mask.scatter(
            -1, scatter_indices, True)[..., :kv_length].unsqueeze(1)
        # if using eager, convert to float mask
        if attention_mask.is_floating_point():
            min_dtype = torch.finfo(attention_mask.dtype).min
            selected_token_mask = torch.where(selected_token_mask,
                                              attention_mask.new_zeros(()), min_dtype)
        return selected_token_mask

    def dispatch(self, hidden_states, position_embeddings, attention_mask,
                 past_key_values):
        if (getattr(Qwen4ExpTextQSAIndexer, "_q4exp_fast", False)
                and past_key_values is None and hidden_states.shape[0] == 1
                and _pure_causal(attention_mask)):
            return _vectorized_indexer_forward(
                self, hidden_states, position_embeddings, attention_mask,
                past_key_values)
        return _reference_forward(self, hidden_states, position_embeddings,
                                  attention_mask, past_key_values)

    Qwen4ExpTextQSAIndexer.forward = dispatch
    Qwen4ExpTextQSAIndexer._reference_forward = _reference_forward
    Qwen4ExpTextQSAIndexer._q4exp_scatter_fix = True


def patch_experts(group: int, model=None):
    """Make ``Qwen4ExpTextExperts`` ternarise its fused banks on the fly (STE).

    Near-copy of the 35B patch: the bank layout is identical
    (``gate_up_proj [E, 2ff, h]``, ``down_proj [E, h, ff]``) and the forward
    signature matches.  ``device_map="auto"`` binds the pre-patch forward onto
    each experts instance, shadowing the class patch; pass the loaded ``model``
    to rebind those instances too.
    """
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextExperts)

    def quantize_slice(w, group):
        """One-expert ternary quantization (no grad): the inference path.

        The fused banks are 5B parameters per layer, so quantizing the whole
        bank at once transiently needs ~10 GB (three fp32-equivalent copies) --
        it OOMs a 20 GB card next to the resident model.  Per-expert slices are
        ~7 MB each and the scale groups never cross expert boundaries, so the
        result is identical to the whole-bank quantizer.
        """
        with torch.no_grad():
            return ternary_ste(w, group)

    def ternary_forward(self, hidden_states, top_k_index, top_k_weights):
        training = torch.is_grad_enabled()
        if training:
            gu = ternary_ste(self.gate_up_proj, group)
            dn = ternary_ste(self.down_proj, group)
        dt = self.gate_up_proj.dtype
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
            if training:
                gu_e, dn_e = gu[expert_idx], dn[expert_idx]
            else:
                gu_e = quantize_slice(self.gate_up_proj[expert_idx], group)
                dn_e = quantize_slice(self.down_proj[expert_idx], group)
            gate, up = F.linear(current_state, gu_e).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = F.linear(h, dn_e)
            h = h * top_k_weights[token_idx, top_k_pos, None].to(dt)
            final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))
        return final_hidden_states

    if not getattr(Qwen4ExpTextExperts, "_q4exp_ternary_patch", False):
        Qwen4ExpTextExperts._q4exp_original_forward = Qwen4ExpTextExperts.forward

        def forward(self, hidden_states, top_k_index, top_k_weights):
            if not getattr(self, "_ternary", False):
                return Qwen4ExpTextExperts._q4exp_original_forward(
                    self, hidden_states, top_k_index, top_k_weights)
            return ternary_forward(self, hidden_states, top_k_index, top_k_weights)

        Qwen4ExpTextExperts.forward = forward
        Qwen4ExpTextExperts._q4exp_ternary_patch = True

    forward = Qwen4ExpTextExperts.forward

    if model is not None:
        for m in model.modules():
            if not isinstance(m, Qwen4ExpTextExperts):
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


def _balanced_forward_q4(self, hidden_states):
    """Stock ``Qwen4ExpTextTopKRouter.forward`` + ALF-LB selection bias.

    The 35B balancer patch (``router_bias``) is router-class specific; this is
    the qwen4_exp port.  It mirrors the stock qwen4_exp forward exactly when no
    bias is set (bit-identical at init), and otherwise adds the bias to the raw
    selection scores while the returned mixture weights stay the unbiased
    softmax over the selected experts -- the ALF-LB rule.  Returns
    ``(router_logits, router_scores, router_indices)``, the stock tuple.

    State (``balance_bias`` buffer, ``_balance_stats``/``_balance_seq``) is the
    same shape ``router_bias.balance_update``/``balance_diagnostics`` consume,
    so those updaters are reused unchanged.
    """
    hidden = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(hidden, self.weight)                     # raw scores
    kind = getattr(self, "_balance_kind", None)
    bias = getattr(self, "balance_bias", None)
    choice = logits
    if bias is not None and kind in ("bias", "quantile", "cbqb"):
        choice = choice + bias
    if kind in ("cb", "cbqb"):
        choice = choice + causal_mass_bias(
            logits, getattr(self, "_balance_cb_eta", 0.05))
    probs = F.softmax(choice, dtype=torch.float, dim=-1)
    _, idx = torch.topk(probs, self.top_k, dim=-1)
    # mixture weights: raw softmax over the selected experts (bias never enters)
    weights = F.softmax(logits.gather(-1, idx).float(), dim=-1).to(logits.dtype)

    if kind is not None and self.training:
        st = self._balance_stats
        st["counts"] = st["counts"] + torch.bincount(
            idx.reshape(-1).cpu(), minlength=self.num_experts)
        # one sequence per batch row (a flat input is one sequence)
        if hidden_states.dim() == 3:
            rows = idx.reshape(hidden_states.shape[0], -1)
        else:
            rows = idx.reshape(1, -1)
        for r in rows:
            _seq_load_update(self, r)
        if kind == "quantile":
            st["margins"].append(margins(logits, bias, self.top_k).detach())
        elif kind == "cbqb":
            st["margins"].append(margins_from_scores(choice, self.top_k).detach())
        elif kind == "zloss":
            z = torch.logsumexp(logits.float(), dim=-1)
            st["z_terms"].append((z ** 2).mean())
    return logits, weights, idx


def patch_router_balance(model, kind: str = "bias", cb_eta: float = 0.05,
                         qb_damp: float = 1.0) -> int:
    """Patch every qwen4_exp gate for ``kind``; returns the gate count.

    Bias arms persist ``balance_bias`` in the checkpoint (it is a ``.gate.``
    key, which ``save`` keeps), so eval must pass the same ``--balance``.
    """
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextTopKRouter)

    if kind not in ("bias", "quantile", "zloss", "cb", "cbqb"):
        raise ValueError(f"kind must be a balancing arm, got {kind!r}")
    n = 0
    for m in model.modules():
        if not isinstance(m, Qwen4ExpTextTopKRouter):
            continue
        if not getattr(m, "_balance_kind", None):
            if not hasattr(m, "balance_bias"):
                # buffer born on the gate's device (device_map runs first)
                m.register_buffer(
                    "balance_bias",
                    torch.zeros(m.num_experts, device=m.weight.device))
            m.forward = types.MethodType(_balanced_forward_q4, m)
            m._balance_stats = _new_stats(m.num_experts)
            m._balance_seq = _new_seq(m.num_experts)
        m._balance_kind = kind
        m._balance_cb_eta = cb_eta
        m._balance_qb_damp = qb_damp
        n += 1
    return n


# ------------------------------------------------------------ model helpers --

def text_layers(model):
    """Decoder-layer list for the text model / causal LM / conditional wrapper."""
    root = model
    if hasattr(root, "language_model"):
        return root.language_model.layers
    if hasattr(root, "layers"):
        return root.layers
    inner = getattr(root, "model", None)
    if inner is not None:
        if hasattr(inner, "language_model"):
            return inner.language_model.layers
        if hasattr(inner, "layers"):
            return inner.layers
    raise AttributeError(f"cannot find decoder layers on {type(model).__name__}")


def model_logits(model, ids):
    """Logits for the conditional wrapper (.logits) or the bare text prefix."""
    out = model(input_ids=ids, use_cache=False)
    logits = getattr(out, "logits", None)
    if logits is None:
        logits = model.lm_head(out.last_hidden_state)
    return logits


def _ce_chunk_sum(logits_chunk, targets_chunk):
    return F.cross_entropy(logits_chunk.float(), targets_chunk,
                           reduction="sum")


class offload_saved_tensors:
    """Context: autograd saves its tensors in host RAM, fetch on use.

    The L40S step needs ~3 GiB of saved activations that only exist to be read
    back in backward; the pod has 188 GiB of host RAM.  ``pack`` copies every
    saved tensor to CPU on save, ``unpack`` brings it back for the op that
    needs it.  Wrap **forward and backward**: the recomputation inside
    ``torch.utils.checkpoint`` saves tensors during ``backward()`` and those
    must be offloaded too.
    """

    def __init__(self, device):
        self.device = torch.device(device)

    def __enter__(self):
        dev = self.device

        def pack(t):
            return t.to("cpu", non_blocking=False)

        def unpack(t):
            return t.to(dev, non_blocking=False)

        self._hooks = torch.autograd.graph.saved_tensors_hooks(pack, unpack)
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc):
        return self._hooks.__exit__(*exc)


def chunked_cross_entropy(logits, targets, chunk: int = 64):
    """Cross-entropy without the full-vocab fp32 softmax buffers.

    The 248k vocab at seq 512 needs a 486 MiB fp32 save in the forward plus a
    486 MiB grad buffer in backward (P2 L40S: the 48 GB card was ~0.1 GiB short
    of the full-CE backward).  Each chunk is recomputed in backward, so only
    one chunk's softmax is live at a time.  Mathematically the same loss
    (fp32 reduction order differs in the last bits).
    """
    flat = logits.reshape(-1, logits.shape[-1])
    tgt = targets.reshape(-1)
    total = flat.new_zeros(())
    for i in range(0, tgt.numel(), chunk):
        total = total + torch.utils.checkpoint.checkpoint(
            _ce_chunk_sum, flat[i:i + chunk], tgt[i:i + chunk],
            use_reentrant=False)
    return total / tgt.numel()


def ternarize_banks(model, args, work_device=None) -> None:
    """Freeze the expert banks in place under the selected scale rule.

    ``work_device`` runs the (memory-hungry) quantizer on that device while the
    banks are still on their current one: the 20 GB local card cannot hold the
    13.5 GB prefix and the Lloyd temporaries at once, and under system-memory
    pressure the ROCm allocator was observed to return non-finite chunks.
    Quantizing the bank alone on the GPU (a few GB) and keeping the result
    removes both the peak and the corruption window.

    Each chunk is verified; a non-finite chunk is redone on the CPU (identical
    math) so a bad GPU chunk can never silently poison a checkpoint.
    """
    redone = 0
    for layer in text_layers(model):
        for proj in (layer.mlp.experts.gate_up_proj,
                     layer.mlp.experts.down_proj):
            if work_device is not None and proj.device != torch.device(work_device):
                pg = proj.detach().to(work_device)
                redone += quantize_bank_chunked(pg, args.group, args.quant)
                proj.data = pg.to(proj.device)
            else:
                redone += quantize_bank_chunked(proj, args.group, args.quant)
            layer.mlp.experts._ternary = False      # banks already quantised
    if redone:
        print(f"WARN: {redone} ternary chunk(s) were non-finite on GPU and "
              f"were redone on CPU", flush=True)


@torch.no_grad()
def quantize_bank_chunked(p: torch.Tensor, group: int = 128,
                          kind: str = "lloyd", chunk: int = 8,
                          cpu_retry: bool = True) -> int:
    """In-place ternary quantization, chunked, with a finiteness guard.

    Returns the number of chunks that had to be redone on the CPU.  The guard
    exists because a corrupted (non-finite) GPU chunk is otherwise silent and
    surfaces thousands of tokens later as a NaN KLD.
    """
    fn = ternary_lloyd if kind == "lloyd" else ternary_absmean
    redone = 0
    for s in range(0, p.shape[0], chunk):
        part = p[s:s + chunk]
        q = fn(part, group)
        if not bool(torch.isfinite(q).all()) and part.numel() > 0:
            if not cpu_retry:
                raise RuntimeError(
                    f"ternary quantization produced non-finite values on "
                    f"{part.device} at rows {s}:{s + chunk}")
            q = fn(part.detach().to("cpu"), group).to(part.device)
            if not bool(torch.isfinite(q).all()):
                raise RuntimeError(
                    "ternary quantization is non-finite on CPU too -- the "
                    "bank itself is corrupt")
            redone += 1
        part.copy_(q)
        del q
        if part.device.type == "cuda":
            torch.cuda.empty_cache()
    return redone


# ------------------------------------------------------- compact banks (P2) --

def pack_ternary_codes(q: torch.Tensor) -> torch.Tensor:
    """Pack ternary codes in ``{-1, 0, +1}`` (last dim % 4 == 0) into 2-bit bytes.

    Four values per byte, little-endian within the byte (codes ``4j+i`` at bits
    ``2*i`` of byte ``j``).  ``+1`` is stored as ``0b10`` so a byte is only ever
    built from the low two bits of each field.
    """
    if q.shape[-1] % 4:
        raise ValueError("the last dim must be a multiple of 4 to pack 2-bit codes")
    s = (q + 1).to(torch.int16).reshape(*q.shape[:-1], q.shape[-1] // 4, 4)
    out = s[..., 0] | (s[..., 1] << 2) | (s[..., 2] << 4) | (s[..., 3] << 6)
    return out.to(torch.uint8)


def unpack_ternary_codes(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_ternary_codes`; returns int8 values in {-1,0,+1}."""
    b = packed.to(torch.int16)
    parts = [(b >> (2 * i)) & 0b11 for i in range(4)]
    q = torch.stack(parts, dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 4)
    return q.to(torch.int8) - 1


def _encode_slice(wf: torch.Tensor, group: int, kind: str):
    """Encode one float32 slice into packed codes + the fp16 per-group scale.

    Reproduces ``ternary_lloyd`` / ``ternary_absmean`` exactly for ``kind`` on
    the *dequantized* weight: the scale is round-tripped through fp16 (the
    deployed container) and the codes are the same ``round(clamp(g/a))``.
    """
    shape = wf.shape
    gv = wf.reshape(*shape[:-1], shape[-1] // group, group)
    if kind == "absmean":
        a = gv.abs().mean(-1, keepdim=True).clamp_min(1e-8)
    else:
        a = _lloyd_scale(gv, gv.abs().mean(-1)).unsqueeze(-1)
    a16 = a.half()
    q = torch.clamp(torch.round(gv / a16.float().clamp_min(1e-12)), -1, 1)
    # pack within each 128-group so the byte layout matches the buffer
    # (..., ng, group // 4)
    return pack_ternary_codes(q), a16.squeeze(-1)


def decode_ternary(codes: torch.Tensor, scales: torch.Tensor,
                   dtype: torch.dtype | None = None) -> torch.Tensor:
    """Decode packed 2-bit codes + fp16 group scales back to a dense tensor.

    ``codes`` is ``[..., ng, group // 4]`` and ``scales`` ``[..., ng]``; the
    result is ``[..., ng * group]`` in ``dtype`` (or float32).  The value is
    bit-identical to ``(q * a_half).to(dtype)`` where ``a_half`` is the stored
    fp16 scale and ``q`` the ternary code.
    """
    q = unpack_ternary_codes(codes).float()
    deq = q * scales.float().unsqueeze(-1)
    deq = deq.reshape(*deq.shape[:-2], -1)
    return deq if dtype is None else deq.to(dtype)


def _compute_dtype(dt: torch.dtype) -> torch.dtype:
    """The dtype the experts compute in: an fp8 master decodes to bf16.

    On the pod's native fp8 route the bank is ``float8_e4m3fn``; decoding the
    ternary slice back into fp8 would quantise the deployed weights a second
    time, so the compute dtype is bf16 there.
    """
    return torch.bfloat16 if str(dt).startswith("torch.float8") else dt


class CompactBank(nn.Module):
    """A frozen expert bank in its *deployed* form: 2-bit codes + fp16 scales.

    The two buffers are the shipped container (``codes`` = 2 bits/param,
    ``scales`` = 2 bytes per 128-group).  ``decode_expert`` materialises a
    single expert's weight to run ``F.linear`` in the training forward, so the
    full bf16/fp8 bank never has to be resident.
    """

    def __init__(self, codes: torch.Tensor, scales: torch.Tensor, group: int,
                 out_dtype: torch.dtype):
        super().__init__()
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        self.group = int(group)
        self.out_dtype = out_dtype

    @classmethod
    def from_tensor(cls, w: torch.Tensor, group: int = 128, kind: str = "lloyd",
                    scale_inv: torch.Tensor | None = None, block: int = FP8_BLOCK,
                    chunk: int = 8) -> "CompactBank":
        """Encode an ``[E, rows, cols]`` bank (fp8 with ``scale_inv`` or dense).

        Chunked over the expert dim so the encode transient stays small next to
        the resident model.
        """
        w = w.detach()
        if w.dim() != 3:
            raise ValueError(f"expected an [E, rows, cols] bank, got {tuple(w.shape)}")
        last = w.shape[-1]
        g = group if (group > 0 and last % group == 0) else last
        ng = last // g
        codes = torch.empty(w.shape[0], w.shape[1], ng, g // 4,
                            dtype=torch.uint8, device=w.device)
        scales = torch.empty(w.shape[0], w.shape[1], ng, dtype=torch.float16,
                             device=w.device)
        for s in range(0, w.shape[0], chunk):
            part = w[s:s + chunk]
            if scale_inv is not None:
                si = scale_inv[s:s + chunk]
                part = torch.stack([dequant_fp8_block(part[i], si[i], block)
                                    for i in range(part.shape[0])])
            else:
                part = part.float()
            c, a = _encode_slice(part, g, kind)
            codes[s:s + chunk] = c
            scales[s:s + chunk] = a
            del part
        return cls(codes, scales, g, _compute_dtype(w.dtype))

    def decode_expert(self, e: int, dtype: torch.dtype | None = None) -> torch.Tensor:
        return decode_ternary(self.codes[e], self.scales[e], dtype or self.out_dtype)

    def decode_experts(self, idx_list, dtype: torch.dtype | None = None,
                       chunk: int = 8) -> torch.Tensor:
        """Batched :meth:`decode_expert` for a list/tensor of expert indices.

        Decodes ``chunk`` experts at a time so the fp32 decode transient stays
        ~chunk x 13 MB while the per-call overhead is amortised (P2 L40S: the
        per-expert decode loop was host-op bound).
        """
        idx = torch.as_tensor(idx_list, device=self.codes.device)
        n = int(idx.numel())
        out = torch.empty((n,) + tuple(self.codes.shape[1:-2])
                          + (self.codes.shape[-2] * self.group,),
                          dtype=dtype or self.out_dtype,
                          device=self.codes.device)
        for i in range(0, n, max(1, int(chunk))):
            out[i:i + chunk] = decode_ternary(
                self.codes[idx[i:i + chunk]], self.scales[idx[i:i + chunk]],
                dtype or self.out_dtype)
        return out

    def decode_all(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return decode_ternary(self.codes, self.scales, dtype or self.out_dtype)

    def nbytes(self) -> int:
        return (self.codes.numel() * self.codes.element_size()
                + self.scales.numel() * self.scales.element_size())

    def param_numel(self) -> int:
        """The dense parameter count the bank represents (codes are 2 bits)."""
        return self.codes.numel() * 4


def _compact_expert_mlp(experts, current_state, expert_idx):
    """One expert's decode + gate/up/down; checkpointed so the decoded weight is
    recomputed in backward instead of saved."""
    dt = experts._compact_gu.out_dtype
    gu_e = experts._compact_gu.decode_expert(int(expert_idx), dt)
    dn_e = experts._compact_dn.decode_expert(int(expert_idx), dt)
    gate, up = F.linear(current_state, gu_e).chunk(2, dim=-1)
    h = experts.act_fn(gate) * up
    return F.linear(h, dn_e)


def _compact_experts_group(experts, hidden_states, hit, top_k_weights,
                           expert_mask):
    """One checkpointed group of hit experts -> its partial block output.

    Batched: the group's weights are decoded with one ``decode_experts`` call,
    the group's (expert, token) pairs are padded into a ``[G, maxc, H]`` tensor
    and the two expert projections run as two ``bmm``s.  This is ~75x fewer
    host ops than the per-expert loop (P2 L40S: the per-expert loop was
    host-op bound at >27 s/step) while the decoded weights stay bounded at
    ~group x 9.8 MB (the reason for group checkpointing at all).
    ``hit`` is a Python list, so no graph is kept for it.
    """
    dt = experts._compact_gu.out_dtype
    dev = hidden_states.device
    g = len(hit)
    out = torch.zeros_like(hidden_states)
    with torch.no_grad():
        sub = expert_mask[torch.as_tensor(hit, device=dev)]      # [G, K, T]
        g_idx, k_idx, t_idx = sub.nonzero(as_tuple=True)         # positions
    if g_idx.numel() == 0:
        return out
    counts = torch.bincount(g_idx, minlength=g)
    maxc = int(counts.max().item())
    gu_w = experts._compact_gu.decode_experts(hit, dt)           # [G, 2F, H]
    dn_w = experts._compact_dn.decode_experts(hit, dt)           # [G, H, F]
    x = hidden_states[t_idx].to(dt)                              # [N, H]
    x_pad = x.new_zeros(g, maxc, x.shape[-1])
    w_pad = x.new_zeros(g, maxc)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(g_idx.numel(), device=dev) - starts[g_idx]
    x_pad[g_idx, rank] = x
    w_pad[g_idx, rank] = top_k_weights[t_idx, k_idx].to(dt)
    gu_out = torch.bmm(x_pad, gu_w.transpose(1, 2))              # [G, maxc, 2F]
    gate, up = gu_out.chunk(2, dim=-1)
    h = experts.act_fn(gate) * up
    y = torch.bmm(h, dn_w.transpose(1, 2))                       # [G, maxc, H]
    y = y * w_pad.unsqueeze(-1)
    out.index_add_(0, t_idx, y[g_idx, rank].to(out.dtype))
    return out


def _compact_experts_block_grouped(experts, hidden_states, top_k_index,
                                   top_k_weights, group_size: int = 32):
    """The sparse block as a chain of checkpointed expert groups.

    Memory is bounded by ``group_size`` decoded experts instead of the whole
    hit set; the group partials are summed in float32 and cast back at the end
    (sequential bf16 accumulation is not bit-reproducible across groupings
    anyway).
    """
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index,
                                                  num_classes=experts.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        hit = [int(e[0]) for e in
               torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
               if int(e[0]) != experts.num_experts]
    out = None
    for i in range(0, len(hit), group_size):
        part = torch.utils.checkpoint.checkpoint(
            _compact_experts_group, experts, hidden_states, hit[i:i + group_size],
            top_k_weights, expert_mask, use_reentrant=False)
        out = part.float() if out is None else out + part.float()
    if out is None:
        return torch.zeros_like(hidden_states)
    return out.to(hidden_states.dtype)


def _compact_experts_block(experts, hidden_states, top_k_index, top_k_weights):
    """The whole sparse-block expert computation for one training forward.

    Used with a single checkpoint per block: the decoded expert weights are
    recomputed in backward instead of saved, and there is one checkpoint node
    per layer instead of one per expert (the per-expert form is launch-bound).
    """
    dt = experts._compact_gu.out_dtype
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index,
                                                  num_classes=experts.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == experts.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx].to(dt)
        gu_e = experts._compact_gu.decode_expert(int(expert_idx), dt)
        dn_e = experts._compact_dn.decode_expert(int(expert_idx), dt)
        gate, up = F.linear(current_state, gu_e).chunk(2, dim=-1)
        h = experts.act_fn(gate) * up
        h = F.linear(h, dn_e)
        h = h * top_k_weights[token_idx, top_k_pos, None].to(dt)
        final_hidden_states.index_add_(0, token_idx,
                                       h.to(final_hidden_states.dtype))
    return final_hidden_states


def compact_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Training forward over the compact banks: decode the hit experts only.

    Mirrors the STE training forward (same loop, dtypes, weighting) but reads
    the deployed codes+scales instead of a resident bank.  With
    ``_grad_checkpoint`` the decode is recomputed in backward: the banks are
    frozen, but autograd otherwise saves every decoded expert weight for
    ``grad_input`` (48 layers x up to 512 hit experts ~ 400 GiB).
    """
    if getattr(self, "_grad_checkpoint", False):
        mode = getattr(self, "_ckpt_mode", "block")
        if mode == "expert":
            return _compact_experts_loop(self, hidden_states, top_k_index,
                                         top_k_weights, checkpoint="expert")
        if mode == "group":
            return _compact_experts_block_grouped(
                self, hidden_states, top_k_index, top_k_weights,
                group_size=getattr(self, "_ckpt_group", 32))
        return torch.utils.checkpoint.checkpoint(
            _compact_experts_block, self, hidden_states, top_k_index,
            top_k_weights, use_reentrant=False)
    return _compact_experts_loop(self, hidden_states, top_k_index,
                                 top_k_weights, checkpoint=None)


def _compact_experts_loop(self, hidden_states, top_k_index, top_k_weights,
                          checkpoint=None):
    gu_bank, dn_bank = self._compact_gu, self._compact_dn
    dt = gu_bank.out_dtype
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
        if checkpoint == "expert":
            h = torch.utils.checkpoint.checkpoint(
                _compact_expert_mlp, self, current_state, expert_idx,
                use_reentrant=False)
        else:
            gu_e = gu_bank.decode_expert(int(expert_idx), dt)
            dn_e = dn_bank.decode_expert(int(expert_idx), dt)
            gate, up = F.linear(current_state, gu_e).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = F.linear(h, dn_e)
        h = h * top_k_weights[token_idx, top_k_pos, None].to(dt)
        final_hidden_states.index_add_(0, token_idx,
                                       h.to(final_hidden_states.dtype))
    return final_hidden_states


def _install_compact_dispatch(m) -> None:
    """Class-level fallback: ``type(m).forward`` dispatches on ``_compact``.

    An instance ``forward`` binding can be overwritten (accelerate restores it
    when hooks are removed); the class-level check survives that.
    """
    cls = type(m)
    if getattr(cls, "_q4exp_compact_patch", False):
        return
    orig = cls.forward

    def forward(self, *a, **k):
        if getattr(self, "_compact", False):
            return compact_experts_forward(self, *a, **k)
        return orig(self, *a, **k)

    cls.forward = forward
    cls._q4exp_compact_patch = True


def compact_banks(model, args, work_device=None, verify: bool = False) -> int:
    """Replace every frozen expert bank with its deployed compact form.

    Two passes: (1) park every bank on host, which frees the ~112 GiB the dense
    fp8 banks hold on the card; (2) encode on the GPU (fp8 -> float is a scalar
    CPU fallback and would take hours on host) and keep the compact bank on the
    bank's original device.  The caller re-runs ``--force-gpu`` afterwards to
    pull the still-offloaded blocks (with their compact banks) onto the card.
    Returns the number of banks encoded.
    """
    layers = text_layers(model)
    dev_arg = getattr(args, "device", None)
    use_gpu = (torch.cuda.is_available() and bool(dev_arg)
               and str(dev_arg).startswith("cuda"))
    enc_dev = dev_arg if use_gpu else None

    parked: dict = {}
    for layer in layers:
        m = moe_block(layer).experts
        for name in ("gate_up_proj", "down_proj"):
            w = getattr(m, name)
            si = getattr(m, name + "_scale_inv", None)
            parked[(id(m), name)] = w.device
            if w.device.type != "cpu":
                setattr(m, name,
                        nn.Parameter(w.detach().to("cpu"), requires_grad=False))
            if si is not None:
                # ``*_scale_inv`` is an nn.Parameter on the native fp8 route;
                # parking it needs a Parameter, not a bare tensor (P2 finding).
                setattr(m, name + "_scale_inv",
                        nn.Parameter(si.detach().to("cpu"), requires_grad=False))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    banks = 0
    for li, layer in enumerate(layers):
        m = moe_block(layer).experts
        for name in ("gate_up_proj", "down_proj"):
            orig_dev = parked[(id(m), name)]
            w = getattr(m, name)
            si = getattr(m, name + "_scale_inv", None)
            w_e = w.to(enc_dev) if enc_dev is not None else w
            si_e = si.to(enc_dev) if (enc_dev is not None and si is not None) else si
            bank = CompactBank.from_tensor(w_e, args.group, args.quant, si_e)
            if verify:
                _verify_bank(bank, w_e, si_e, args.group, args.quant)
            setattr(m, "_compact_gu" if name == "gate_up_proj" else "_compact_dn",
                    bank.to(orig_dev))
            setattr(m, name,
                    nn.Parameter(torch.empty(0, device=orig_dev), requires_grad=False))
            if si is not None:
                delattr(m, name + "_scale_inv")
            banks += 1
            del w, si, w_e, si_e
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        m._compact = True
        _install_compact_dispatch(m)
        m.forward = types.MethodType(compact_experts_forward, m)
        if li == 0 or (li + 1) % 4 == 0:
            print(f"compact banks: {li + 1}/{len(layers)} layers encoded",
                  flush=True)
    return banks


def _reference_bank_expert(w_e: torch.Tensor, si_e, group: int, kind: str,
                           dtype: torch.dtype | None = None):
    """The ternary banker for one expert (the bit-equality target)."""
    wf = dequant_fp8_block(w_e, si_e) if si_e is not None else w_e.float()
    fn = ternary_absmean if kind == "absmean" else ternary_lloyd
    return fn(wf, group).to(dtype if dtype is not None else w_e.dtype)


def _verify_bank(bank: CompactBank, w, si, group: int, kind: str) -> None:
    """Per-expert bit-equality of the decoded compact bank vs the current path."""
    out_dtype = bank.out_dtype
    for e in range(w.shape[0]):
        ref = _reference_bank_expert(w[e], None if si is None else si[e],
                                     group, kind, out_dtype)
        got = bank.decode_expert(e)
        if not torch.equal(ref, got):
            diff = float((ref.float() - got.float()).abs().max())
            raise AssertionError(
                f"compact decode differs from the ternary banker on expert {e} "
                f"(max |diff| {diff})")


def expert_bank_bytes(model) -> dict:
    """Byte accounting for the compact expert banks and the rest of the model."""
    codes_b = scales_b = 0
    dense_numel = 0
    for layer in text_layers(model):
        m = moe_block(layer).experts
        for attr in ("_compact_gu", "_compact_dn"):
            bank = getattr(m, attr, None)
            if bank is None:
                continue
            codes_b += bank.codes.numel() * bank.codes.element_size()
            scales_b += bank.scales.numel() * bank.scales.element_size()
            dense_numel += bank.param_numel()
    branch_b = sum(p.numel() * p.element_size() for n, p in model.named_parameters()
                   if ".branch." in n and not p.is_meta)
    router_b = sum(p.numel() * p.element_size() for n, p in model.named_parameters()
                   if ".gate." in n and not p.is_meta)
    other_b = sum(p.numel() * p.element_size() for n, p in model.named_parameters()
                  if ".branch." not in n and ".gate." not in n and not p.is_meta)
    return {"expert_codes": codes_b, "expert_scales": scales_b,
            "expert_dense_numel": dense_numel,
            "expert_bf16_equivalent": dense_numel * 2,
            "branches": branch_b, "routers": router_b, "other_params": other_b}


def memory_report(model, args, label: str = "") -> dict:
    """Print the compact-bank accounting (G2(d)); returns the dict."""
    rep = expert_bank_bytes(model)
    total_b = sum(v for k, v in rep.items() if k != "expert_dense_numel")
    print(f"memory[{label or 'compact'}] expert codes "
          f"{rep['expert_codes']/2**30:.2f} GiB + scales "
          f"{rep['expert_scales']/2**30:.2f} GiB (dense bf16 would be "
          f"{rep['expert_bf16_equivalent']/2**30:.1f} GiB), branches "
          f"{rep['branches']/2**30:.3f} GiB, routers {rep['routers']/2**30:.3f} GiB, "
          f"other params {rep['other_params']/2**30:.2f} GiB -> resident "
          f"~{total_b/2**30:.2f} GiB", flush=True)
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
        peak = torch.cuda.max_memory_allocated() / 2**30
        rep.update({"cuda_allocated_gib": alloc, "cuda_reserved_gib": reserved,
                    "cuda_peak_gib": peak})
        print(f"memory[{label or 'compact'}] cuda allocated {alloc:.2f} GiB "
              f"reserved {reserved:.2f} GiB peak {peak:.2f} GiB", flush=True)
    return rep


def rebind_compact_forwards(model) -> int:
    """Re-bind ``compact_experts_forward`` on every compacted experts module.

    ``accelerate.hooks.remove_hook_from_module`` restores each module's
    ``forward`` to the value captured when the offload hook was attached, which
    shadows the compact binding (P2 finding: the class fp8 forward then runs and
    asks for the deleted ``*_scale_inv``).  Re-bind after removing hooks.
    """
    n = 0
    for layer in text_layers(model):
        m = moe_block(layer).experts
        if getattr(m, "_compact", False):
            m.forward = types.MethodType(compact_experts_forward, m)
            n += 1
    return n


def ple_non_table_tensors(ple):
    """The PLE layer's own tensors: everything except the ngram table.

    The table (47.7 GiB) is the only tensor allowed to stay host-side: the
    upstream PLE forward moves the gathered ids to the table's device and the
    rows back.  Everything else (head vocab sizes/offsets, layer multipliers,
    projections, conv) must sit with the activations once the accelerate hooks
    are gone (P2 L40S finding: cuda ``mixed_ids`` vs cpu
    ``ngram_heads_vocab_sizes`` -> RuntimeError in the remainder at
    ``modeling_qwen4_exp.py:1170``).
    """
    emb = getattr(getattr(ple, "ple_embedding", None), "ngram_embedding", None)
    keep = set()
    if emb is not None:
        keep.update(id(t) for t in emb.parameters())
        keep.update(id(t) for t in emb.buffers())
    return [t for t in list(ple.parameters()) + list(ple.buffers())
            if id(t) not in keep]


def place_ple_non_table(model, device: str) -> int:
    """Move every PLE tensor except the ngram table to ``device``.

    Returns the number of tensors moved.
    """
    dev = torch.device(device)
    moved = 0
    for layer in text_layers(model):
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        for t in ple_non_table_tensors(ple):
            if t.is_meta or t.device == dev:
                continue
            try:
                t.data = t.data.to(dev)
                moved += 1
            except (RuntimeError, NotImplementedError):
                pass
    return moved


def offload_unused_vision(model) -> int:
    """Move the multimodal vision tower to CPU: text-only stages never use it.

    ``AutoModelForImageTextToText`` loads ``model.visual`` (0.84 GiB / 333
    tensors) and ``device_map="auto"`` places it on the GPU; the text forward
    never calls it.  On a 48 GB card the training step is ~3 GiB short, so
    unused weights are pure loss (P2 L40S fit).
    """
    n = 0
    with torch.no_grad():
        for name, t in (list(model.named_parameters())
                        + list(model.named_buffers())):
            if "visual" not in name or t.is_meta or t.device.type == "cpu":
                continue
            try:
                t.data = t.data.to("cpu")
                n += 1
            except (RuntimeError, NotImplementedError):
                pass
    return n


class CpuEmbedding(nn.Module):
    """``nn.Embedding`` whose weight stays on the host (rows move per call).

    The 248k x 2560 token table is 1.18 GiB bf16 on the card and is fetched
    exactly once per step (512 rows, ~2.6 MB HtoD); the L40S fit needs the GiB
    back (P2: the backward OOMed 2 MiB short inside the expert-decode
    recompute).  The weight is frozen (only branches/routers train), so the
    lookup needs no autograd.
    """

    def __init__(self, emb: nn.Embedding):
        super().__init__()
        self.weight = nn.Parameter(emb.weight.data.detach().to("cpu").clone(),
                                   requires_grad=False)
        self.padding_idx = getattr(emb, "padding_idx", None)

    @property
    def num_embeddings(self) -> int:
        return self.weight.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.weight.shape[1]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        dev = ids.device
        with torch.no_grad():
            out = F.embedding(ids.to("cpu"), self.weight)
        return out.to(dev, non_blocking=True)


def offload_embed_tokens(model) -> str | None:
    """Replace ``embed_tokens`` with :class:`CpuEmbedding` (1.18 GiB freed).

    Skipped when the LM head shares the embedding (tied weights).  Returns the
    module key that was replaced, or None.
    """
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "tie_word_embeddings", False):
        return None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding) and name.endswith("embed_tokens"):
            parent_name, _, leaf = name.rpartition(".")
            parent = (model.get_submodule(parent_name) if parent_name else model)
            setattr(parent, leaf, CpuEmbedding(mod))
            return name
    return None


def harden_placement(model, device: str = "cuda:0") -> int:
    """Drop every accelerate hook and move every non-PLE weight to ``device``.

    After the banks are compacted the student is ~40 GiB, but the hooks left by
    ``device_map="auto"`` re-materialise offloaded weights on each forward and
    the process OOMs (P2 finding: 41 GiB resident, then 138 GiB inside the
    forward).  The PLE table stays host-side (48 GiB and lazy/meta).  Returns
    the number of tensors moved.
    """
    try:
        from accelerate import hooks as ah
        ah.remove_hook_from_module(model, recurse=True)
    except ImportError:
        pass
    except Exception:
        pass
    dev = torch.device(device)
    moved = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if (p.is_meta or p.device.type == "cuda"
                    or "ple" in name or "visual" in name
                    or "embed_tokens" in name):
                continue
            try:
                p.data = p.data.to(dev)
                moved += 1
            except (RuntimeError, NotImplementedError):
                pass
        for name, b in model.named_buffers():
            if (b.is_meta or b.device.type == "cuda"
                    or "ple" in name or "visual" in name
                    or "embed_tokens" in name):
                continue
            try:
                b.data = b.data.to(dev)
                moved += 1
            except (RuntimeError, NotImplementedError):
                pass
    moved += offload_unused_vision(model)
    moved += place_ple_non_table(model, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return moved


def enable_grad_checkpointing(model, mode: str = "block") -> int:
    """Turn on the decode checkpoint on every compact experts module.

    Autograd saves every decoded expert weight for ``grad_input``; with 48
    layers x up to 512 hit experts that is ~400 GiB of saved constants (P2
    finding: a 41 GiB student OOMed at 138 GiB in the first forward).  ``mode``
    is ``block`` (one checkpoint per sparse block -- fewer nodes) or ``expert``.
    Returns the number of experts modules patched.
    """
    n = 0
    for layer in text_layers(model):
        m = moe_block(layer).experts
        if getattr(m, "_compact", False):
            m._grad_checkpoint = True
            m._ckpt_mode = mode
            n += 1
    return n


def enable_layer_checkpointing(model) -> bool:
    """Checkpoint the decoder layers themselves (recompute in backward).

    P2 L40S fit: the 48 layers' saved activations (~2.6 GiB) do not fit next
    to the 41 GiB compact student on a 48 GB card -- the first forward OOMed
    at 43.8 GiB.  This is exact (recompute) and costs one extra forward per
    step; the compact experts' decode checkpoint nests inside it.
    """
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "use_cache", None):
        cfg.use_cache = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    return True


def build_student(model, args, quant_work_device=None) -> int:
    """The deployable student: ternarised banks + correction branches."""
    if getattr(args, "compact_banks", False):
        compact_banks(model, args, work_device=quant_work_device)
        memory_report(model, args, "build_student")
        if getattr(args, "force_gpu", False):
            # the pre-compaction --force-gpu could not fit the dense teacher;
            # now that the banks are gone, drop every hook and place the
            # (compact) student fully on the card.
            moved = harden_placement(model, args.device)
            n_re = rebind_compact_forwards(model)
            print(f"compact: harden-placement moved {moved} tensor(s) to "
                  f"{args.device}; re-bound {n_re} compact forwards", flush=True)
            # the frozen 1.18 GiB token table is fetched once per step; host
            # side it does not fit on the 48 GB card otherwise (P2 L40S).
            emb_key = offload_embed_tokens(model)
            if emb_key:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"compact: token embedding -> host ({emb_key})", flush=True)
            memory_report(model, args, "post-harden")
    else:
        ternarize_banks(model, args, work_device=quant_work_device)
    return attach_branches(model, args)


def attach_branches(model, args) -> int:
    """Wrap the MoE / attention outputs in correction branches (see module doc).

    Returns the number of trainable parameters.
    """
    layers = text_layers(model)
    cfg = getattr(model, "config", None)
    if cfg is None or not hasattr(cfg, "hidden_size"):
        cfg = cfg.text_config
    hidden = cfg.hidden_size
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
                    proj, proj.in_features, args.rank, args.branch_quant,
                    args.quant, out_dim=proj.out_features, gate=gate).to(dev)
    if gate != "none":
        print(f"branch gates: {gate} (identity at init)", flush=True)
    for name, p in model.named_parameters():
        p.requires_grad_(".branch." in name or ".gate." in name)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------ PLE handling ---

class SparseNGramTable(nn.Module):
    """Row-compact stand-in for the 320M-row PLE n-gram table.

    The official runtime table is ~95 GiB bf16 (unchanged in the fp8 release:
    the table itself is fp8 with a per-tensor scale).  Prefix work only ever
    looks up the rows the run's windows hash to, so the FP8 prefix loader
    gathers just those into ``rows`` and this module reproduces the embedding
    lookup for them.  A missing row raises instead of silently returning zeros.

    ``weight`` is exposed because the upstream ``Qwen4ExpTextNGramEmbedding``
    probes ``self.ngram_embedding.weight.device`` to place the gather.
    """

    def __init__(self, dim: int, padded_vocab_size: int):
        super().__init__()
        self.dim = dim
        self.padded_vocab_size = int(padded_vocab_size)
        self.out_dtype = torch.bfloat16
        self.register_buffer("rows", torch.zeros(0, dim, dtype=torch.bfloat16),
                             persistent=False)
        self.register_buffer("ids_sorted",
                             torch.zeros(0, dtype=torch.long), persistent=False)
        self._recording = False
        self._seen: list[torch.Tensor] = []

    @property
    def weight(self) -> torch.Tensor:
        return self.rows

    def start_recording(self):
        self._seen = []
        self._recording = True

    def stop_recording(self) -> torch.Tensor:
        self._recording = False
        if not self._seen:
            return torch.zeros(0, dtype=torch.long)
        return torch.unique(torch.cat(self._seen).long().cpu())

    def set_rows(self, ids_sorted: torch.Tensor, rows: torch.Tensor) -> None:
        if rows.shape[0] != ids_sorted.numel():
            raise ValueError("ids_sorted and rows disagree on the row count")
        self.rows = rows.to(self.out_dtype)
        # keep the lookup table on the rows' device: the loader calls this
        # before .to(device) (both CPU), the A/B re-quantises on the GPU model
        # (both CUDA) -- a mixed pair fails inside searchsorted.
        self.ids_sorted = ids_sorted.long().to(self.rows.device)

    def n_rows(self) -> int:
        return int(self.rows.shape[0])

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if self._recording:
            self._seen.append(ids.reshape(-1).detach().cpu())
            return torch.zeros(*ids.shape, self.dim, dtype=self.out_dtype,
                               device=ids.device)
        if self.rows.shape[0] == 0:
            raise RuntimeError(
                "PLE rows are not loaded; load the prefix with ple='rows' and "
                "the run's input ids (collect_ple_ids) before forwarding")
        flat = ids.reshape(-1)
        pos = torch.searchsorted(self.ids_sorted, flat.clamp(min=0))
        in_range = pos < self.ids_sorted.numel()
        pos_c = pos.clamp(max=max(self.ids_sorted.numel() - 1, 0))
        hit = in_range & (self.ids_sorted[pos_c] == flat)
        if not bool(hit.all()):
            missing = flat[~hit]
            raise KeyError(
                f"{missing.numel()} PLE rows are not loaded (e.g. "
                f"{missing[:5].tolist()}); the row set must cover every window "
                f"the forward will see")
        return self.rows[pos_c].reshape(*ids.shape, self.dim)


def _sparse_ngram_init(self, config, embedding_dim: int, layer_idx: int,
                       ple_layer_index: int = 0):
    """Mirror of ``Qwen4ExpTextNGramEmbedding.__init__`` without the 95 GiB table.

    Everything (head vocab sizes, offsets, hash multipliers) is built exactly as
    upstream; only the ``nn.Embedding`` is replaced by ``SparseNGramTable``.
    A test pins the buffers and the hashed ids against the upstream class on a
    tiny config.
    """
    nn.Module.__init__(self)
    self.layer_idx = layer_idx
    self.ngram_size = config.ngram_size
    self.context_len = self.ngram_size - 1
    self.heads_per_ngram = config.heads_per_ngram
    self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
    self.ple_layer_index = ple_layer_index
    self.unigram_vocab_size = config.vocab_size
    self.ngram_vocab_size_base = config.ngram_vocab_size_base
    head_dim_per_ngram = embedding_dim // self.ngram_heads
    self.seed = config.seed
    self.eos_token_id = (config.eos_token_id[0]
                         if isinstance(config.eos_token_id, list)
                         else config.eos_token_id)

    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        _build_layer_multipliers, _find_nth_prime_after)

    self.head_vocab_sizes = []
    self.head_offsets = []
    self.total_vocab_size = 0
    for head_idx in range(self.ngram_heads):
        global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
        size = _find_nth_prime_after(self.ngram_vocab_size_base - 1,
                                     global_head_idx + 1)
        self.head_vocab_sizes.append(size)
        self.head_offsets.append(self.total_vocab_size)
        self.total_vocab_size += size

    self.layer_multipliers = nn.Buffer(
        _build_layer_multipliers(self.unigram_vocab_size, self.ngram_size,
                                 self.ple_layer_index, self.seed))
    self.ngram_heads_vocab_sizes = nn.Buffer(
        torch.tensor(self.head_vocab_sizes, dtype=torch.long))
    self.ngram_heads_offsets = nn.Buffer(
        torch.tensor(self.head_offsets, dtype=torch.long))
    ngram_vocab_divisor = config.make_ngram_vocab_size_divisible_by
    padded_vocab_size = (math.ceil(self.total_vocab_size / ngram_vocab_divisor)
                         * ngram_vocab_divisor)
    self.ngram_embedding = SparseNGramTable(head_dim_per_ngram, padded_vocab_size)


class sparse_ple_construction:
    """Context manager: construct qwen4_exp models without the dense PLE table."""

    def __enter__(self):
        from transformers.models.qwen4_exp import modeling_qwen4_exp as M
        self._orig = M.Qwen4ExpTextNGramEmbedding.__init__
        M.Qwen4ExpTextNGramEmbedding.__init__ = _sparse_ngram_init
        return self

    def __exit__(self, *exc):
        from transformers.models.qwen4_exp import modeling_qwen4_exp as M
        M.Qwen4ExpTextNGramEmbedding.__init__ = self._orig
        return False


@torch.no_grad()
def collect_ple_ids(model, batches) -> torch.Tensor:
    """Unique PLE row ids the given input batches hash to (all PLE layers).

    ``batches`` must be the **exact** per-forward batches the run will use
    (``[1, S]`` rows, one per window).  The n-gram hash depends on segment
    boundaries, so a concatenated multi-window row produces different ids at
    the window starts than the per-window forwards do -- gather with what the
    model will actually see.
    """
    found = []
    for layer in text_layers(model):
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        emb = ple.ple_embedding
        tab = emb.ngram_embedding
        if not isinstance(tab, SparseNGramTable):
            continue
        tab.start_recording()
        for ids in batches:
            emb(ids, None)
        found.append(tab.stop_recording())
    if not found:
        return torch.zeros(0, dtype=torch.long)
    return torch.unique(torch.cat(found))


@torch.no_grad()
def load_ple_rows(needed_ids: torch.Tensor, layer_idx: int, weight_map: dict,
                  open_shard, dtype=torch.bfloat16):
    """Gather PLE rows for ``needed_ids`` from the sharded fp8 table.

    The runtime table is ``torch.cat([shard_0, shard_1, ...], dim=0)`` (the
    official loader concatenates the sorted parts), each part dequantised with
    the single per-tensor ``weight_scale``.  Only requested rows are
    materialised, so RAM stays at one part + the gathered rows.
    """
    parts = {}
    scale_key = None
    for k, s in weight_map.items():
        if f"layers.{layer_idx}." not in k:
            continue
        if "ngram_embedding.shard_" in k and k.endswith(".weight"):
            idx = int(k.split(".shard_")[1].split(".")[0])
            parts[idx] = (k, s)
        elif k.endswith("ngram_embedding.weight_scale"):
            scale_key = (k, s)
    if not parts:
        raise KeyError(f"no PLE shards found for layer {layer_idx} in the index")
    scale = open_shard(scale_key[1]).get_tensor(scale_key[0]).float().reshape(1)

    ids = torch.unique(needed_ids.long().cpu())
    out_ids, out_rows = [], []
    offset = 0
    for p in sorted(parts):
        key, shard = parts[p]
        t = open_shard(shard).get_tensor(key)
        r = t.shape[0]
        sel = (ids >= offset) & (ids < offset + r)
        if bool(sel.any()):
            local = (ids[sel] - offset)
            rows = t.index_select(0, local).to(torch.float32) * scale
            out_ids.append(ids[sel])
            out_rows.append(rows.to(dtype))
        offset += r
    if not out_ids:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, 0, dtype=dtype)
    ids_sorted = torch.cat(out_ids)
    rows = torch.cat(out_rows)
    order = torch.argsort(ids_sorted)
    return ids_sorted[order], rows[order]


def quantize_rows(rows: torch.Tensor, bits: int, group: int = 32
                  ) -> tuple[torch.Tensor, float]:
    """Symmetric per-group re-quantisation of PLE rows (the precision A/B).

    Returns ``(dequantized_rows, payload_bytes_per_row)``.  ``bits >= 16`` is a
    pass-through.  ``group`` must divide the row width (160 for the official
    table; group 32 = 5 groups/row).
    """
    if bits >= 16:
        return rows, rows.shape[-1] * 2.0
    width = rows.shape[-1]
    if width % group != 0:
        raise ValueError(f"group {group} does not divide row width {width}")
    g = rows.float().reshape(*rows.shape[:-1], width // group, group)
    if bits == 8:
        qmax = 127.0
    elif bits == 4:
        qmax = 7.0
    elif bits == 2:
        qmax = 1.0
    else:
        raise ValueError(f"unsupported PLE bits {bits} (8/4/2/16)")
    scale = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.clamp(torch.round(g / scale), -qmax, qmax)
    deq = (codes * scale).reshape(rows.shape).to(rows.dtype)
    bytes_per_row = width * bits / 8.0 + (width // group) * 2.0
    return deq, bytes_per_row


# ------------------------------------------------------------- FP8 loader ----

def _q4_text_config(model_dir: Path):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(model_dir))
    return getattr(cfg, "text_config", cfg)


def _load_index(model_dir: Path) -> dict:
    path = model_dir / "model.safetensors.index.json"
    if not path.exists():
        raise FileNotFoundError(f"no safetensors index at {path}")
    return json.loads(path.read_text())["weight_map"]


def _ple_layer_ids(tcfg) -> set:
    """0-based decoder-layer indices that carry PLE."""
    return {int(x) - 1 for x in (tcfg.ple_layer_ids or [])}


def fp8_prefix_plan(n_layers: int, model_dir: Path, include_ple: bool = True
                    ) -> dict:
    """Which shards and tensors an N-layer FP8 prefix needs.

    Returns ``{"tensors": {key: shard}, "expert_layers": [i...],
    "ple_layers": [i...], "shards": [file...]}``.  ``expert_layers`` lists the
    layers whose per-expert checkpoint tensors must be merged into fused banks.
    """
    weight_map = _load_index(model_dir)
    tcfg = _q4_text_config(model_dir)
    ple_layers = _ple_layer_ids(tcfg)
    tensors, expert_layers, expert_shards = {}, set(), set()
    for key, shard in weight_map.items():
        if key == "lm_head.weight":
            tensors[key] = shard
            continue
        if key.startswith("model.language_model."):
            stripped = key[len("model.language_model."):]
            if stripped.startswith("layers."):
                layer_idx = int(stripped.split(".")[1])
                if layer_idx >= n_layers:
                    continue
                if ".mlp.experts." in stripped:
                    expert_layers.add(layer_idx)
                    expert_shards.add(shard)
                    continue
                if "ngram_embedding.shard_" in stripped:
                    if include_ple and layer_idx in ple_layers:
                        tensors[key] = shard
                    continue
                if stripped.endswith("ngram_embedding.weight_scale"):
                    if include_ple and layer_idx in ple_layers:
                        tensors[key] = shard
                    continue
                tensors[key] = shard
            elif stripped in ("embed_tokens.weight",) or (
                    stripped.startswith("hyper_connection_mixer.")):
                tensors[key] = shard
            continue
        # everything else (mtp.*, model.visual.*) is not part of the prefix
    return {
        "tensors": tensors,
        "expert_layers": sorted(expert_layers),
        "ple_layers": sorted(ple_layers & set(range(n_layers))) if include_ple else [],
        "shards": sorted(set(tensors.values()) | expert_shards),
    }


def dequant_fp8_block(w: torch.Tensor, scale_inv: torch.Tensor,
                      block: int = FP8_BLOCK) -> torch.Tensor:
    """Dequantize a fine-grained fp8 weight with its 128x128 block scales.

    ``out[r, c] = w[r, c] * scale_inv[r // block, c // block]`` (the reference
    layout in ``transformers/integrations/finegrained_fp8.py``).  fp32 math,
    fp16-safe on the block boundary because the scale grid covers the tensor.
    """
    if w.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(w.shape)}")
    r, c = w.shape
    rows = math.ceil(r / block)
    cols = math.ceil(c / block)
    si = scale_inv[:rows, :cols].float()
    si = si.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:r, :c]
    return w.float() * si


def _expert_state(layer_idx: int, num_experts: int, ff: int, hidden: int,
                  weight_map: dict, open_shard, dtype) -> dict:
    """Merge per-expert checkpoint tensors into the fused banks, dequantized.

    ``weight_map`` is the full index mapping (expert tensors are not part of
    ``fp8_prefix_plan``'s kept set).  Merge order gate-then-up matches the
    official ``MergeModulelist`` + ``Concatenate`` conversion.
    """
    prefix = f"model.language_model.layers.{layer_idx}.mlp.experts."
    gu = torch.empty(num_experts, 2 * ff, hidden, dtype=dtype)
    dn = torch.empty(num_experts, hidden, ff, dtype=dtype)
    for e in range(num_experts):
        g_key = f"{prefix}{e}.gate_proj.weight"
        u_key = f"{prefix}{e}.up_proj.weight"
        d_key = f"{prefix}{e}.down_proj.weight"
        for k in (g_key, u_key, d_key):
            if k not in weight_map:
                raise KeyError(f"missing expert tensor {k} for layer {layer_idx}")
        g = dequant_fp8_block(
            open_shard(weight_map[g_key]).get_tensor(g_key),
            open_shard(weight_map[g_key + "_scale_inv"]).get_tensor(g_key + "_scale_inv"))
        u = dequant_fp8_block(
            open_shard(weight_map[u_key]).get_tensor(u_key),
            open_shard(weight_map[u_key + "_scale_inv"]).get_tensor(u_key + "_scale_inv"))
        d = dequant_fp8_block(
            open_shard(weight_map[d_key]).get_tensor(d_key),
            open_shard(weight_map[d_key + "_scale_inv"]).get_tensor(d_key + "_scale_inv"))
        gu[e] = torch.cat([g, u], dim=0).to(dtype)
        dn[e] = d.to(dtype)
    return {f"layers.{layer_idx}.mlp.experts.gate_up_proj": gu,
            f"layers.{layer_idx}.mlp.experts.down_proj": dn}


def load_fp8_prefix(n_layers: int, device: str = "cpu",
                    dtype: torch.dtype = torch.bfloat16,
                    model_dir: Path | None = None, shard_dir: Path | None = None,
                    ple: str = "rows", ple_ids: torch.Tensor | None = None,
                    verbose: bool = True):
    """Build an N-layer prefix from the official FP8 checkpoint shards.

    ``ple="rows"`` gathers only the rows ``ple_ids`` hash to (required when a
    PLE layer is in range; pass the run's windows **stacked** as ``[N, S]``,
    each row a batch of 1 -- the hash is boundary-sensitive); ``ple="none"``
    refuses a PLE layer in range because its table cannot be materialised.

    Returns ``(model, missing, unexpected, plan)`` -- no tokenizer (the caller
    already has the model dir).
    """
    from safetensors import safe_open
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel

    patch_indexer()
    model_dir = Path(model_dir or MODEL)
    shard_dir = Path(shard_dir) if shard_dir else (
        model_dir / PLE_SHARD_DIR if (model_dir / PLE_SHARD_DIR).is_dir() else model_dir)
    tcfg = _q4_text_config(model_dir)
    n_layers = min(int(n_layers), tcfg.num_hidden_layers)
    plan = fp8_prefix_plan(n_layers, model_dir, include_ple=(ple != "none"))
    if plan["ple_layers"] and ple != "rows":
        raise SystemExit(
            f"prefix layers {plan['ple_layers']} carry PLE; load with ple='rows' "
            f"and pass ple_ids (the dense table is ~95 GiB)")

    # prefix config: N layers, truncated layer types, PLE ids still in range
    tcfg.num_hidden_layers = n_layers
    tcfg.layer_types = list(tcfg.layer_types)[:n_layers]
    if hasattr(tcfg, "mtp_num_hidden_layers"):
        tcfg.mtp_num_hidden_layers = 0

    _prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with sparse_ple_construction():
            model = Qwen4ExpTextModel(tcfg)
    finally:
        torch.set_default_dtype(_prev)

    missing_shards = [s for s in plan["shards"] if not (shard_dir / s).exists()]
    if missing_shards:
        raise SystemExit(
            f"{len(missing_shards)} shard(s) missing under {shard_dir}: "
            f"{missing_shards[:3]}... run the fetcher first")

    handles: dict[str, object] = {}

    def open_shard(shard: str):
        h = handles.get(shard)
        if h is None:
            h = handles[shard] = safe_open(str(shard_dir / shard),
                                           framework="pt", device="cpu")
        return h

    state, loaded = {}, 0
    for key, shard in plan["tensors"].items():
        stripped = (key[len("model.language_model."):]
                    if key.startswith("model.language_model.") else key)
        if ".mlp.experts." in stripped:
            continue                              # assembled below
        if "ngram_embedding.shard_" in stripped or stripped.endswith(
                "ngram_embedding.weight_scale"):
            continue                              # PLE rows, gathered below
        t = open_shard(shard).get_tensor(key)
        if key.endswith(".weight") and (key + "_scale_inv") in plan["tensors"]:
            t = dequant_fp8_block(
                t, open_shard(plan["tensors"][key + "_scale_inv"]
                              ).get_tensor(key + "_scale_inv")).to(dtype)
        state[stripped] = t
        loaded += 1

    if "lm_head.weight" in state:
        # the bare text model has no head; attach one so the prefix can produce
        # logits (LM/KD and the smoke both need it), exactly like the 35B loader
        head = state["lm_head.weight"]
        model.lm_head = nn.Linear(tcfg.hidden_size, head.shape[0], bias=False)

    num_experts = int(tcfg.num_experts)
    ff = int(tcfg.moe_intermediate_size)
    hidden = int(tcfg.hidden_size)
    # assign the non-expert state, then stream the expert banks one layer at a
    # time.  Holding the full bf16 state (experts are ~121B params) next to the
    # constructed model OOM-kills the 286 GB pod cgroup on the 48-layer teacher
    # (2026-10-03); assign=True frees each layer's constructed originals as the
    # real tensors take their place.
    _missing, unexpected = model.load_state_dict(state, strict=False,
                                                 assign=True)
    del state
    gc.collect()
    full_map = _load_index(model_dir)
    for layer_idx in plan["expert_layers"]:
        banks = _expert_state(layer_idx, num_experts, ff, hidden,
                              full_map, open_shard, dtype)
        _m, _u = model.load_state_dict(banks, strict=False, assign=True)
        unexpected.extend(_u)
        del banks
        gc.collect()
        loaded += 2
    if verbose:
        print(f"fp8 prefix: {n_layers} layers, {loaded} checkpoint tensors, "
              f"{len(plan['shards'])} shards", flush=True)
    missing = [k for k, p in model.named_parameters() if p.is_meta]
    missing += [k for k, b in model.named_buffers() if b.is_meta]

    # PLE rows
    if plan["ple_layers"]:
        if ple_ids is None:
            raise SystemExit("PLE layers in range need ple_ids (the run's "
                             "per-window token batches) so only the used rows "
                             "are loaded")
        batches = [ple_ids[i:i + 1] for i in range(ple_ids.shape[0])]
        needed = collect_ple_ids(model, batches)
        weight_map = _load_index(model_dir)
        for layer_idx in plan["ple_layers"]:
            layer = model.layers[layer_idx]
            tab = layer.ple.ple_embedding.ngram_embedding
            tab.out_dtype = dtype
            ids_sorted, rows = load_ple_rows(needed, layer_idx, weight_map,
                                             open_shard, dtype=dtype)
            tab.set_rows(ids_sorted, rows)
            if verbose:
                print(f"PLE layer {layer_idx}: {tab.n_rows()} rows loaded "
                      f"({rows.numel() * rows.element_size() / 1e6:.1f} MB)",
                      flush=True)

    model.to(dtype=dtype, device=device)
    model.eval()
    return model, missing, unexpected, plan


def load_prefix(n_layers: int, device: str = "cpu",
                dtype: torch.dtype = torch.bfloat16,
                model_dir: Path | None = None):
    """Load embedding + N decoder layers from a converted bf16 checkpoint.

    Mirrors the 35B loader for a local bf16 mirror of the model.  PLE layers in
    range are built row-compact but their rows are *not* loaded here -- use
    ``load_fp8_prefix`` (the official release path) when PLE is involved.
    """
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel

    patch_indexer()
    model_dir = Path(model_dir or MODEL)
    tcfg = _q4_text_config(model_dir)
    tcfg.num_hidden_layers = n_layers
    tcfg.layer_types = list(tcfg.layer_types)[:n_layers]
    if hasattr(tcfg, "mtp_num_hidden_layers"):
        tcfg.mtp_num_hidden_layers = 0
    _prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with sparse_ple_construction():
            model = Qwen4ExpTextModel(tcfg)
    finally:
        torch.set_default_dtype(_prev)

    idx = _load_index(model_dir)
    handles, state, skipped = {}, {}, set()
    for key, shard in idx.items():
        if key.startswith("model.language_model."):
            stripped = key[len("model.language_model."):]
            if stripped.startswith("layers."):
                if int(stripped.split(".")[1]) >= n_layers:
                    continue
            elif stripped not in ("embed_tokens.weight",):
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
            h = handles[shard] = safe_open(str(path), framework="pt", device="cpu")
        state[stripped] = h.get_tensor(key)
    if skipped:
        print(f"warning: {len(skipped)} shard(s) not on disk; skipped "
              f"({len(state)} tensors loaded)", flush=True)
    if "lm_head.weight" in state:
        head = state["lm_head.weight"]
        model.lm_head = nn.Linear(tcfg.hidden_size, head.shape[0], bias=False)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    del state
    model.to(dtype=dtype, device=device)
    model.eval()
    return model, missing, unexpected


# ------------------------------------------------------------ cache record ---

def make_record(logits: torch.Tensor, args, generator=None) -> dict:
    """Per-window cache record, the 35B schema: idx/val/w[/tidx/tlp].

    ``logits`` is the teacher block already shifted to predict the next token
    (``[:, :-1]``).  Factored out of the stage so tests exercise exactly what
    the cache writes.
    """
    t_top = logits.topk(args.top_logits, dim=-1)
    # w: the teacher's own mass on the support it was cached at; the top-k KD
    # term renormalises over the support and is blind to it (--kd-tail-weight).
    wmass = support_mass(logits, t_top.indices)
    rec = {"idx": t_top.indices.cpu().to(torch.int32),
           "val": t_top.values.cpu().to(torch.float16),
           "w": wmass.cpu().to(torch.float16)}
    if args.tail_logits:
        # TAD's D_KL2 needs probabilities a top-k cache cannot hold, so the
        # teacher's tail conditional is sampled (Sparse Logit Sampling),
        # seeded per window so a rebuild samples the same tokens.
        tidx, tlp = sample_tail_tokens(logits, t_top.indices, args.tail_logits,
                                       generator=generator)
        rec["tidx"] = tidx.cpu().to(torch.int32)
        rec["tlp"] = tlp.cpu().to(torch.float16)
    return rec


def _corpus_windows(tok, args):
    from ayot import load_traces, mix_windows, windows_from_texts
    data = windows(tok, args.windows, args.seq, args.seed,
                   max_chars=args.corpus_chars)
    if args.corpus_file:
        agentic = windows_from_texts(tok, load_traces(args.corpus_file),
                                     args.windows, args.seq, args.seed)
        data = mix_windows(data, agentic, args.agentic_frac, args.seed)
        print(f"corpus: {int(round(args.windows * args.agentic_frac))}/"
              f"{args.windows} windows from {args.corpus_file}", flush=True)
    return data


def load_full(args):
    """Full teacher for the pod stages; prefix routes go through the manual
    FP8 loader (local) or ``from_pretrained`` (native fp8 on the pod)."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    patch_indexer(fast=getattr(args, "fast_indexer", False))
    model_dir = Path(args.model_dir) if args.model_dir else MODEL
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    if getattr(args, "prefix_layers", 0):
        if getattr(args, "fp8", True):
            model, _, _, _ = load_fp8_prefix(
                args.prefix_layers, args.device, model_dir=model_dir,
                shard_dir=Path(args.shard_dir) if args.shard_dir else None,
                ple=args.ple)
        else:
            model, _, _ = load_prefix(args.prefix_layers, args.device,
                                      model_dir=model_dir)
        return model, tok
    mm = None
    if args.max_memory:
        mm = {}
        for part in args.max_memory.split(","):
            k, v = part.split(":")
            mm[int(k) if k.strip().lstrip("-").isdigit() else k.strip()] = v.strip()
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, device_map=args.device_map,
        max_memory=mm)
    if getattr(args, "force_gpu", False):
        force_gpu_placement(model, args.device)
        # On 48 GB cards auto-offload can leave the token table on the host
        # (the move loop skips it when the card fills first); the wrapper
        # moves ids/outputs across devices.  The student path got this from
        # build_student; the teacher path needs it too (gate-teacher, first
        # full-forward on a 48 GB card: CPU embedding + cuda ids -> index_select
        # device error).
        emb_key = offload_embed_tokens(model)
        if emb_key:
            print(f"load_full: token embedding -> host ({emb_key})", flush=True)
        n_small = place_unmapped_small(model, args.device)
        if n_small:
            print(f"load_full: moved {n_small} unmapped small module(s) to "
                  f"{args.device}", flush=True)
        n_ess = place_essential(model, args.device)
        if n_ess:
            print(f"load_full: moved {n_ess} essential module(s) to "
                  f"{args.device}", flush=True)
        try:  # placement diagnostic for the 48 GB full path
            emb = model.get_input_embeddings()
            head = model.get_output_embeddings()
            ew = getattr(emb, "weight", None)
            hw = getattr(head, "weight", None)
            print(f"load_full devices: embed={ew.device if ew is not None else None} "
                  f"head={hw.device if hw is not None else None}", flush=True)
        except Exception:
            pass
    return model, tok


def unmapped_small_modules(model, max_gib: float = 2.0) -> list[str]:
    """Names of small modules accelerate did not map (rotary buffers, norms,
    the output head).  These stay on the CPU with device_map=auto and break
    the forward on 48 GB cards (first full-forward device errors: RoPE bmm,
    head matmul).  Modules inside or above a mapped key are skipped so the
    offloaded decoder blocks keep their hooks; PLE/visual stay host-side.
    """
    hm = set(getattr(model, "hf_device_map", None) or {})
    out = []
    for name, mod in model.named_modules():
        if not name or "ple" in name or "visual" in name \
                or "embed_tokens" in name:
            continue
        if name in hm or any(name.startswith(k + ".") for k in hm) \
                or any(k.startswith(name + ".") for k in hm):
            continue
        nbytes = sum(p.numel() * p.element_size() for p in mod.parameters())
        nbytes += sum(b.numel() * b.element_size() for b in mod.buffers())
        if 0 < nbytes <= max_gib * 2**30:
            out.append(name)
    return out


def place_unmapped_small(model, device: str, max_gib: float = 2.0) -> int:
    """Move the unmapped small modules onto the execution device."""
    moved = 0
    for name in unmapped_small_modules(model, max_gib):
        try:
            model.get_submodule(name).to(device)
            moved += 1
        except (RuntimeError, ValueError) as exc:
            print(f"place-small: keeping {name} off-device ({exc})", flush=True)
    if moved and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return moved


def essential_module_names(model, max_gib: float = 2.0) -> list[str]:
    """Rotary embeddings and the output head: small and essential, and their
    accelerate hooks do not move the direct child-call inputs (position_ids /
    hidden states), so they must sit on the execution device even when mapped
    to the host."""
    out = []
    for name, mod in model.named_modules():
        if not (name.endswith("rotary_emb") or name.endswith("lm_head")):
            continue
        if "visual" in name or "ple" in name:
            continue
        nbytes = sum(p.numel() * p.element_size() for p in mod.parameters())
        nbytes += sum(b.numel() * b.element_size() for b in mod.buffers())
        if 0 < nbytes <= max_gib * 2**30:
            out.append(name)
    return out


def place_essential(model, device: str, max_gib: float = 2.0) -> int:
    """Force the rotary/head modules onto the device (hook removed first)."""
    try:
        from accelerate import hooks as ah
    except ImportError:
        ah = None
    dev = torch.device(device)
    moved = 0
    for name in essential_module_names(model, max_gib):
        mod = model.get_submodule(name)
        tensors = list(mod.parameters()) + list(mod.buffers())
        if tensors and all(t.device.type == dev.type for t in tensors):
            continue
        if ah is not None:
            try:
                ah.remove_hook_from_module(mod, recurse=True)
            except Exception:
                pass
        try:
            mod.to(dev)
            moved += 1
        except (RuntimeError, ValueError) as exc:
            print(f"place-essential: keeping {name} off-device ({exc})", flush=True)
    if moved and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return moved


def force_gpu_placement(model, device: str, reserve_gib: float = 2.0) -> list[str]:
    """Move accelerate-offloaded layers onto the GPU, dropping their hooks.

    ``device_map="auto"`` estimates with the config dtype and offloads 12-22 of
    the 48 layers even though the fp8 weights leave ~15 GiB of headroom; the
    per-forward pageable HtoD copies of those layers are ~95% of the window
    time (measured on the H200).  The PLE subtree is left alone: it is meant to
    stay host-side (``_no_placement_params``) and its weight may be meta.

    The card is a few GiB short of the full 48 layers, so each block is moved
    only if its bytes fit the current free memory minus a reserve; the rest
    stay hooked (a partial win is the difference between ~2 s and ~12 s per
    window).  Returns the list of moved module keys.
    """
    from accelerate import hooks as ah

    # 1) The official loader places the 47.7 GiB PLE table on the card when it
    # "fits"; the table is designed for host offload (the upstream forward
    # moves ids to the table's device and the rows back -- but its accelerate
    # hook still routes to the card, so drop the hook first).  The 48 GiB is
    # better spent holding the layers accelerate pushed to CPU.
    for layer in text_layers(model):
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        try:
            ah.remove_hook_from_module(ple, recurse=True)
        except (ValueError, AttributeError):
            pass
        emb = getattr(getattr(ple, "ple_embedding", None), "ngram_embedding", None)
        weight = getattr(emb, "weight", None) if emb is not None else None
        if weight is not None and weight.device.type != "cpu":
            try:
                emb.to("cpu")
                print(f"force-gpu: PLE table -> cpu (layer {ple.layer_idx})",
                      flush=True)
            except (RuntimeError, ValueError) as exc:
                print(f"force-gpu: keeping the PLE table on device ({exc})",
                      flush=True)
    n_ple = place_ple_non_table(model, device)
    if n_ple:
        print(f"force-gpu: moved {n_ple} PLE tensor(s) to {device}", flush=True)
    torch.cuda.empty_cache()

    def subtree_bytes(mod) -> int:
        total = sum(p.numel() * p.element_size() for p in mod.parameters())
        total += sum(b.numel() * b.element_size() for b in mod.buffers())
        return total

    hm = getattr(model, "hf_device_map", None) or {}
    moved, skipped = [], []
    for key, dev in list(hm.items()):
        if str(dev) in ("0", "cuda:0", "cpu0"):
            continue
        if ".ple" in key or "ple_embedding" in key:
            continue                      # host-side by design
        if key.endswith("layers.1"):
            continue                      # carries the PLE subtree
        try:
            mod = model.get_submodule(key)
        except AttributeError:
            continue
        need = subtree_bytes(mod)
        free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
        if need + reserve_gib * 2**30 > free:
            skipped.append(key)
            continue
        try:
            ah.remove_hook_from_module(mod, recurse=True)
            mod.to(device)
            moved.append(key)
        except (RuntimeError, ValueError) as exc:
            skipped.append(key)
            print(f"force-gpu: keeping {key} off-device ({exc})", flush=True)
    torch.cuda.empty_cache()
    if moved:
        print(f"force-gpu: moved {len(moved)} block(s) to {device}: "
              f"{moved[:4]}{'...' if len(moved) > 4 else ''}", flush=True)
    if skipped:
        print(f"force-gpu: {len(skipped)} block(s) left off-device (no room): "
              f"{skipped[:4]}{'...' if len(skipped) > 4 else ''}", flush=True)
    return moved


# ------------------------------------------------------------------ stages ---

def build_tiny_model(seed: int = 0):
    """Random-init qwen4_exp text model + head -- the local plumbing rig.

    Three layers cover all block types (GDN, GDN+PLE, QSA full attention); the
    tokenizer-independent ids make it runnable anywhere (no checkpoint, no
    corpus).  Same construction as ``moe/tests/test_qwen4exp_port.py``.
    """
    from transformers import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel
    cfg = Qwen4ExpTextConfig(
        vocab_size=1024, eos_token_id=1000, bos_token_id=1000,
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16,
        linear_conv_kernel_dim=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=4,
        linear_num_value_heads=8,
        moe_intermediate_size=32, shared_expert_intermediate_size=32,
        num_experts=8, num_experts_per_tok=2,
        layer_types=["linear_attention", "linear_attention", "full_attention"],
        hc_count=4, hc_lowrank=8,
        ple_layer_ids=[2], ple_embed_dim=64, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=64,
        make_ngram_vocab_size_divisible_by=16,
        indexer_n_heads=1, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=8, indexer_compress_ratio=4,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default",
                         "partial_rotary_factor": 0.25,
                         "mrope_section": [3, 3, 2], "mrope_interleaved": True},
    )
    patch_indexer()
    torch.manual_seed(seed)
    model = Qwen4ExpTextModel(cfg)
    model.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    return model, cfg


def stage_smoke(args):
    """Prefix smoke: FP vs ternary drift + a few correction steps (port check)."""
    patch_experts(args.group)
    model_dir = Path(args.model_dir) if args.model_dir else MODEL
    if args.tiny:
        model, tcfg = build_tiny_model()
        vocab = tcfg.vocab_size
        gen = torch.Generator().manual_seed(999)
        data = torch.randint(0, vocab, (2, args.seq), generator=gen)
        model.to(args.device)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        data = windows(tok, 2, args.seq, 999, "wikitext")
        if args.fp8:
            # PLE layers in range need the rows the smoke windows hash to
            ple_ids = torch.stack([data[i] for i in range(len(data))])
            model, _, _, _ = load_fp8_prefix(args.layers, args.device,
                                             model_dir=model_dir, ple=args.ple,
                                             ple_ids=ple_ids)
        else:
            model, _, _ = load_prefix(args.layers, args.device, model_dir=model_dir)
    ids = data[0:1].to(args.device)

    def capture():
        routes = {}
        handles = [layer.mlp.gate.register_forward_hook(gate_hook(routes, i))
                   for i, layer in enumerate(text_layers(model))]
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        for h in handles:
            h.remove()
        return out, routes

    experts = [layer.mlp.experts for layer in text_layers(model)]
    for e in experts:
        e._ternary = False
    fp_out, fp_route = capture()
    for e in experts:
        e._ternary = True
    tern_out, tern_route = capture()
    hidden = getattr(fp_out, "last_hidden_state", None)
    if hidden is None:
        hidden = fp_out.hidden_states[-1]
        tern_hidden = tern_out.hidden_states[-1]
    else:
        tern_hidden = tern_out.last_hidden_state
    drift = float((tern_hidden - hidden).norm() / (hidden.norm() + 1e-12))
    agree = float(torch.stack([
        (fp_route[i][2].unsqueeze(-1) == tern_route[i][2].unsqueeze(-2)).any(-1).float().mean()
        for i in fp_route]).mean())
    print(f"FP vs ternary: hidden drift {drift:.4f}, "
          f"router top-{model.config.num_experts_per_tok} agreement {agree:.4f}",
          flush=True)

    # freeze the deployable body: banks ternarised in place (chunked) or held
    # in deployed compact codes+scales; no STE temporaries either way.  The
    # correction branches then train on top (the real recipe).
    if getattr(args, "compact_banks", False):
        compact_banks(model, args)
        memory_report(model, args, "smoke")
    else:
        ternarize_banks(model, args)
    for layer in text_layers(model):
        layer.mlp = MoEWithCorrection(layer.mlp, model.config.hidden_size,
                                      args.rank, args.branch_quant).to(args.device)
    for name, p in model.named_parameters():
        p.requires_grad_(".branch." in name or ".gate." in name)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_tr/1e6:.2f}M (branches + routers)", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adafactor(params, lr=args.lr, weight_decay=0.0)
    model.train()
    for step in range(1, args.steps + 1):
        out = model(input_ids=ids, use_cache=False)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out.hidden_states[-1]
        logits = model.lm_head(hidden)
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                             ids[:, 1:].reshape(-1))
        lm.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 5 == 0 or step == 1:
            print(f"smoke step {step}: lm {lm.item():.4f}", flush=True)
    print("smoke ok", flush=True)


def stage_compact_check(args):
    """G2(c): 2-layer real-weights compact-bank one-step LM+KD smoke.

    Loads the fp8 prefix with the run's PLE rows, records the teacher logits for
    one window, encodes the banks to deployed codes+scales (verifying each
    decoded expert is bit-equal to the current in-place banker), attaches the
    correction branches, and runs one LM+KD step at the real recipe weights.
    """
    from transformers import AutoTokenizer
    model_dir = Path(args.model_dir) if args.model_dir else MODEL
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    data = windows(tok, 1, args.seq, args.seed, args.split)
    ple_ids = torch.stack([data[i] for i in range(len(data))])

    patch_experts(args.group)
    model, _, _, plan = load_fp8_prefix(
        args.layers, args.device, model_dir=model_dir,
        shard_dir=Path(args.shard_dir) if args.shard_dir else None,
        ple=args.ple, ple_ids=ple_ids)
    model.eval()

    ids = data[0:1].to(args.device)
    with torch.no_grad():
        t_logits = model_logits(model, ids)[:, :-1].float().cpu()
    print(f"teacher logits {tuple(t_logits.shape)}", flush=True)

    compact_banks(model, args, verify=True)     # bit-equality gate
    memory_report(model, args, "compact-check")

    n_tr = attach_branches(model, args)
    if args.balance != "none":
        patch_router_balance(model, args.balance, args.balance_cb_eta,
                             args.balance_qb_damp)
    print(f"trainable {n_tr/1e6:.2f}M (branches + routers)", flush=True)

    gen = torch.Generator().manual_seed(int(args.seed))
    rec = make_record(t_logits, args, generator=gen)
    tv = rec["val"].to(args.device).float()
    ti = rec["idx"].to(args.device)
    w_t = rec["w"].to(args.device).float().reshape(-1) if "w" in rec else None

    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    logits = model_logits(model, ids)
    lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                         ids[:, 1:].reshape(-1))
    s_sel = logits[:, :-1].gather(-1, ti).reshape(-1, ti.shape[-1])
    kd = F.kl_div(F.log_softmax(s_sel.float() / args.temp, dim=-1),
                  F.log_softmax(tv.reshape(-1, tv.shape[-1]) / args.temp,
                                dim=-1),
                  log_target=True, reduction="batchmean") * (args.temp ** 2)
    loss = lm + args.kd_weight * kd
    tail = tcond = None
    if args.kd_tail_weight > 0 or args.kd_tailcond_weight > 0:
        w_s, lse_s = support_mass_lse(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            ti.reshape(-1, ti.shape[-1]))
    if args.kd_tail_weight > 0:
        tail = residual_mass_kl(w_s, w_t).mean()
        loss = loss + args.kd_tail_weight * tail
    if args.kd_tailcond_weight > 0:
        m = rec["tidx"].shape[-1]
        tcond = tail_conditional_piece(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            rec["tidx"].to(args.device).reshape(-1, m),
            rec["tlp"].to(args.device).float().reshape(-1, m),
            ti.reshape(-1, ti.shape[-1]), w_t,
            student_mass=w_s, student_lse=lse_s).mean()
        loss = loss + args.kd_tailcond_weight * tcond
    loss.backward()
    grads = [p.grad for p in model.parameters()
             if p.requires_grad and p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in grads) and bool(grads)
    print(f"compact-check: lm {lm.item():.4f} kd {kd.item():.4f} "
          f"tail {0.0 if tail is None else tail.item():.4f} "
          f"tcond {0.0 if tcond is None else tcond.item():.4f} "
          f"total {loss.item():.4f} grads_finite={finite} n_grad={len(grads)}",
          flush=True)
    if not finite:
        raise SystemExit("compact-check FAILED: missing or non-finite gradients")
    memory_report(model, args, "compact-check-after-step")
    print("compact-check ok", flush=True)


def stage_qat_screen(args):
    """PLE-QAT convergence screen: two short LM arms on the 2-layer mirror.

    Arm ``qat0`` trains branches with full-precision PLE rows; arm ``qat2``
    trains with the 2-bit STE hook (``apply_ple_qat``).  Each arm then reads
    its 2-bit damage ``KLD(rows@2bit || rows@full)`` on held-out windows.
    A working hook shows (a) both arms converging (LM falling, grads finite)
    and (b) the QAT arm with the smaller damage number — the model adapted
    to the quantized table.
    """
    from kld_eval import log_probs, kld_from_logprobs, top1_agreement
    from transformers import AutoTokenizer
    model_dir = Path(args.model_dir) if args.model_dir else MODEL
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    train_data = windows(tok, args.qat_train_windows, args.seq, 1001, args.split)
    eval_data = windows(tok, args.qat_eval_windows, args.seq, 999, "wikitext")
    if args.qat_eval_same:
        eval_data = train_data[:args.qat_eval_windows \
            if args.qat_eval_windows <= len(train_data) else len(train_data)]
    ple_ids = torch.stack([train_data[i] for i in range(len(train_data))] +
                          [eval_data[i] for i in range(len(eval_data))])
    out = {"layers": args.layers, "steps": args.qat_steps,
           "rank": args.rank, "train_windows": len(train_data),
           "eval_windows": len(eval_data),
           "eval_same_as_train": bool(args.qat_eval_same), "arms": {}}

    def run_arm(qat_bits):
        patch_experts(args.group)
        model, _, _, plan = load_fp8_prefix(
            args.layers, args.device, model_dir=model_dir,
            shard_dir=Path(args.shard_dir) if args.shard_dir else None,
            ple="rows", ple_ids=ple_ids)
        ternarize_banks(model, args)
        n_br = attach_branches(model, args)
        for name, p in model.named_parameters():
            p.requires_grad_(".branch." in name or ".gate." in name)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adafactor(params, lr=args.lr, weight_decay=0.0)
        n_qat = apply_ple_qat(model, qat_bits, args.ple_qat_group)
        model.train()
        lms = []
        finite = True
        for step in range(1, args.qat_steps + 1):
            ids = train_data[(step - 1) % len(train_data):
                             (step - 1) % len(train_data) + 1].to(args.device)
            logits = model_logits(model, ids)
            lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                 ids[:, 1:].reshape(-1))
            lm.backward()
            grads = [p.grad for p in params if p.grad is not None]
            finite = finite and bool(grads) and all(
                bool(torch.isfinite(g).all()) for g in grads)
            opt.step()
            opt.zero_grad(set_to_none=True)
            lms.append(float(lm.item()))
            if step % 10 == 0 or step == 1:
                print(f"qat-screen qat{qat_bits} step {step}: lm {lm.item():.4f}",
                      flush=True)
        # damage read: 2-bit rows vs the arm's own full-precision rows.
        # (grads were already checked inside the loop, before each zero_grad;
        # checking here would see only Nones from set_to_none=True.)
        model.eval()
        tables = [(model.layers[i].ple.ple_embedding.ngram_embedding, i)
                  for i in plan["ple_layers"]]
        base = [(t.ids_sorted.clone(), t.rows.clone()) for t, _ in tables]

        def ev():
            logps = []
            with torch.no_grad():
                for i in range(len(eval_data)):
                    ids = eval_data[i:i + 1].to(args.device)
                    logps.append(log_probs(model_logits(model, ids)[0, :-1],
                                           args.chunk).cpu())
            return logps

        ref = ev()
        for k, (tab, _) in enumerate(tables):
            ids_sorted, rows = base[k]
            deq, _ = quantize_rows(rows, 2, args.ple_group)
            tab.set_rows(ids_sorted, deq)
        cur = ev()
        per_token = torch.cat([kld_from_logprobs(ref[i], cur[i], args.chunk)
                               for i in range(len(eval_data))])
        agree = sum(top1_agreement(ref[i], cur[i])
                    for i in range(len(eval_data))) / len(eval_data)
        return {"qat_tables": n_qat, "branches": n_br,
                "lm_first": lms[0], "lm_last": lms[-1], "grads_finite": finite,
                "dmg_mean": float(per_token.mean()),
                "dmg_p99": float(torch.quantile(per_token, 0.99)),
                "dmg_max": float(per_token.max()),
                "dmg_top1": float(agree)}

    for bits in (0, args.ple_qat_bits or 2):
        arm = run_arm(bits)
        out["arms"][f"qat{bits}"] = arm
        print(f"qat-screen arm qat{bits}: lm {arm['lm_first']:.4f} -> "
              f"{arm['lm_last']:.4f} grads_finite={arm['grads_finite']} "
              f"2bit-damage mean {arm['dmg_mean']:.4f} "
              f"p99 {arm['dmg_p99']:.4f} max {arm['dmg_max']:.4f} "
              f"top1 {arm['dmg_top1']:.4f}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}", flush=True)


@torch.no_grad()
def stage_cache(args):
    model, tok = load_full(args)
    data = _corpus_windows(tok, args)
    path = Path(args.cache_file) if args.cache_file else CACHE
    partial = Path(str(path) + ".partial")
    recs: list = []
    start = 0
    if getattr(args, "resume_cache", False) and partial.exists():
        recs = torch.load(partial, map_location="cpu")
        start = len(recs)
        print(f"resuming cache: {start} windows already in {partial}", flush=True)
    for w in range(start, len(data)):
        ids = data[w:w + 1].to(args.device)
        store = {}
        handles = [layer.mlp.gate.register_forward_hook(gate_hook(store, i))
                   for i, layer in enumerate(text_layers(model))]
        logits = model_logits(model, ids)
        for h in handles:
            h.remove()
        router = {i: (store[i][2].cpu().to(torch.int16),
                      store[i][1].cpu().to(torch.float16)) for i in store}
        gen = None
        if args.tail_logits:
            gen = torch.Generator(device=logits.device).manual_seed(
                int(args.seed) * 1_000_003 + w)
        rec = make_record(logits[:, :-1], args, generator=gen)
        rec["router"] = router
        recs.append(rec)
        if w % 100 == 0:
            print(f"cached {w}/{len(data)} (teacher support mass "
                  f"{float(rec['w'].mean()):.4f})", flush=True)
        # incremental park: an interrupt costs minutes, not the whole pass
        if args.save_every and (w + 1) % args.save_every == 0:
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save(recs, partial)
            print(f"  partial cache saved ({len(recs)} windows, "
                  f"{partial.stat().st_size/1e9:.2f} GB)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(recs, path)
    if partial.exists():
        partial.unlink()
    print(f"wrote {path} ({path.stat().st_size/1e9:.2f} GB)", flush=True)


@torch.no_grad()
def stage_ref(args):
    """Precompute teacher router refs for the in-run eval windows."""
    from transformers import AutoTokenizer
    model_dir = Path(args.model_dir) if args.model_dir else MODEL
    tok = AutoTokenizer.from_pretrained(str(model_dir))
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


def stage_train(args):
    patch_experts(args.group)
    model, tok = load_full(args)
    n_tr = build_student(model, args)
    print(f"trainable {n_tr/1e6:.2f}M (branches + routers) "
          f"target {args.branch_target}", flush=True)
    if getattr(args, "compact_banks", False):
        memory_report(model, args, "train")
    if args.balance != "none":
        n_gates = patch_router_balance(model, args.balance,
                                       args.balance_cb_eta, args.balance_qb_damp)
        print(f"router balance: {args.balance} on {n_gates} gates "
              f"(delta {args.balance_delta}, z {args.balance_z_coeff})", flush=True)

    if getattr(args, "grad_checkpoint", False):
        n_ck = enable_grad_checkpointing(
            model, getattr(args, "checkpoint_mode", "block"))
        if getattr(args, "checkpoint_mode", "block") == "group":
            # bound the decoded-weight saves inside the recompute (P2 L40S:
            # block mode saves every hit expert's weights, ~5 GiB)
            n_g = 0
            for layer in text_layers(model):
                m = moe_block(layer).experts
                if getattr(m, "_compact", False):
                    m._ckpt_group = max(1, int(getattr(args, "expert_group_size", 32)))
                    n_g += 1
            print(f"gradient checkpointing: group size "
                  f"{getattr(args, 'expert_group_size', 32)} on {n_g} modules",
                  flush=True)
        print(f"gradient checkpointing ({getattr(args, 'checkpoint_mode', 'block')}): "
              f"{n_ck} experts modules", flush=True)
        if getattr(args, "offload_saved", False):
            # the decoder layers are NOT checkpointed: autograd's saved tensors
            # go to host RAM through offload_saved_tensors (the checkpoint's own
            # recomputation hooks shadow any user hooks, so the layer saves
            # must not be inside a checkpoint; the expert blocks stay
            # checkpointed because their decoded weights would be huge).
            print("gradient checkpointing: decoder layers not checkpointed; "
                  "saved tensors -> host RAM", flush=True)
        else:
            # L40S fit without host offload: the decoder layers' saved
            # activations (~2.6 GiB over 48 layers) do not fit next to the
            # 41 GiB student; checkpoint them too (exact recompute).
            enable_layer_checkpointing(model)
            print("gradient checkpointing (decoder layers): enabled", flush=True)

    cache_path = Path(args.cache_file) if args.cache_file else CACHE
    cache = torch.load(cache_path, map_location="cpu")
    data = _corpus_windows(tok, args)

    ref = None
    ev = None
    if args.eval_every:
        if not args.ref_file:
            raise SystemExit("--eval-every needs --ref-file (build with 'ref')")
        ev = windows(tok, 2, args.seq, 999, "wikitext")
        ref = torch.load(args.ref_file, map_location="cpu")
        print(f"loaded eval refs from {args.ref_file}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    else:
        opt = torch.optim.Adafactor(params, lr=args.lr, weight_decay=0.0)
    n_qat = apply_ple_qat(model, getattr(args, "ple_qat_bits", 0) or 0,
                          getattr(args, "ple_qat_group", 32) or 32)
    if n_qat:
        print(f"PLE-QAT: {args.ple_qat_bits}-bit STE hook on {n_qat} tables",
              flush=True)
    start_step = 0
    if getattr(args, "resume", ""):
        start_step = load_resume_weights(model, args.resume, args)
        lr0 = lr_at_step(args, start_step)
        for g in opt.param_groups:
            g["lr"] = lr0
        print(f"resuming at step {start_step} (lr {lr0:.3e}; optimizer "
              f"restarts fresh, branches/routers restored)", flush=True)
    model.train()
    step = start_step
    start_offset = resume_offsets(start_step, len(cache))
    for epoch in range(args.epochs):
        for ci, rec in enumerate(cache):
            if epoch == 0 and ci < start_offset:
                continue
            bal, zl = None, None
            ids = data[step % len(data):step % len(data) + 1].to(args.device)
            off_ctx = (offload_saved_tensors(args.device)
                       if getattr(args, "offload_saved", False)
                       else contextlib.nullcontext())
            with off_ctx:
                logits = model_logits(model, ids)
                lm = chunked_cross_entropy(logits[:, :-1], ids[:, 1:])
            ti = rec["idx"].to(logits.device)
            tv = rec["val"].to(logits.device).float()
            s_sel = logits[:, :-1].gather(-1, ti).reshape(-1, ti.shape[-1])

            w_t = None
            if (args.kd_support_w == "wt" or args.kd_tail_weight > 0
                    or args.kd_tailcond_weight > 0):
                if "w" not in rec:
                    raise SystemExit(
                        f"--kd-support-w/--kd-tail-* need a cache with per-token "
                        f"teacher support mass, and {cache_path} has none. "
                        f"Rebuild it: cache with --tail-logits etc.")
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
                w_s, lse_s = support_mass_lse(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    ti.reshape(-1, ti.shape[-1]))
            if args.kd_tail_weight > 0:
                tail = residual_mass_kl(w_s, w_t).mean()
                loss = loss + args.kd_tail_weight * tail
            if args.kd_tailcond_weight > 0:
                if "tidx" not in rec or "tlp" not in rec:
                    raise SystemExit(
                        f"--kd-tailcond-weight needs a cache with sampled tail "
                        f"tokens, and {cache_path} has none (cache --tail-logits)")
                m = rec["tidx"].shape[-1]
                tcond = tail_conditional_piece(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    rec["tidx"].to(logits.device).reshape(-1, m),
                    rec["tlp"].to(logits.device).float().reshape(-1, m),
                    ti.reshape(-1, ti.shape[-1]), w_t,
                    student_mass=w_s, student_lse=lse_s).mean()
                loss = loss + args.kd_tailcond_weight * tcond
            if args.balance == "zloss":
                zl, bal = balance_z_loss(model, args.balance_z_coeff)
                loss = loss + zl
            if args.log_entropy:
                with torch.no_grad():
                    flat = logits[:, :-1].reshape(-1, logits.shape[-1]).float()
                    lp = F.log_softmax(flat, dim=-1)
                    ent = float(-(lp.exp() * lp).sum(-1).mean())
                    peak = float(lp.exp().max(-1).values.mean())
                    del lp, flat
            with off_ctx:
                loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if args.balance in ("bias", "quantile", "cbqb", "cb"):
                bal = balance_update(model, args.balance, args.balance_delta)
            step += 1
            if step <= 5 and torch.cuda.is_available():
                print(f"  [mem] step {step} alloc "
                      f"{torch.cuda.memory_allocated()/2**30:.2f} GiB peak "
                      f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB",
                      flush=True)
            if (args.lr_half_every and step >= args.lr_decay_start
                    and step % args.lr_half_every == 0):
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                print(f"step {step}: lr -> {opt.param_groups[0]['lr']:.3e}",
                      flush=True)
            if step % args.log_every == 0:
                extra = ""
                if args.log_entropy:
                    extra = f" H {ent:.3f} peak {peak:.4f}"
                if tail is not None:
                    extra += (f" mass s {float(w_s.mean()):.4f}"
                              f" t {float(w_t.mean()):.4f}")
                if tcond is not None:
                    extra += f" tcond {float(tcond):.4f}"
                if zl is not None:
                    extra += f" zl {zl.item():.2e}"
                if bal is not None:
                    extra += f" loadH {bal['mean_load_entropy']:.3f}"
                    seqv = bal.get("mean_seq_load_var", float("nan"))
                    if math.isfinite(seqv):
                        extra += f" seqvar {seqv:.5f}"
                print(f"step {step} lm {lm.item():.4f} kd {kd.item():.4f} "
                      f"tail {0.0 if tail is None else tail.item():.4f} "
                      f"total {loss.item():.4f}{extra}", flush=True)
            # save before the (monitoring-only) eval: an eval bug must never
            # cost the checkpoint (the step-1000 P2b crash lost the run this way)
            if args.ckpt_every and step % args.ckpt_every == 0:
                save(model, args, step)
            if args.eval_every and step % args.eval_every == 0 and ev is not None:
                try:
                    ppl, ag = quick_eval(model, ev, args, ref)
                    print(f"  [eval] step {step} ppl {ppl:.2f} "
                          f"router_agree {ag:.4f}", flush=True)
                except Exception as exc:  # noqa: BLE001 - keep training
                    model.train()
                    print(f"  [eval] step {step} failed: {exc!r} "
                          f"(training continues)", flush=True)
            if args.steps and step >= args.steps:
                break
        if args.steps and step >= args.steps:
            break
    save(model, args, step)
    stats = gate_stats(model)
    if stats:
        OUT.mkdir(parents=True, exist_ok=True)
        gpath = OUT / f"qwen4exp-branch-gates-{getattr(args, 'tag', '') or 'run'}.json"
        gpath.write_text(json.dumps(stats, indent=2))
        for r in stats:
            print("gate " + json.dumps(r), flush=True)
        print(f"wrote {gpath}", flush=True)
    print("training done", flush=True)


@torch.no_grad()
@torch.no_grad()
def quick_eval(model, data, args, ref=None):
    model.eval()
    total, ntok, agree = 0.0, 0, []
    for i in range(len(data)):
        ids = data[i:i + 1].to(args.device)
        store = {}
        # unwrap the correction wrapper: after attach_branches the block is a
        # MoEWithCorrection and the gate lives one level down (P2b bug #12)
        hs = [moe_block(layer).gate.register_forward_hook(gate_hook(store, j))
              for j, layer in enumerate(text_layers(model))]
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


def save(model, args, step):
    """Write the branch state dict (``--tag`` is part of the filename)."""
    sd = {k: v for k, v in model.state_dict().items()
          if ".branch." in k or ".gate." in k}
    tag = "" if args.branch_quant == "fp32" else f"-{args.branch_quant}"
    name = f"qwen4exp-corr-r{args.rank}{tag}-step{step}"
    if getattr(args, "tag", ""):
        name += f"-{args.tag}"
    p = OUT / f"{name}.pt"
    torch.save(sd, p)
    print(f"saved {p}", flush=True)


def resume_offsets(start_step: int, cache_len: int) -> int:
    """First-epoch cache offset for ``--resume`` (records before it are done)."""
    if not start_step:
        return 0
    return start_step % max(1, cache_len)


def parse_resume_step(path: str, explicit: int = 0) -> int:
    """Step number for ``--resume``: explicit flag wins, else ``step<N>``."""
    if explicit:
        return int(explicit)
    m = re.search(r"step(\d+)", Path(path).name)
    if m:
        return int(m.group(1))
    raise SystemExit(f"--resume {path}: no step<N> in the filename and no "
                     f"--resume-step given")


def lr_at_step(args, step: int) -> float:
    """Adafactor LR at an absolute step under the halving schedule.

    Mirrors the training loop exactly: LR halves at every multiple of
    ``lr_half_every`` at/after ``lr_decay_start`` (step 0 never triggers,
    since the loop checks after incrementing).
    """
    base = float(args.lr)
    every = int(getattr(args, "lr_half_every", 0) or 0)
    start = int(getattr(args, "lr_decay_start", 0) or 0)
    if not every or step < max(start, 1):
        return base
    first = ((max(start, 1) + every - 1) // every) * every
    if first > step:
        return base
    return base * (0.5 ** (1 + (step - first) // every))


def load_resume_weights(model, path: str, args) -> int:
    """Load a branch+router checkpoint for ``--resume``; returns start step.

    Verifies every checkpoint key exists in the model (rank/arch mismatch ->
    SystemExit); shape mismatches raise from ``load_state_dict`` and are
    re-reported as SystemExit.  Optimizer state is NOT stored (Adafactor
    restarts fresh — the standard caveat, noted in the runlog); the LR is
    recomputed by :func:`lr_at_step` at the call site.
    """
    try:
        sd = torch.load(path, map_location="cpu")
    except FileNotFoundError:
        raise SystemExit(f"--resume {path}: file not found")
    if not isinstance(sd, dict) or not sd:
        raise SystemExit(f"--resume {path}: not a branch checkpoint dict")
    have = set(model.state_dict())
    unknown = [k for k in sd if k not in have]
    if unknown:
        raise SystemExit(f"--resume {path}: {len(unknown)} keys match no "
                         f"model parameter (rank/arch mismatch?), e.g. "
                         f"{unknown[0]}")
    try:
        model.load_state_dict(sd, strict=False)
    except RuntimeError as exc:
        raise SystemExit(f"--resume {path}: shape mismatch ({exc})")
    n = sum(1 for k in sd if ".branch." in k or ".gate." in k)
    print(f"resumed {len(sd)} tensors ({n} branch/router) from {path}",
          flush=True)
    return parse_resume_step(path, int(getattr(args, "resume_step", 0) or 0))


@torch.no_grad()
def stage_eval(args):
    from transformers import AutoTokenizer
    model_dir = Path(args.model_dir) if args.model_dir else MODEL
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    data = windows(tok, args.eval_windows, args.seq, 999, "wikitext")

    patch_experts(args.group)
    if getattr(args, "prefix_layers", 0) and getattr(args, "fp8", True):
        # a prefix that includes a PLE layer needs the rows its eval windows
        # hash to (the dense table is ~95 GiB); load_full has no ids to pass.
        ple_ids = torch.stack([data[i] for i in range(len(data))])
        model, _, _, _ = load_fp8_prefix(
            args.prefix_layers, args.device, model_dir=model_dir,
            shard_dir=Path(args.shard_dir) if args.shard_dir else None,
            ple=args.ple, ple_ids=ple_ids)
    else:
        model, _ = load_full(args)
    build_student(model, args)
    if args.balance != "none":
        n_gates = patch_router_balance(model, args.balance,
                                       args.balance_cb_eta, args.balance_qb_damp)
        print(f"router balance: {args.balance} on {n_gates} gates", flush=True)
    if args.load:
        missing, unexpected = load_branch_state(model, args.load)
        print(f"loaded {args.load}: missing={len(missing)} "
              f"unexpected={len(unexpected)}", flush=True)
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
           "branch_quant": args.branch_quant, "tag": getattr(args, "tag", "")}
    OUT.mkdir(parents=True, exist_ok=True)
    name = (f"qwen4exp-eval-{args.tag}.json" if getattr(args, "tag", "")
            else "qwen4exp-eval.json")
    (OUT / name).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


@torch.no_grad()
def apply_ple_qat(model, bits: int = 0, group: int = 32,
                  force: bool = False) -> int:
    """Quantize PLE lookup outputs with a straight-through estimator (PLE-QAT).

    When ``bits > 0``, every PLE n-gram table's forward output is replaced by
    ``out + (dequant(quant(out)) - out.detach())``: the forward sees the
    release quantization (default 2-bit group-32 rows), the backward passes
    gradients through unchanged, so branches/routers adapt to the quantized
    table.  The table itself is untouched (frozen); only rows actually looked
    up are quantized, so the full-table cost never materializes.  With
    ``force=False`` (default) the hook is a no-op outside training mode, so
    monitoring evals stay full-precision; pass ``force=True`` to measure the
    adapted operating point explicitly.
    """
    if not bits:
        return 0
    n = 0
    try:
        layers = text_layers(model)
    except AttributeError:
        return 0
    for layer in layers:
        ple = getattr(layer, "ple", None)
        emb = getattr(ple, "ple_embedding", None)
        tab = getattr(emb, "ngram_embedding", None)
        if tab is None:
            continue

        def _qat_hook(mod, _args, output, _bits=bits, _group=group,
                      _force=force):
            if not _force and not mod.training:
                return output
            with torch.no_grad():
                deq, _ = quantize_rows(output.detach(), _bits, _group)
            return output + (deq - output.detach())

        tab.register_forward_hook(_qat_hook)
        n += 1
    return n


def alloc_mixed_bits(freq: torch.Tensor, spec: list[tuple[int, float]]
                     ) -> torch.Tensor:
    """Bits per row from a ``[(bits, fraction)]`` spec by corpus frequency.

    Rows are ranked by descending ``freq`` (ties by row order); the top
    ``fraction`` of rows gets ``bits``.  Fractions must sum to 1.  Pure,
    CPU-testable; the stage applies the resulting per-row widths.
    """
    total = sum(f for _, f in spec)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"mixed-precision fractions must sum to 1, got {total}")
    n = int(freq.numel())
    order = torch.argsort(freq, descending=True, stable=True)
    out = torch.empty(n, dtype=torch.int64)
    lo = 0
    for bits, frac in spec:
        hi = lo + int(round(frac * n))
        hi = min(hi, n)
        out[order[lo:hi]] = bits
        lo = hi
    out[order[lo:]] = spec[-1][0]  # rounding spill goes to the last bucket
    return out


def svd_compress(rows: torch.Tensor, rank: int) -> tuple[torch.Tensor, float]:
    """Rank-``rank`` SVD reconstruction of a row matrix + bytes/row (fp16)."""
    if rank <= 0 or rank >= min(rows.shape):
        raise ValueError(f"SVD rank {rank} invalid for {tuple(rows.shape)}")
    f = rows.float()
    u, s, vh = torch.linalg.svd(f, full_matrices=False)
    rec = (u[:, :rank] * s[:rank]) @ vh[:rank]
    n, d = rows.shape
    bpr = ((n * rank + rank + rank * d) * 2.0) / n
    return rec.to(rows.dtype), bpr


def quantize_mixed_rows(rows: torch.Tensor, bits: torch.Tensor, group: int = 32
                        ) -> tuple[torch.Tensor, float]:
    """Per-row-width re-quantisation for a mixed-precision allocation.

    ``bits`` (int64, one width per row, values in {2, 4, 8}) selects the
    ``quantize_rows`` codebook per row.  Returns ``(dequantized, mean
    bytes/row)``.
    """
    uniq = sorted(set(bits.tolist()))
    if any(b not in (2, 4, 8) for b in uniq):
        raise ValueError(f"mixed widths must be in {{2, 4, 8}}, got {uniq}")
    out = torch.empty_like(rows)
    tot = 0.0
    for b in uniq:
        m = bits == b
        dq, bpr = quantize_rows(rows[m], b, group)
        out[m] = dq.to(out.dtype)
        tot += bpr * int(m.sum())
    return out, tot / rows.shape[0]


def parse_mixed_spec(spec: str) -> list[tuple[int, float]]:
    """``"8:0.1,4:0.6,2:0.3"`` -> ``[(8, 0.1), (4, 0.6), (2, 0.3)]``."""
    out = []
    for part in spec.split(","):
        b, f = part.split(":")
        out.append((int(b), float(f)))
    return out


def stage_ple_ab(args):
    """PLE precision A/B: re-quantise only the gathered n-gram rows.

    The prefix (fp8-dequantized, no ternary, no branches) is the reference; for
    each ``--ple-bits`` value the row table is quantized symmetrically and the
    full-vocab KLD of the resulting distribution against the reference is
    reported.  This prices the release container's n-gram precision, which the
    recipe does not otherwise cover.
    """
    from kld_eval import log_probs, kld_from_logprobs, top1_agreement
    from transformers import AutoTokenizer

    model_dir = Path(args.model_dir) if args.model_dir else MODEL
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    data = windows(tok, args.eval_windows, args.seq, args.seed, args.split)
    ple_ids = torch.stack([data[i] for i in range(len(data))])
    freq_data = None
    if args.ple_freq_windows > 0:
        freq_data = windows(tok, args.ple_freq_windows, args.seq,
                            args.seed + 7919, args.split)
        fall = torch.stack([freq_data[i] for i in range(len(freq_data))])
        ple_ids = torch.cat([ple_ids, fall])

    model, _, _, plan = load_fp8_prefix(
        args.layers, args.device, model_dir=model_dir,
        shard_dir=Path(args.shard_dir) if args.shard_dir else None,
        ple="rows", ple_ids=ple_ids)
    if not plan["ple_layers"]:
        raise SystemExit("the prefix has no PLE layer; raise --layers past "
                         f"{min(_ple_layer_ids(_q4_text_config(model_dir))) + 1}")
    tables = [(model.layers[i].ple.ple_embedding.ngram_embedding, i)
              for i in plan["ple_layers"]]
    embs = [model.layers[i].ple.ple_embedding for i in plan["ple_layers"]]
    base_rows = [(tab.ids_sorted.clone(), tab.rows.clone()) for tab, _ in tables]

    def run():
        logps = []
        with torch.no_grad():
            for i in range(len(data)):
                ids = data[i:i + 1].to(args.device)
                logps.append(log_probs(model_logits(model, ids)[0, :-1], args.chunk).cpu())
        return logps

    def entropy(logps):
        hs = [float((-(p.exp() * p)).sum(-1).mean()) for p in logps]
        return sum(hs) / len(hs)

    def count_freq():
        """Row-hit counts over eval + freq windows, per table (ids_sorted order)."""
        counts = []
        with torch.no_grad():
            for (tab, _), emb in zip(tables, embs):
                tab.start_recording()
                for ids in ple_ids:
                    emb(ids.unsqueeze(0).to(args.device), None)
                seen = torch.cat(tab._seen).long().cpu()
                tab.stop_recording()
                pos = torch.searchsorted(tab.ids_sorted.cpu(), seen.clamp(min=0))
                pos_c = pos.clamp(max=tab.n_rows() - 1)
                hit = (tab.ids_sorted.cpu()[pos_c] == seen)
                c = torch.zeros(tab.n_rows(), dtype=torch.int64)
                c.index_add_(0, pos_c[hit],
                             torch.ones(int(hit.sum()), dtype=torch.int64))
                counts.append(c)
        return counts

    ref = run()
    ref_H = entropy(ref)
    print(f"PLE A/B: {args.layers} layers, {len(data)} windows, "
          f"{total_rows(tables)} rows, group {args.ple_group}, "
          f"ref entropy {ref_H:.4f}", flush=True)
    out = {"layers": args.layers, "eval_windows": args.eval_windows,
           "freq_windows": args.ple_freq_windows, "rows": total_rows(tables),
           "group": args.ple_group, "ref_entropy": ref_H, "arms": {}}

    def score(name, bpr):
        cur = run()
        klds = [kld_from_logprobs(ref[i], cur[i], args.chunk) for i in range(len(data))]
        per_token = torch.cat(klds)
        agree = sum(top1_agreement(ref[i], cur[i]) for i in range(len(data))) / len(data)
        arm = {"mean_kld": float(per_token.mean()),
               "p99": float(torch.quantile(per_token, 0.99)),
               "max": float(per_token.max()),
               "top1_agreement": float(agree),
               "entropy": entropy(cur),
               "bytes_per_row": float(bpr) if not isinstance(bpr, str) else bpr}
        out["arms"][name] = arm
        print(f"PLE {name}: mean {arm['mean_kld']:.4f} "
              f"p99 {arm['p99']:.4f} max {arm['max']:.4f} "
              f"top1 {agree:.4f} H {arm['entropy']:.4f} "
              f"bytes/row {arm['bytes_per_row']}", flush=True)

    for bits in [int(b) for b in str(args.ple_bits).split(",")]:
        for k, (tab, _) in enumerate(tables):
            ids_sorted, rows = base_rows[k]
            deq, bpr = quantize_rows(rows, bits, args.ple_group)
            tab.set_rows(ids_sorted, deq)
        score(f"{bits}bit", bpr)
    if args.ple_mixed:
        freq = count_freq()
        for spec in str(args.ple_mixed).split(";"):
            pairs = parse_mixed_spec(spec)
            bprs = []
            for k, (tab, _) in enumerate(tables):
                ids_sorted, rows = base_rows[k]
                bits = alloc_mixed_bits(freq[k], pairs)
                deq, bpr = quantize_mixed_rows(rows, bits, args.ple_group)
                tab.set_rows(ids_sorted, deq)
                bprs.append(bpr)
            score(f"mixed[{spec}]",
                  sum(bprs) / len(bprs) if bprs else 0.0)
    if args.ple_off:
        for k, (tab, _) in enumerate(tables):
            ids_sorted, rows = base_rows[k]
            tab.set_rows(ids_sorted, torch.zeros_like(rows))
        score("off", 0.0)
    if args.ple_svd:
        for rank in [int(r) for r in str(args.ple_svd).split(",")]:
            bprs = []
            for k, (tab, _) in enumerate(tables):
                ids_sorted, rows = base_rows[k]
                rec, bpr = svd_compress(rows.cpu(), rank)
                tab.set_rows(ids_sorted, rec.to(rows.device))
                bprs.append(bpr)
            score(f"svd{rank}", sum(bprs) / len(bprs) if bprs else 0.0)
    for k, (tab, _) in enumerate(tables):       # restore full precision
        tab.set_rows(*base_rows[k])
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}", flush=True)


def total_rows(tables) -> int:
    return sum(tab.n_rows() for tab, _ in tables)


@torch.no_grad()
def stage_preflight(args):
    """Pod pre-flight: environment + loader + one forward + cache dry-run."""
    import importlib
    import platform
    report = {"python": platform.python_version()}
    import torch as _t
    report["torch"] = _t.__version__
    report["cuda_available"] = bool(_t.cuda.is_available())
    if _t.cuda.is_available():
        report["gpu"] = _t.cuda.get_device_name(0)
        report["capability"] = list(_t.cuda.get_device_capability(0))
    import transformers
    report["transformers"] = transformers.__version__
    report["qwen4_exp"] = bool(importlib.util.find_spec(
        "transformers.models.qwen4_exp"))
    for mod in ("kernels", "fla", "causal_conv1d"):
        report[mod] = bool(importlib.util.find_spec(mod))
    model_dir = Path(args.model_dir) if args.model_dir else MODEL
    report["model_dir"] = str(model_dir)
    report["has_index"] = (model_dir / "model.safetensors.index.json").exists()
    if report["has_index"]:
        weight_map = _load_index(model_dir)
        all_shards = sorted(set(weight_map.values()))
        shard_dir = Path(args.shard_dir) if args.shard_dir else (
            model_dir / PLE_SHARD_DIR if (model_dir / PLE_SHARD_DIR).is_dir()
            else model_dir)
        have = [s for s in all_shards if (shard_dir / s).exists()]
        report["shards_total"] = len(all_shards)
        report["shards_present"] = len(have)
        report["bytes_present"] = int(sum((shard_dir / s).stat().st_size for s in have))
        plan = fp8_prefix_plan(args.layers, model_dir, include_ple=args.ple != "none")
        report["prefix_shards_needed"] = plan["shards"]
        report["ple_layers_in_range"] = plan["ple_layers"]
    ok = True

    # loader + one forward on the prefix
    from transformers import AutoTokenizer
    import time
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    data = windows(tok, 1, min(args.seq, 128), 999, "wikitext")
    ple_ids = torch.stack([data[i] for i in range(len(data))])
    t_start = time.time()
    model, missing, unexpected, plan = load_fp8_prefix(
        args.layers, args.device, model_dir=model_dir,
        shard_dir=Path(args.shard_dir) if args.shard_dir else None,
        ple=args.ple, ple_ids=ple_ids)
    load_s = time.time() - t_start
    ids = data[0:1].to(args.device)
    t_start = time.time()
    logits = model_logits(model, ids)
    fwd_s = time.time() - t_start
    report["prefix_load_s"] = round(load_s, 1)
    report["forward_s"] = round(fwd_s, 2)
    report["forward_logits"] = list(logits.shape)
    report["missing_tensors"] = len(missing)
    report["unexpected_tensors"] = len(unexpected)
    if missing or unexpected:
        print(f"WARNING: missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}",
              flush=True)
        ok = False

    # cache dry-run
    class A:
        top_logits = args.top_logits
        tail_logits = args.tail_logits
    rec = make_record(logits[:, :-1], A)
    report["cache_record_keys"] = sorted(rec.keys())
    report["cache_record_shapes"] = {k: list(v.shape) for k, v in rec.items()
                                     if torch.is_tensor(v)}
    need = {"idx", "val", "w"}
    if args.tail_logits:
        need |= {"tidx", "tlp"}
    if not need.issubset(rec):
        print(f"FAIL: cache record missing {need - set(rec)}", flush=True)
        ok = False

    if args.native:
        # the cache stage's loader: official from_pretrained with native fp8
        from transformers import AutoModelForImageTextToText
        mm = None
        if args.max_memory:
            mm = {int(k): v for k, v in
                  (part.split(":") for part in args.max_memory.split(","))}
        t_start = time.time()
        full = AutoModelForImageTextToText.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, device_map=args.device_map,
            max_memory=mm)
        report["native_load_s"] = round(time.time() - t_start, 1)
        banks = []
        for layer in text_layers(full):
            banks.append(str(layer.mlp.experts.gate_up_proj.dtype))
        report["native_expert_dtype"] = banks[0]
        report["native_fp8_kept"] = banks[0].startswith("torch.float8")
        t_start = time.time()
        out = full(input_ids=ids, use_cache=False)
        report["native_forward_s"] = round(time.time() - t_start, 2)
        logits = getattr(out, "logits", None)
        if logits is None:
            logits = full.lm_head(out.last_hidden_state)
        report["native_logits"] = list(logits.shape)
        del full
        gc.collect()
        torch.cuda.empty_cache()

    report["ok"] = ok
    Path(args.out or (OUT / "preflight.json")).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out or (OUT / "preflight.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print("PREFLIGHT OK" if ok else "PREFLIGHT FAILED", flush=True)
    if not ok:
        raise SystemExit(1)


# ------------------------------------------------------------------- CLI -----

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "cache", "ref", "train", "eval",
                                      "ple-ab", "preflight", "compact-check",
                                      "qat-screen"])
    ap.add_argument("--model-dir", default="",
                    help="checkpoint dir (default $Q4_MODEL or the local mirror)")
    ap.add_argument("--shard-dir", default="",
                    help="where model-*.safetensors live (default <model-dir>/shards)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--max-memory", default="", help="e.g. '0:70GiB,1:70GiB'")
    ap.add_argument("--fp8", action="store_true", default=True,
                    help="read the official fp8 shards (default); --no-fp8 uses a "
                         "converted bf16 mirror")
    ap.add_argument("--no-fp8", dest="fp8", action="store_false")
    ap.add_argument("--ple", choices=["rows", "none"], default="rows",
                    help="PLE handling for the fp8 prefix loader: rows = gather "
                         "only the used rows (needed when a PLE layer is in range)")
    ap.add_argument("--layers", type=int, default=2, help="smoke prefix depth")
    ap.add_argument("--tiny", action="store_true",
                    help="smoke: use the random-init tiny model (no checkpoint); "
                         "the plumbing rig for the port's own tests")
    ap.add_argument("--prefix-layers", type=int, default=0,
                    help="run full stages on an N-layer prefix (local dry runs)")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--quant", choices=["absmean", "lloyd"], default="lloyd",
                    help="per-group scale rule for the frozen banks (lloyd = "
                         "deployable)")
    ap.add_argument("--compact-banks", action="store_true",
                    help="P2: store the frozen banks in deployed form (2-bit "
                         "codes + fp16 scale per group) and decode per-expert in "
                         "the training forward; default off (frozen 35B/eval "
                         "paths untouched)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="train: checkpoint the decoder layers so the per-expert "
                         "decode is recomputed in backward (bounds the saved "
                         "decoded-weight memory; exact)")
    ap.add_argument("--offload-saved", action="store_true",
                    help="train: keep autograd's saved tensors in host RAM "
                         "(pack on save / unpack on use) instead of "
                         "checkpointing the decoder layers; the 48 GB card "
                         "fits because the pod has 188 GB of host RAM")
    ap.add_argument("--checkpoint-mode", choices=["block", "expert", "group"],
                    default="block",
                    help="granularity of the compact-decode checkpoint "
                         "(block = one node per sparse block (default); expert "
                         "= one per expert; group = one per expert-group, "
                         "bounds the decoded-weight saves at ~group x 9.8 MB)")
    ap.add_argument("--expert-group-size", type=int, default=32,
                    help="train: experts per checkpointed group for "
                         "--checkpoint-mode group (32 saves ~0.3 GiB)")
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--branch-quant", choices=["fp32", "g128", "rank"],
                    default="fp32")
    ap.add_argument("--branch-target", choices=["moe_out", "attn_out", "both"],
                    default="both",
                    help="moe_out (block output, pre-injection), attn_out "
                         "(out_proj/o_proj), or both -- the 35B recipe placement")
    ap.add_argument("--branch-gate", choices=["none", "rw"], default="none",
                    help="Phase C residual-stream gates (identity at init)")
    ap.add_argument("--windows", type=int, default=4096)
    ap.add_argument("--corpus-chars", type=int, default=50_000_000)
    ap.add_argument("--corpus-file", default="")
    ap.add_argument("--agentic-frac", type=float, default=0.05,
                    help="fraction of rows drawn from --corpus-file (cur05: 0.05)")
    ap.add_argument("--cache-file", default="")
    ap.add_argument("--save-every", type=int, default=500,
                    help="cache stage: park a .partial file every N windows "
                         "(0 = off); an interrupt costs minutes, not the pass")
    ap.add_argument("--resume-cache", action="store_true",
                    help="cache stage: resume from the .partial file if present "
                         "(windows must be the same corpus/seed)")
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="fineweb")
    ap.add_argument("--top-logits", type=int, default=50)
    ap.add_argument("--tail-logits", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-half-every", type=int, default=0)
    ap.add_argument("--lr-decay-start", type=int, default=0)
    ap.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor")
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--kd-weight", type=float, default=2.0)
    ap.add_argument("--kd-filter-frac", type=float, default=0.0)
    ap.add_argument("--kd-support-w", choices=["one", "wt"], default="one")
    ap.add_argument("--kd-tail-weight", type=float, default=2.0,
                    help="residual-mass (marginal) KL (cur05 recipe: 2.0)")
    ap.add_argument("--kd-tailcond-weight", type=float, default=3.0,
                    help="TAD D_KL2 weight (cur05 recipe: 3.0)")
    ap.add_argument("--balance", choices=["none", "bias", "quantile", "zloss",
                                          "cb", "cbqb"], default="none")
    ap.add_argument("--balance-delta", type=float, default=1e-3)
    ap.add_argument("--balance-cb-eta", type=float, default=0.05)
    ap.add_argument("--balance-qb-damp", type=float, default=1.0)
    ap.add_argument("--balance-z-coeff", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--ref-file", default="")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-entropy", action="store_true")
    ap.add_argument("--load", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", default="",
                    help="train: branch+router checkpoint to resume from "
                         "(e.g. a step-3000 ckpt); training continues at its "
                         "step with the cache offset to match")
    ap.add_argument("--resume-step", type=int, default=0,
                    help="train: explicit start step (default: step<N> parsed "
                         "from the --resume filename)")
    ap.add_argument("--ple-bits", default="8,4,2",
                    help="ple-ab: comma-separated bit widths to price")
    ap.add_argument("--ple-group", type=int, default=32,
                    help="ple-ab: quantization group along the row (160 must "
                         "divide it; 32 = 5 groups)")
    ap.add_argument("--ple-mixed", default="",
                    help="ple-ab: mixed-precision spec(s), e.g. "
                         "\"8:0.1,4:0.6,2:0.3\" (bits:fraction by corpus "
                         "frequency); ';'-separated for several")
    ap.add_argument("--ple-off", action="store_true",
                    help="ple-ab: add a zero-table (PLE-off) ablation arm")
    ap.add_argument("--ple-svd", default="",
                    help="ple-ab: comma-separated SVD ranks, e.g. \"16,32,64\"")
    ap.add_argument("--ple-freq-windows", type=int, default=0,
                    help="ple-ab: extra windows (past --eval-windows) whose "
                         "row hits drive the mixed-precision frequencies")
    ap.add_argument("--ple-qat-bits", type=int, default=0,
                    help="train + qat-screen: PLE-QAT width (0 = off); the "
                         "student forward sees release-quantized PLE rows "
                         "with STE gradients so branches adapt")
    ap.add_argument("--ple-qat-group", type=int, default=32,
                    help="train + qat-screen: PLE-QAT group along the row")
    ap.add_argument("--qat-steps", type=int, default=30,
                    help="qat-screen: LM training steps per arm")
    ap.add_argument("--qat-eval-windows", type=int, default=8,
                    help="qat-screen: eval windows for the 2-bit damage read")
    ap.add_argument("--qat-train-windows", type=int, default=8,
                    help="qat-screen: distinct train windows (seed 1001)")
    ap.add_argument("--qat-eval-same", action="store_true",
                    help="qat-screen: eval on the train windows (in-sample "
                         "damage; tests whether QAT adapts covered rows)")
    ap.add_argument("--native", action="store_true",
                    help="preflight: also load the full checkpoint through the "
                         "official from_pretrained route (native fp8, the cache "
                         "stage's loader) and forward once -- pod only")
    ap.add_argument("--force-gpu", action="store_true",
                    help="after from_pretrained(auto), move the offloaded "
                         "blocks to the GPU (hooks removed; the PLE stays "
                         "host-side) -- the per-forward HtoD copies of "
                         "offloaded layers dominate the window time")
    ap.add_argument("--fast-indexer", action="store_true",
                    help="route the QSA indexer through the batched pure-causal "
                         "path (identical masks in the official all-blocks "
                         "regime; the reference's per-query Python loop is the "
                         "cache's per-window bottleneck)")
    ap.add_argument("--chunk", type=int, default=32, help="log_softmax token chunk")
    ap.add_argument("--out", default="")
    return ap


def main():
    args = build_parser().parse_args()
    if args.stage == "smoke":
        stage_smoke(args)
    elif args.stage == "cache":
        stage_cache(args)
    elif args.stage == "ref":
        stage_ref(args)
    elif args.stage == "train":
        stage_train(args)
    elif args.stage == "eval":
        stage_eval(args)
    elif args.stage == "ple-ab":
        stage_ple_ab(args)
    elif args.stage == "compact-check":
        stage_compact_check(args)
    elif args.stage == "qat-screen":
        stage_qat_screen(args)
    else:
        stage_preflight(args)


if __name__ == "__main__":
    main()

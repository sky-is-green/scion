"""The Clef joint schema head + lm_head sidecar for the dense route.

The backbone is quantised and served as a GGUF; the joint head and ``lm_head``
stay in torch (bf16).  This module loads both, encodes a SystemOne-style
request into the exact token order the head expects, and runs the head over a
backbone's post-norm hidden states to get per-option logits.

Used by the teacher cache (to label decision targets) and by the correction
trainer (decision KD on the student's states).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

from clef_paths import CLEF_MODEL

DEFAULT_MODEL_DIR = CLEF_MODEL
DEFAULT_INSTRUCTION = "The candidate answer is correct and complete for the task."


def _import_joint_schema(model_dir: Path):
    head_dir = model_dir / "hf-head"
    if str(head_dir) not in sys.path:
        sys.path.insert(0, str(head_dir))
    import joint_schema_model as js  # type: ignore
    return js


def load_joint_head(model_dir: str | Path = DEFAULT_MODEL_DIR, *,
                    device="cpu", dtype=torch.bfloat16):
    from safetensors.torch import load_file
    model_dir = Path(model_dir)
    js = _import_joint_schema(model_dir)
    cfg = json.loads((model_dir / "hf-head" / "joint_head_config.json").read_text())
    head = js.JointSchemaHead(**cfg)
    head.load_state_dict(load_file(model_dir / "hf-head" / "joint_head.safetensors"), strict=True)
    head = head.to(device=device, dtype=dtype).eval()
    return head, js


def load_lm_head(model_dir: str | Path = DEFAULT_MODEL_DIR, *,
                 device="cpu", dtype=torch.bfloat16) -> torch.Tensor:
    from safetensors.torch import load_file
    model_dir = Path(model_dir)
    weights = load_file(model_dir / "lm_head.safetensors")
    key = "lm_head.weight" if "lm_head.weight" in weights else next(iter(weights))
    return weights[key].to(device=device, dtype=dtype)


def load_tokenizer(model_dir: str | Path = DEFAULT_MODEL_DIR):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(Path(model_dir) / "hf-head"))


def build_request(prompt: str, candidate: str, instruction: str = DEFAULT_INSTRUCTION) -> dict:
    """A SystemOne request with one ``noul`` verdict question."""
    return {
        "model": "clef-flash",
        "state": {"task": prompt, "candidate_answer": candidate},
        "questions": {"verdict": {"type": "noul", "instructions": instruction}},
    }


def encode(tokenizer, prompt: str, candidate: str, *,
           instruction: str = DEFAULT_INSTRUCTION, max_length: int = 16384):
    """Encode one (prompt, candidate) pair to the head's exact input order."""
    head_dir = Path(getattr(tokenizer, "name_or_path", DEFAULT_MODEL_DIR / "hf-head"))
    if str(head_dir) not in sys.path:
        sys.path.insert(0, str(head_dir))
    import joint_schema_model as js  # type: ignore
    return js.encode_record(tokenizer, build_request(prompt, candidate, instruction),
                            max_length=max_length)


def head_logits(head, hidden_states: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor, records: list, lm_head_weight: torch.Tensor):
    """Run the head; returns ``list[list[Tensor]]`` (per record, per question)."""
    return head(hidden_states, input_ids, attention_mask, records, lm_head_weight)


def noul_prob(head, hidden_states: torch.Tensor, input_ids: torch.Tensor,
              attention_mask: torch.Tensor, records: list, lm_head_weight: torch.Tensor,
              record_index: int = 0) -> float:
    """P(true) for the first (noul) question of one record."""
    with torch.no_grad():
        logits = head_logits(head, hidden_states, input_ids, attention_mask,
                             records, lm_head_weight)[record_index][0]
        return float(logits.float().softmax(-1)[0])

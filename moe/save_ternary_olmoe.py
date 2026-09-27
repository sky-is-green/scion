"""Materialise the RTN-ternary OLMoE to a HF dir for tooling (e.g. AUTOGRID scan).

Loads on CPU, quantises the expert banks in place (same code path as the
training builds), and saves a standard safetensors checkpoint plus config and
tokenizer.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from olmoe_corrections import quantize_bank_inplace  # noqa: E402
from olmoe_proxy import load_model  # noqa: E402

ART = Path(os.environ.get("MOE_ARTIFACTS", HERE / "artifacts"))
OUT = ART / "olmoe-ternary-hf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=128)
    args = ap.parse_args()

    model, tok = load_model("cpu")
    for layer in model.model.layers:
        quantize_bank_inplace(layer.mlp.experts.gate_up_proj, args.group)
        quantize_bank_inplace(layer.mlp.experts.down_proj, args.group)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)
    print(f"saved ternary OLMoE to {OUT}")


if __name__ == "__main__":
    main()

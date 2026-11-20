"""Extract an embedded correction adapter from a full Scion model GGUF.

The v1/v2 bodies carry their adapter as ``*.lora_a``/``*.lora_b`` tensors plus
``adapter.*`` metadata.  ``merge_adapter_into_body.py`` wants a standalone
adapter file, so this writes exactly those tensors and keys.

Usage:
  PYTHONPATH=.../gguf-py python v2/hotcold/extract_adapter.py \
    --src .../Scion-35B-A3B-Q2_0_G128-corr-v1.gguf \
    --out /tmp/opencode/v1-adapter.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scion-llama.cpp" / "gguf-py"))

from gguf import GGUFReader, GGUFWriter, GGUFValueType, Keys  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = GGUFReader(args.src)
    arch = src.fields[Keys.General.ARCHITECTURE].contents()
    tensors = [t for t in src.tensors
               if t.name.endswith(".lora_a") or t.name.endswith(".lora_b")]
    if not tensors:
        raise SystemExit("no .lora_a/.lora_b tensors in source")
    if "adapter.recipe" not in src.fields:
        raise SystemExit("source has no adapter.recipe metadata")

    writer = GGUFWriter(args.out, arch=arch)
    for field in src.fields.values():
        if not field.name.startswith("adapter."):
            continue
        if field.name == "adapter.embedded":
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type,
                             sub_type=sub_type)
    for t in tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes,
                               t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in tensors:
        writer.write_tensor_data(t.data)
    writer.close()
    print(f"wrote {args.out}: {len(tensors)} tensors")


if __name__ == "__main__":
    main()

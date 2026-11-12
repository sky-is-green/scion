"""Merge a correction adapter GGUF into the body GGUF -> single-file release.

Produces a model file that carries the adapter's ``.lora_a``/``.lora_b`` tensors
and the ``adapter.*`` metadata, plus ``adapter.embedded = true``.  The
PrismML/TAARDIS fork detects that flag at load and attaches the adapter from the
model file itself (``llama_adapter_lora_init_embedded``), so every context
applies the corrections with no ``--lora`` argument.  The strata engine's
``src/core/corrections.cpp`` loads the same embedded tensors.

Nothing is re-quantized: body and adapter tensors are copied byte-for-byte from
their source files, so the single file must reproduce the two-file runtime PPL
exactly.

``--replace`` (the v2 release semantic) drops body tensors that collide with
the adapter's: v2 branches train from zero, so the v1 factors at the same
sites must not survive into the merged file (they would apply twice).  The
body's ``adapter.*`` metadata is skipped too, so the file carries the new
adapter's provenance.  Body ``adapter.embedded`` stays true either way.

Usage:
  PYTHONPATH=<fork>/gguf-py python merge_adapter_into_body.py \
      --body qwen35-body-pq2_0-q8rest.gguf \
      --adapter qwen35-adapter.lora-soup.gguf \
      --out qwen35-release.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm
from gguf import GGUFReader, GGUFWriter, GGUFValueType, Keys


def copy_body_metadata(reader: GGUFReader, writer: GGUFWriter, skip_general: bool) -> None:
    for field in reader.fields.values():
        if field.name.startswith('GGUF.'):
            continue
        if field.name == Keys.General.ARCHITECTURE:
            # GGUFWriter writes this from the arch argument
            continue
        if skip_general and field.name.startswith('general.'):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--body", required=True, type=Path, help="deployment body GGUF")
    ap.add_argument("--adapter", required=True, type=Path, help="correction adapter GGUF")
    ap.add_argument("--out", required=True, type=Path, help="output single-file GGUF")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    ap.add_argument("--replace", action="store_true",
                    help="the adapter SUPERSEDES body corrections at the same "
                         "sites: body tensors whose names collide with the "
                         "adapter's are dropped (adapter wins) and body "
                         "adapter.* metadata is skipped so the file carries "
                         "the new adapter's provenance.  This is the v2 "
                         "release semantic: v2 branches train from zero, so "
                         "keeping the v1 factors would apply both.  Default "
                         "off (strict collision refusal, the v1 behaviour)")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        print(f"error: {args.out} exists (use --force)", file=sys.stderr)
        return 2

    body = GGUFReader(args.body)
    adapter = GGUFReader(args.adapter)

    body_arch = body.fields[Keys.General.ARCHITECTURE].contents()
    adapter_arch = adapter.fields[Keys.General.ARCHITECTURE].contents()
    if body_arch != adapter_arch:
        print(f"error: arch mismatch: body {body_arch}, adapter {adapter_arch}", file=sys.stderr)
        return 2
    if "adapter.recipe" not in adapter.fields:
        print("error: adapter has no adapter.recipe metadata; is this a TAARDIS export?", file=sys.stderr)
        return 2

    body_names = {t.name for t in body.tensors}
    adapter_names = {t.name for t in adapter.tensors}
    dupes = body_names & adapter_names
    if dupes and not args.replace:
        print(f"error: {len(dupes)} tensor name(s) collide, e.g. {sorted(dupes)[:3]}", file=sys.stderr)
        return 2
    if dupes:
        print(f"replace: {len(dupes)} body tensor(s) superseded by the adapter, "
              f"e.g. {sorted(dupes)[:3]}")

    writer = GGUFWriter(args.out, arch=body_arch)

    # body metadata (its general.* wins), then the adapter's adapter.* keys.
    # With --replace the body's own adapter.* keys are skipped: a replacement
    # adapter must not leave the previous adapter's provenance in the file.
    for field in body.fields.values():
        if field.name.startswith('GGUF.'):
            continue
        if field.name == Keys.General.ARCHITECTURE:
            continue
        if args.replace and field.name.startswith("adapter."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    copy_body_metadata(adapter, writer, skip_general=True)
    writer.add_key_value("adapter.embedded", True, GGUFValueType.BOOL)

    # tensor infos: body first, then adapter; with --replace the collided
    # body tensors are dropped entirely (the adapter's factors take the site)
    body_tensors = [t for t in body.tensors
                    if not (args.replace and t.name in adapter_names)]
    for t in body_tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    for t in adapter.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    total_bytes = sum(t.n_bytes for t in body_tensors) + sum(t.n_bytes for t in adapter.tensors)
    print(f"body:    {len(body_tensors):5d} tensors, {sum(t.n_bytes for t in body_tensors)/2**30:.3f} GiB")
    print(f"adapter: {len(adapter.tensors):5d} tensors, {sum(t.n_bytes for t in adapter.tensors)/2**30:.3f} GiB")
    print(f"out:     {len(body_tensors) + len(adapter.tensors):5d} tensors, {total_bytes/2**30:.3f} GiB")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)
    for t in body_tensors:
        writer.write_tensor_data(t.data, tensor_endianess=body.endianess)
        bar.update(t.n_bytes)
    for t in adapter.tensors:
        writer.write_tensor_data(t.data, tensor_endianess=adapter.endianess)
        bar.update(t.n_bytes)
    writer.close()
    bar.close()

    # sanity: reopen and verify the tensor set and sizes
    check = GGUFReader(args.out)
    check_names = {t.name for t in check.tensors}
    expected = {t.name for t in body_tensors} | adapter_names
    missing = expected - check_names
    if missing:
        print(f"error: output is missing {len(missing)} tensors, e.g. {sorted(missing)[:3]}", file=sys.stderr)
        return 1
    if check.fields.get("adapter.embedded") is None:
        print("error: output has no adapter.embedded flag", file=sys.stderr)
        return 1

    print(f"wrote {args.out} ({args.out.stat().st_size/2**30:.3f} GiB): "
          f"{len(check.tensors)} tensors, adapter.embedded=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

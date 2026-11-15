"""Export a trained MTP head to the sidecar GGUF the runtime will load.

The sidecar carries only what the runtime cannot get from the target itself:

  mtp.fc1.weight  [width, 2*hidden]   (GGUF ne = [2*hidden, width])
  mtp.fc2.weight  [hidden, width]
  metadata: format/version/hidden/width/source + optional acceptance numbers

The head's output goes through the *target's own* output_norm + output
tensors, and its inputs are the target's post-norm hidden + the token
embedding, so those are deliberately NOT duplicated here.

Verification (`--verify`) reloads the GGUF, reconstructs the head and checks a
forward against the source checkpoint on random inputs -- the round-trip
guard for the dim convention.

Usage:
  python moe/mtp_sidecar_export.py --head $MOE/qwen35/mtp-release-head-v2.pt \
    --out $MOE/qwen35/mtp-sidecar-v2.gguf --source qwen35-release \
    --accept-fineweb 0.483 --accept-wikitext 0.360 --verify
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from scion_paths import GGUF_PY


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="checkpoint with _mtp_head.*")
    ap.add_argument("--out", required=True, help="sidecar .gguf to write")
    ap.add_argument("--gguf-py", default=str(GGUF_PY))
    ap.add_argument("--target-gguf", default="",
                    help="the target model: its tokenizer section is copied into the "
                         "sidecar (the draft ctx needs a vocab for the sampler chain) "
                         "and its context length is inherited")
    ap.add_argument("--source", default="", help="release/model id for metadata")
    ap.add_argument("--accept-fineweb", type=float, default=None)
    ap.add_argument("--accept-wikitext", type=float, default=None)
    ap.add_argument("--verify", action="store_true",
                    help="reload the sidecar and check a forward vs the source")
    return ap


def head_forward(fc1, fc2, h, e):
    """Mirror of mtp.MTPHead (rms inputs, gelu, no norm/head here)."""
    def rms(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    x = torch.cat([rms(h), rms(e)], dim=-1)
    return (fc2 @ torch.nn.functional.gelu(fc1 @ x.T)).T


def copy_vocab(w, target_gguf: str) -> int:
    """Copy the target's tokenizer section (+ context length) into the sidecar:
    the draft context builds its sampler chain against the draft model's vocab,
    so the sidecar must ship one.  The section is a few MB.

    ReaderField.data holds INDEXES into .parts (for strings, into byte parts),
    so every value is re-materialized from .parts before it is re-added."""
    from gguf import GGUFReader, GGUFValueType
    r = GGUFReader(target_gguf)
    copied = 0
    for f in r.fields.values():
        name = f.name
        if not (name.startswith("tokenizer.") or name == "general.context_length"):
            continue
        if name == "tokenizer.ggml.token_type":
            # the fork's target GGUF carries a token_type array the C reader
            # cannot see (the target loads with all-NORMAL attrs), so the
            # sidecar must match: copying it would activate the BYTE attr on
            # short tokens (" is type 6, 1 char) and crash token_to_byte
            continue
        typ = f.types[0] if f.types else GGUFValueType.STRING
        if typ == GGUFValueType.STRING:
            w.add_string(name, "".join(bytes(f.parts[i]).decode("utf-8") for i in f.data))
        elif typ == GGUFValueType.ARRAY:
            sub = f.types[-1] if len(f.types) > 1 else GGUFValueType.STRING
            if sub == GGUFValueType.STRING:
                w.add_array(name, [bytes(f.parts[i]).decode("utf-8") for i in f.data])
            else:
                w.add_array(name, [int(x) for x in f.data])
        elif typ == GGUFValueType.INT32:
            w.add_int32(name, int(f.data[0]))
        elif typ == GGUFValueType.UINT32:
            w.add_uint32(name, int(f.data[0]))
        elif typ == GGUFValueType.FLOAT32:
            w.add_float32(name, float(f.data[0]))
        elif typ == GGUFValueType.INT64:
            w.add_int64(name, int(f.data[0]))
        elif typ == GGUFValueType.UINT64:
            w.add_uint64(name, int(f.data[0]))
        elif typ == GGUFValueType.BOOL:
            w.add_bool(name, bool(f.data[0]))
        else:
            continue
        copied += 1
    return copied


def main() -> None:
    args = build_parser().parse_args()
    import sys
    sys.path.insert(0, args.gguf_py)
    from gguf import GGUFWriter

    sd = torch.load(args.head, map_location="cpu")
    fc1 = sd["_mtp_head.fc1.weight"].float().numpy()
    fc2 = sd["_mtp_head.fc2.weight"].float().numpy()
    fc3 = sd["_mtp_head.fc3.weight"].float().numpy() if "_mtp_head.fc3.weight" in sd else None
    width, two_h = fc1.shape
    hidden = fc2.shape[0]
    if fc2.shape[1] != width or two_h != 2 * hidden:
        raise SystemExit(f"head shape mismatch: fc1 {fc1.shape}, fc2 {fc2.shape}")
    if fc3 is not None and (fc3.shape[0] != hidden or fc3.shape[1] != width):
        raise SystemExit(f"state head shape mismatch: fc3 {fc3.shape}")

    w = GGUFWriter(args.out, arch="mtp-sidecar")
    w.add_string("general.name", args.source or "scion-mtp-sidecar")
    # the loader requires these; block_count=1 is a dummy trunk (the class loads
    # no layers), embedding_length = the head input's width (the target's n_embd)
    w.add_uint32(f"{w.arch}.block_count", 1)
    w.add_uint32(f"{w.arch}.embedding_length", int(hidden))
    # the sidecar's head IS the one MTP layer (the context gate wants n_layer_nextn > 0)
    w.add_uint32(f"{w.arch}.nextn_predict_layers", 1)
    w.add_float32(f"{w.arch}.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_string("mtp.format", "scion-mtp-head")
    w.add_uint32("mtp.version", 2)
    w.add_uint32("mtp.hidden_size", int(hidden))
    w.add_uint32("mtp.width", int(width))
    w.add_uint32("mtp.state_head", 1 if fc3 is not None else 0)
    if args.source:
        w.add_string("mtp.source", args.source)
    if args.accept_fineweb is not None:
        w.add_float32("mtp.acceptance.fineweb", float(args.accept_fineweb))
    if args.accept_wikitext is not None:
        w.add_float32("mtp.acceptance.wikitext", float(args.accept_wikitext))
    if args.target_gguf:
        n = copy_vocab(w, args.target_gguf)
        if n == 0:
            raise SystemExit(f"no tokenizer.* fields copied from {args.target_gguf}")
        print(f"copied {n} vocab/context fields from the target")
        # the loader reads hyperparameters under the arch prefix; inherit the
        # target's context length (copy_vocab only carries general.context_length)
        from gguf import GGUFReader as _R
        ctx_len = 262144
        for f in _R(args.target_gguf).fields.values():
            if f.name == "general.context_length" and f.data:
                ctx_len = int(f.data[0])
        w.add_uint32(f"{w.arch}.context_length", ctx_len)
    # fp16 keeps the sidecar small; gguf-py reverses the numpy shape, so GGUF
    # ne = [2*hidden, width] for fc1 as the loader expects.
    w.add_tensor("mtp.fc1.weight", fc1.astype(np.float16))
    w.add_tensor("mtp.fc2.weight", fc2.astype(np.float16))
    if fc3 is not None:
        w.add_tensor("mtp.fc3.weight", fc3.astype(np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}: hidden {hidden}, width {width}, state {'yes' if fc3 is not None else 'no'}")

    if args.verify:
        from gguf import GGUFReader
        r = GGUFReader(args.out)
        got = {}
        for t in r.tensors:
            # gguf-py reverses dims on write, so the reader hands arrays back
            # in the original torch layout [out, in] -- no transpose needed.
            got[t.name] = torch.from_numpy(np.asarray(t.data).copy()).float()
        torch.manual_seed(0)
        h = torch.randn(4, hidden)
        e = torch.randn(4, hidden)
        ref = head_forward(torch.from_numpy(fc1), torch.from_numpy(fc2), h, e)
        out = head_forward(got["mtp.fc1.weight"], got["mtp.fc2.weight"], h, e)
        md = float((ref - out).abs().max())
        rel = md / max(float(ref.abs().max()), 1e-6)
        print(f"verify: max abs diff {md:.3e} (rel {rel:.2e})")
        if rel > 1e-3:
            raise SystemExit("round-trip mismatch -- dim convention is wrong")
        if fc3 is not None:
            ref3 = (torch.from_numpy(fc3) @ torch.nn.functional.gelu(
                torch.from_numpy(fc1) @ torch.cat([rms_t(h), rms_t(e)], dim=-1).T)).T
            out3 = (got["mtp.fc3.weight"] @ torch.nn.functional.gelu(
                got["mtp.fc1.weight"] @ torch.cat([rms_t(h), rms_t(e)], dim=-1).T)).T
            md3 = float((ref3 - out3).abs().max())
            rel3 = md3 / max(float(ref3.abs().max()), 1e-6)
            print(f"verify state head: max abs diff {md3:.3e} (rel {rel3:.2e})")
            if rel3 > 1e-3:
                raise SystemExit("state-head round-trip mismatch")
        print("verify OK")


def rms_t(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


if __name__ == "__main__":
    main()

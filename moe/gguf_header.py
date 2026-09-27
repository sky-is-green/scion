"""Read a GGUF header (metadata + tensor infos) over HTTP range requests.

No weights are downloaded: only the header region is fetched.  Used to census
per-tensor quantization types of a remote GGUF (e.g. an APEX quant) without
pulling gigabytes.

Stdlib only (urllib) so it runs in any env.
"""

from __future__ import annotations

import json
import struct
import sys
import urllib.request

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 31: "Q4_0_4_4", 32: "Q4_0_4_8", 33: "Q4_0_8_8",
    34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
    142: "PQ2_0", 143: "PTQ1_0",
}

# bytes per weight for block formats we care about
BPW = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0, "Q8_0": 8.5, "Q6_K": 6.5625,
    "Q5_K": 5.5, "Q5_1": 6.0, "Q5_0": 5.5, "Q4_K": 4.5, "Q4_1": 5.0,
    "Q4_0": 4.5, "Q3_K": 3.4375, "Q2_K": 2.625, "IQ4_NL": 4.5,
    "IQ4_XS": 4.25, "IQ3_S": 3.44, "IQ3_XXS": 3.06, "IQ2_S": 2.5,
    "IQ2_XS": 2.31, "IQ2_XXS": 2.06, "IQ1_S": 1.56, "IQ1_M": 1.75,
    "TQ1_0": 1.6875, "TQ2_0": 2.0625, "PQ2_0": 2.125, "PTQ1_0": 1.75,
    "MXFP4": 4.25, "NVFP4": 4.5,
}


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0], o + 4


def _u64(b, o):
    return struct.unpack_from("<Q", b, o)[0], o + 8


def _s(b, o):
    n, o = _u64(b, o)
    return b[o:o + n].decode("utf-8", "replace"), o + n


def _val(b, o, t):
    if t == 0:
        return b[o], o + 1
    if t == 1:
        return struct.unpack_from("<b", b, o)[0], o + 1
    if t == 2:
        return struct.unpack_from("<H", b, o)[0], o + 2
    if t == 3:
        return struct.unpack_from("<h", b, o)[0], o + 2
    if t == 4:
        return _u32(b, o)
    if t == 5:
        return struct.unpack_from("<i", b, o)[0], o + 4
    if t == 6:
        return struct.unpack_from("<f", b, o)[0], o + 4
    if t == 7:
        return bool(b[o]), o + 1
    if t == 8:
        return _s(b, o)
    if t == 9:  # array
        at, o = _u32(b, o)
        n, o = _u64(b, o)
        out = []
        if at == 8:
            # strings: skip contents but remember a few
            for i in range(n):
                v, o = _s(b, o)
                if i < 3:
                    out.append(v)
            return {"__array__": "str", "len": n, "head": out}, o
        step = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[at]
        out = []
        for i in range(n):
            v, _ = _val(b, o, at)
            o += step
            if i < 3:
                out.append(v)
        return {"__array__": "num", "len": n, "head": out}, o
    if t == 10:
        return _u64(b, o)
    if t == 11:
        return struct.unpack_from("<q", b, o)[0], o + 8
    if t == 12:
        return struct.unpack_from("<d", b, o)[0], o + 8
    raise ValueError(f"unknown gguf value type {t} at {o}")


def fetch_range(url, start, end, timeout=120):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse(url, budget=64 << 20):
    """Fetch up to `budget` header bytes and parse metadata + tensor infos."""
    b = fetch_range(url, 0, budget - 1)
    if b[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    o = 4
    version, o = _u32(b, o)
    n_tensors, o = _u64(b, o)
    n_kv, o = _u64(b, o)
    meta = {}
    for _ in range(n_kv):
        k, o = _s(b, o)
        t, o = _u32(b, o)
        v, o = _val(b, o, t)
        meta[k] = v
    tensors = []
    for _ in range(n_tensors):
        name, o = _s(b, o)
        nd, o = _u32(b, o)
        dims = []
        for _ in range(nd):
            d, o = _u64(b, o)
            dims.append(d)
        t, o = _u32(b, o)
        off, o = _u64(b, o)
        n = 1
        for d in dims:
            n *= d
        tensors.append({"name": name, "dims": dims, "type": GGML_TYPES.get(t, str(t)),
                        "type_id": t, "offset": off, "n": n})
    return {"version": version, "n_tensors": n_tensors, "n_kv": n_kv,
            "meta": meta, "tensors": tensors, "header_bytes": o,
            "fetched_bytes": len(b)}


def role(name):
    if name.startswith("blk."):
        parts = name.split(".")
        suffix = ".".join(parts[2:])
        layer = int(parts[1])
    else:
        suffix, layer = name, -1
    if "exps" in suffix:
        kind = "expert" if "shexp" not in suffix else "shared"
        return f"{kind}:{suffix.split('.')[0]}"
    if "ffn_gate_inp" in suffix or "router" in suffix:
        return "router"
    if "attn" in suffix or "ssm" in suffix or "linear_attn" in suffix or "conv1d" in suffix:
        return "attn_ssm"
    if "norm" in suffix:
        return "norm"
    return "other"


def summarize(parsed):
    tot_bits = 0
    tot_n = 0
    by_type = {}
    by_role = {}
    for t in parsed["tensors"]:
        t = dict(t)
        bits = BPW.get(t["type"], 16.0) * t["n"]
        tot_bits += bits
        tot_n += t["n"]
        by_type.setdefault(t["type"], {"tensors": 0, "n": 0})
        by_type[t["type"]]["tensors"] += 1
        by_type[t["type"]]["n"] += t["n"]
        r = role(t["name"])
        by_role.setdefault(r, {"tensors": 0, "n": 0, "bits": 0.0})
        by_role[r]["tensors"] += 1
        by_role[r]["n"] += t["n"]
        by_role[r]["bits"] += bits
    for r in by_role.values():
        r["bpw"] = round(r["bits"] / r["n"], 3) if r["n"] else 0.0
    return {"total_params": tot_n, "effective_bpw": round(tot_bits / tot_n, 3),
            "by_type": by_type, "by_role": by_role}


def main():
    url = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    p = parse(url)
    s = summarize(p)
    print(json.dumps({"version": p["version"], "n_tensors": p["n_tensors"],
                      "header_bytes": p["header_bytes"], **s}, indent=2))
    if out:
        with open(out, "w") as fh:
            json.dump({"source": url, "summary": s,
                       "meta_keys": sorted(p["meta"].keys()),
                       "tensors": p["tensors"]}, fh, indent=1)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

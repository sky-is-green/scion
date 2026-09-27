#!/usr/bin/env python3
"""Collect the 35B quant-retention grid from llama-perplexity logs.

Reads:  results/<tag>-{ppl,hellaswag,winogrande,kld}.log  (pod grid)
        results/local-release-*.log                        (the release, measured on card 1)
Writes: retention-grid.md and retention-grid.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SIZES_GB = {          # decimal GB, from the empero GGUF model card / the release file
    "release": 11.337,
    "IQ2_M": 12.558, "Q2_K": 13.839, "IQ3_M": 16.340, "Q3_K_M": 17.664,
    "IQ4_XS": 19.628, "Q4_K_M": 21.713, "Q5_K_M": 25.348, "Q6_K": 29.209,
    "Q8_0": 37.802, "BF16": 71.067,
}
PARAMS = 34.7e9  # text-path parameters
ORDER = ["release", "IQ2_M", "Q2_K", "IQ3_M", "Q3_K_M", "IQ4_XS", "Q4_K_M",
         "Q5_K_M", "Q6_K", "Q8_0", "BF16"]


def parse(path: Path) -> dict:
    t = path.read_text(errors="ignore")
    out: dict = {}
    m = re.findall(r"Final estimate: PPL =\s*([\d.]+)\s*\+/-\s*([\d.]+)", t)
    if m:
        out["ppl"], out["ppl_err"] = float(m[-1][0]), float(m[-1][1])
    rows = re.findall(r"^\s*400\s+([\d.]+)%", t, re.M)
    if rows:
        out["hellaswag"] = float(rows[-1])
    m = re.findall(r"Final Winogrande score\(400 tasks\):\s*([\d.]+)\s*\+/-\s*([\d.]+)", t)
    if m:
        out["winogrande"], out["winogrande_err"] = float(m[-1][0]), float(m[-1][1])
    for key, pat in [
        ("kld_mean",   r"Mean\s+KLD:\s*([\d.]+)"),
        ("kld_median", r"Median\s+KLD:\s*([\d.]+)"),
        ("kld_999",    r"99\.9%\s+KLD:\s*([\d.]+)"),
        ("kld_max",    r"Maximum KLD:\s*([\d.]+)"),
    ]:
        m = re.findall(pat, t)
        if m:
            out[key] = float(m[-1])
    return out


def main() -> int:
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    data: dict[str, dict] = {}
    for tag in ORDER:
        row: dict = {}
        if tag == "release":
            local = Path("local-release")
            for kind, fname in [("ppl", "ppl.log"), ("hellaswag", "hellaswag.log"),
                                ("winogrande", "winogrande.log"), ("kld", "kld.log")]:
                f = (res / "local-release" / fname)
                if f.exists():
                    row.update(parse(f))
            # fall back to the raw local logs
            for f in Path(".").glob("eval-release.log"):
                row.update({k: v for k, v in parse(f).items() if k not in row})
            for f in Path(".").glob("bench-release.log"):
                row.update({k: v for k, v in parse(f).items() if k not in row})
        else:
            for kind in ["ppl", "hellaswag", "winogrande", "kld"]:
                for cand in (res / f"{tag}-{kind}.log", res / f"{tag.lower()}-{kind}.log"):
                    if cand.exists():
                        row.update(parse(cand))
                        break
        if row:
            data[tag] = row

    if "BF16" in data:
        bf = data["BF16"]
        for tag, row in data.items():
            if tag == "BF16":
                continue
            if "ppl" in row and "ppl" in bf:
                row["ppl_ratio"] = round(row["ppl"] / bf["ppl"], 4)
            if "hellaswag" in row and "hellaswag" in bf:
                row["hs_retention"] = round(row["hellaswag"] / bf["hellaswag"], 4)

    lines = ["| model | size (GB) | bpw | PPL | KLD mean | KLD median | KLD 99.9% | KLD max | HellaSwag | Winogrande |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for tag in ORDER:
        if tag not in data:
            continue
        r = data[tag]
        size = SIZES_GB[tag]
        bpw = size * 8 / (PARAMS / 1e9)
        def f(k, nd=4):
            return f"{r[k]:.{nd}f}" if k in r else "—"
        lines.append(
            f"| {tag} | {size:.2f} | {bpw:.2f} | {f('ppl')} | {f('kld_mean')} | {f('kld_median')} | "
            f"{f('kld_999')} | {f('kld_max')} | {f('hellaswag',2)}% | {f('winogrande',2)}% |")
    md = "\n".join(lines)
    print(md)
    (res / "retention-grid.md").write_text(md + "\n")
    (res / "retention-grid.json").write_text(json.dumps(data, indent=2))
    print(f"\nwrote {res}/retention-grid.md and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Export the token -> bytes table the engine's logit dump reads, and read the dump back.

Table file (`byte-table.bin`), read by `STRATA_Q36_BYTE_TABLE`:

    magic  "SBT1"         4 bytes
    vocab  uint32         number of tokens
    width  uint32         bytes per row (the widest token, plus the EOT slot when eot=1)
    eot    uint32         1: row [len] holds 256 (end-of-token); 0: Marginalize-It form
    rows   int16[vocab][width]   bytes 0..255, 256 = EOT, -1 = padding
    (a row's true length is the count of entries >= 0)

Dump file (`logit-dump.bin`), written by the engine (STRATA_Q36_LOGIT_DUMP):

    magic  "SBD1"  4 bytes
    n_sym  uint32  256 or 257 (EOT is the extra symbol)
    then one record per text position that has a known next token:
      token   int32          the next token (the gold continuation)
      p_token float32        the teacher's probability of that token
      p_exact float32        ...summed over every token with the same byte string
      n_rows  int32          the token's byte length + (1 with EOT)
      dist    float16[n_rows][n_sym]   the converted byte distributions
    The check: the product of dist[k][gold[k]] along a record's rows equals p_exact.

Usage:
    python byte_table.py --pack <pack>/tokenizer --out byte-table.bin          # EOT table
    python byte_table.py --pack <pack>/tokenizer --out byte-table-mi.bin --no-eot
    python byte_table.py --verify <logit-dump.bin>                             # parse and check records
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from byte_logits import EOT, byte_distributions, load_byte_table  # noqa: E402

TABLE_MAGIC = b"SBT1"
DUMP_MAGIC = b"SBD1"


def export_table(pack: Path, out: Path, use_eot: bool) -> None:
    table, lengths = load_byte_table(pack, use_eot=use_eot)
    vocab, width = table.shape
    with out.open("wb") as f:
        f.write(TABLE_MAGIC)
        f.write(struct.pack("<III", vocab, width, 1 if use_eot else 0))
        f.write(table.astype("<i2").tobytes())
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB: {vocab:,} x {width}, "
          f"{'EOT' if use_eot else 'Marginalize-It'}, lengths {int(lengths.min())}..{int(lengths.max())})")


def read_table(path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    raw = path.read_bytes()
    assert raw[:4] == TABLE_MAGIC, f"{path}: not a byte table"
    vocab, width, eot = struct.unpack_from("<III", raw, 4)
    table = np.frombuffer(raw, dtype="<i2", count=vocab * width, offset=16).reshape(vocab, width)
    lengths = (table >= 0).sum(axis=1).astype(np.int32)
    return table, lengths, bool(eot)


def record_size(n_rows: int, n_sym: int) -> int:
    return 16 + n_rows * n_sym * 2


def verify_dump(path: Path, tol: float = 1e-2) -> int:
    """Parse a dump and check the End-Of-Token identity: product along the gold path == p_exact."""
    raw = path.read_bytes()
    assert raw[:4] == DUMP_MAGIC, f"{path}: not a logit dump"
    (n_sym,) = struct.unpack_from("<I", raw, 4)
    off, n, bad, worst = 8, 0, 0, 0.0
    prods: list[float] = []
    while off < len(raw):
        token, p_token, p_exact, n_rows = struct.unpack_from("<iffi", raw, off)
        off += 16
        dist = np.frombuffer(raw, dtype="<f2", count=n_rows * n_sym, offset=off).reshape(n_rows, n_sym)
        off += n_rows * n_sym * 2
        # the gold path is the token's own bytes, then EOT when the alphabet has it
        table, lengths, eot = _last_table  # set by --table
        row = table[token]
        m = int(lengths[token]) - (1 if eot else 0)
        gold = [int(b) for b in row[:m]] + ([EOT] if eot else [])
        if len(gold) != n_rows:
            bad += 1
            continue
        prod = 1.0
        for k in range(n_rows):
            prod *= float(dist[k, gold[k]])
        prods.append(prod)
        worst = max(worst, abs(prod - float(p_exact)))
        if abs(prod - float(p_exact)) > tol * max(1e-6, float(p_exact)):
            bad += 1
        n += 1
    a = np.asarray(prods) if prods else np.zeros(1)
    print(f"dump: {n:,} records, n_sym {n_sym}, product-vs-p_exact worst |diff| {worst:.3e}, "
          f"{bad} over tolerance ({tol:.0e} relative)")
    print(f"      product: median {np.median(a):.3e}, mean {a.mean():.3e} "
          f"(the teacher's true next-token probabilities)")
    return 1 if bad else 0


_last_table: tuple[np.ndarray, np.ndarray, bool] = (np.zeros((0, 0), dtype="<i2"),
                                                    np.zeros(0, dtype=np.int32), True)


def selftest(pack: Path, table_path: Path) -> int:
    """Build a dump the way the engine does (same conversion, same record shape) and verify it."""
    import tempfile
    table, lengths, eot = read_table(table_path)
    from byte_logits import _encode
    rng = np.random.default_rng(3)
    vocab = table.shape[0]
    ids = _encode(pack, "The quick brown fox jumps over the lazy dog. 0123456789")
    probs = rng.random(vocab) * 1e-3
    probs[ids] += 1.0
    probs /= probs.sum()
    n_sym = 257 if eot else 256
    buf = bytearray()
    buf += DUMP_MAGIC + struct.pack("<I", n_sym)
    pack_ = b""
    for t in ids:
        m = int(lengths[t]) - (1 if eot else 0)
        gold = [int(b) for b in table[t, :m]] + ([EOT] if eot else [])
        d = byte_distributions(probs, table, lengths, gold, use_eot=eot)
        p_tok = float(probs[t])
        rows = np.where((table[:, :m] == np.asarray(gold[:m], dtype=np.int16)).all(axis=1)
                        & (lengths == m + (1 if eot else 0)))[0]
        p_exact = float(probs[rows].sum())
        dist = np.stack(d).astype(np.float16)
        pack_ += struct.pack("<iffi", t, p_tok, p_exact, len(gold)) + dist.astype("<f2").tobytes()
    buf += pack_
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(bytes(buf))
        path = Path(f.name)
    global _last_table
    _last_table = (table, lengths, eot)
    rc = verify_dump(path)
    path.unlink()
    print(f"selftest: a dump of {len(ids)} records written the engine's way "
          f"{'passes' if rc == 0 else 'FAILS'} the identity check")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", type=Path, default=Path("/home/penis/models/scion-strata/pack-q6k/tokenizer"))
    ap.add_argument("--out", type=Path, default=Path("byte-table.bin"))
    ap.add_argument("--no-eot", action="store_true")
    ap.add_argument("--verify", type=Path, default=None, help="a logit dump to check")
    ap.add_argument("--table", type=Path, default=None, help="the table --verify should use")
    ap.add_argument("--selftest", action="store_true", help="build a dump in Python and check it")
    args = ap.parse_args()
    if args.selftest:
        return selftest(args.pack, args.table or Path("byte-table.bin"))
    if args.verify is not None:
        global _last_table
        tpath = args.table or Path("byte-table.bin")
        _last_table = read_table(tpath)
        return verify_dump(args.verify)
    export_table(args.pack, args.out, use_eot=not args.no_eot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

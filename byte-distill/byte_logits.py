#!/usr/bin/env python
"""Token logits -> byte logits, the two ways the paper gives (arXiv:2609.12303).

  Marginalize-It   approximate: for each byte position, marginalize the teacher's token
                   probabilities over the tokens whose decoded bytes match the prefix, and
                   renormalize; mass that would continue past a differing byte is dropped.
  End-Of-Token     exact: append an <eot> symbol after every token's bytes, so the residual
                   mass at each position lands on "ends here" and nothing is lost.

Both are single-pass over a teacher distribution (no extra teacher forwards).  This module is
the reference implementation for a dump format: the engine writes one converted distribution
per text position, and that file is what a distillation run trains against.

  python byte_logits.py --self-test
  python byte_logits.py --stats --pack <pack>/tokenizer --text <file>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / "strata-dh" / "tools"))

EOT = 256   # the end-of-token symbol (byte alphabet 0..255, then EOT)


def load_byte_table(pack: Path, use_eot: bool) -> tuple[np.ndarray, np.ndarray]:
    """(table, lengths): table[i, k] is token i's k-th byte (EOT at its end when use_eot), -1 padded.

    The token strings in vocab.json are GPT-2 byte-level unicode: map every character back to its
    byte, so the table holds the model's true UTF-8 bytes for each token.
    """
    from strata_tokenizer import UNICODE_TO_BYTE
    vocab = json.loads((pack / "vocab.json").read_text(encoding="utf-8"))
    n = max(vocab.values()) + 1
    byte_seqs: list[bytes | None] = [None] * n
    for token, i in vocab.items():
        byte_seqs[i] = bytes(UNICODE_TO_BYTE[ch] for ch in token)
    width = max(len(b) for b in byte_seqs) + 1
    table = np.full((n, width), -1, dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int32)
    for i, b in enumerate(byte_seqs):
        assert b is not None
        table[i, : len(b)] = np.frombuffer(b, dtype=np.uint8)
        lengths[i] = len(b) + (1 if use_eot else 0)
        if use_eot:
            table[i, len(b)] = EOT
    return table, lengths


def byte_distributions(probs: np.ndarray, table: np.ndarray, lengths: np.ndarray,
                       gold: list[int], use_eot: bool) -> list[np.ndarray]:
    """P(byte at position k | gold[:k], context) for every k of `gold` (the paper's Appendix F.1).

    `probs`: the teacher's (vocab,) next-token distribution.  `gold`: the observed byte sequence,
    with a trailing EOT when use_eot.  Each returned array has 257 entries with EOT (index 256)
    when use_eot, else 256.
    """
    n_bytes = 257 if use_eot else 256
    assert probs.size == table.shape[0], f"probs {probs.size} vs table {table.shape[0]}"
    live = np.ones(probs.size, dtype=bool)
    out: list[np.ndarray] = []
    for k in range(len(gold)):
        active = live & (lengths > k)
        acc = np.bincount(table[active, k], weights=probs[active], minlength=n_bytes)
        total = acc.sum()
        out.append(acc / total if total > 0 else acc)
        live &= (table[:, k] == gold[k]) & (lengths > k + 1)
    return out


def gold_bytes(table: np.ndarray, length: int, token_id: int, use_eot: bool) -> list[int]:
    """The byte sequence of one token (plus EOT) as the conversion needs it."""
    g = [int(b) for b in table[token_id, :length]]
    if use_eot:
        g.append(EOT)
    return g


# --------------------------------------------------------------------------------- self-test
def paper_example() -> None:
    """The paper's Figure 2 numbers: after `iram`, isu .5, isk .125, is .125, others ~0.

    Marginalize-It at the third byte: P(u)=0.8, P(k)=0.2.  End-Of-Token: P(u<eot>)=0.667,
    P(k<eot>)=0.167, P(is<eot>)=0.167.
    """
    vocab = ["<bos>", "T", "iram", "is", "isu", "isk"]
    n = len(vocab)
    probs = np.zeros(n)
    for tok, p in (("isu", 0.5), ("isk", 0.125), ("is", 0.125)):
        probs[vocab.index(tok)] = p
    b = [t.encode() for t in vocab]
    width = max(len(x) for x in b) + 1

    def table_for(use_eot: bool):
        table = np.full((n, width), -1, dtype=np.int32)
        lengths = np.zeros(n, dtype=np.int32)
        for i, x in enumerate(b):
            table[i, : len(x)] = np.frombuffer(x, dtype=np.uint8)
            lengths[i] = len(x) + (1 if use_eot else 0)
            if use_eot:
                table[i, len(x)] = EOT
        return table, lengths

    gold = list(b"isu")
    t_mi, l_mi = table_for(False)
    d = byte_distributions(probs, t_mi, l_mi, gold, use_eot=False)
    assert np.isclose(d[0][ord("i")], 1.0), "first byte is exactly 'i'"
    assert np.isclose(d[1][ord("s")], 1.0), "second byte is exactly 's'"
    assert np.isclose(d[2][ord("u")], 0.8) and np.isclose(d[2][ord("k")], 0.2), \
        f"Marginalize-It: expected 0.8/0.2, got {d[2][ord('u')]:.3f}/{d[2][ord('k')]:.3f}"

    t_eot, l_eot = table_for(True)
    d = byte_distributions(probs, t_eot, l_eot, gold + [EOT], use_eot=True)
    assert np.isclose(d[2][ord("u")], 0.5 / 0.75) and np.isclose(d[2][ord("k")], 0.125 / 0.75) \
        and np.isclose(d[2][EOT], 0.125 / 0.75), \
        f"EOT: expected 0.667/0.167/0.167, got {d[2][ord('u')]:.3f}/{d[2][ord('k')]:.3f}/{d[2][EOT]:.3f}"
    print("paper example: Marginalize-It 0.8/0.2, End-Of-Token 0.667/0.167/0.167 -- ok")


def real_checks(pack: Path, texts: list[str], seed: int = 7) -> dict:
    """Exactness and mass on a real tokenizer table with synthetic teacher distributions.

    The teacher here is Zipf-ish over `support` random tokens; for tokens whose exact-byte
    probability is not zero, compare the product of the converted byte distributions along the
    token's path against the exact probability (sum over tokens with those exact bytes).  The
    End-Of-Token path is exact; the Marginalize-It path loses the mass of longer continuations,
    so its gap is the method's approximation error on a real vocabulary.
    """
    rng = np.random.default_rng(seed)
    table_eot, len_eot = load_byte_table(pack, use_eot=True)
    table_mi, len_mi = load_byte_table(pack, use_eot=False)
    vocab = table_eot.shape[0]
    text_ids = [t for text in texts for t in _encode(pack, text)]
    idx = np.unique(np.concatenate([np.asarray(text_ids, dtype=np.int64),
                                    rng.choice(vocab, size=max(0, 4000 - len(set(text_ids))), replace=False)]))
    w = 1.0 / (1.0 + np.arange(idx.size))
    probs = np.zeros(vocab)
    probs[idx] = w / w.sum()
    supported = np.zeros(vocab, dtype=bool)
    supported[idx] = True

    worst_eot = 0.0
    mi_rel: list[float] = []
    n_checked = 0
    for text in texts:
        ids = _encode(pack, text)
        for t in ids[:300]:
            m = int(len_mi[t])
            if m == 0 or not supported[t]:
                continue   # only tokens the synthetic teacher actually puts mass on
            gold_mi = [int(x) for x in table_mi[t, :m]]
            gold_eot = gold_mi + [EOT]
            rows = np.where((table_mi[:, :m] == np.asarray(gold_mi, dtype=np.int32)).all(axis=1)
                            & (len_mi == m))[0]
            want = float(probs[rows].sum())
            if want <= 1e-9:
                continue
            d_eot = byte_distributions(probs, table_eot, len_eot, gold_eot, use_eot=True)
            prod_eot = float(np.prod([d_eot[k][gold_eot[k]] for k in range(len(gold_eot))]))
            worst_eot = max(worst_eot, abs(prod_eot - want))

            d_mi = byte_distributions(probs, table_mi, len_mi, gold_mi, use_eot=False)
            prod_mi = float(np.prod([d_mi[k][gold_mi[k]] for k in range(len(gold_mi))]))
            mi_rel.append(abs(prod_mi - want) / want)
            n_checked += 1
    a = np.asarray(mi_rel) if mi_rel else np.zeros(1)
    return {"eot_max_abs_err": worst_eot, "n_checked": n_checked,
            "mi_rel_median": float(np.median(a)), "mi_rel_p90": float(np.percentile(a, 90)),
            "mi_rel_max": float(a.max()), "mi_rel_over_10pct": float(np.mean(a > 0.10))}


def _encode(pack: Path, text: str) -> list[int]:
    import strata_tokenizer as ST
    vocab = json.loads((pack / "vocab.json").read_text(encoding="utf-8"))
    tokens: list[str | None] = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (pack / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((pack / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types).encode(text, parse_special=True)


def cmd_stats(args) -> int:
    text = Path(args.text).read_text(encoding="utf-8") if args.text else "The quick brown fox jumps over the lazy dog. " * 200
    table, lengths = load_byte_table(args.pack, use_eot=False)
    ids = _encode(args.pack, text)
    lens = np.asarray([lengths[t] for t in ids], dtype=np.float64)
    mean = float(lens.mean())
    print(f"stats: {args.pack}")
    print(f"  vocab {table.shape[0]:,} tokens, widest {lengths.max()} bytes")
    print(f"  text: {len(text):,} chars -> {len(ids):,} tokens, {mean:.3f} bytes/token "
          f"(p10 {np.percentile(lens,10):.0f}, p90 {np.percentile(lens,90):.0f})")
    for name, b in (("float32", 4), ("float16", 2)):
        rows = mean + 1  # the EOT row
        per_token = rows * 257 * b
        print(f"  EOT dump, {name}: {per_token:,.0f} B/token -> "
              f"{per_token * 1e6 / 1e9:.2f} GB per 1M tokens")
    eot_f32 = (mean + 1) * 257 * 4
    per_token_full = table.shape[0] * 4
    print(f"  token, full float32 logits: {per_token_full:,} B/token -> "
          f"{per_token_full * 1e6 / 1e9:.2f} GB per 1M tokens")
    for k in (600, 4096, 16384):
        per_token_k = k * 8 + 4  # k float32 logits + int32 indices, the paper's sparse form
        print(f"  token top-{k}: {per_token_k:,} B/token -> {per_token_k * 1e6 / 1e9:.2f} GB per 1M tokens, "
              f"{per_token_k / eot_f32:.2f}x the EOT float32")
    return 0


def cmd_convert(args) -> int:
    """Demo: a random teacher distribution for one text, converted both ways, sizes reported."""
    table, lengths = load_byte_table(args.pack, use_eot=True)
    rng = np.random.default_rng(args.seed)
    vocab = table.shape[0]
    probs = np.zeros(vocab)
    idx = rng.choice(vocab, size=args.support, replace=False)
    w = 1.0 / (1.0 + np.arange(idx.size))
    probs[idx] = w / w.sum()
    text = "The quick brown fox jumps over the lazy dog."
    ids = _encode(args.pack, text)
    rows = 0
    for t in ids:
        m = int(lengths[t]) - 1
        gold = [int(x) for x in table[t, :m]] + [EOT]
        d = byte_distributions(probs, table, lengths, gold, use_eot=True)
        rows += len(d)
    print(f"convert: {len(ids)} tokens -> {rows} byte positions, "
          f"{rows * 257 * 2 / 1024:.1f} KiB as float16 for this sentence")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", type=Path, default=Path("/home/penis/models/scion-strata/pack-q6k/tokenizer"))
    ap.add_argument("--text", default=None, help="a file for --stats (default: a repeated sentence)")
    ap.add_argument("--topk", type=int, default=600, help="token top-k to compare against (the paper's k)")
    ap.add_argument("--support", type=int, default=4000, help="--convert: tokens with probability mass")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--real-checks", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        paper_example()
        if (args.pack / "vocab.json").exists():
            r = real_checks(args.pack, ["The quick brown fox jumps over the lazy dog. " * 3,
                                        "def f(x): return x * x  # code, numbers 1234567890"],
                            seed=args.seed)
            print(f"real table: End-Of-Token max |prod - P(exact bytes)| = {r['eot_max_abs_err']:.3e} "
                  f"(exact when ~0), {r['n_checked']} tokens")
            print(f"            Marginalize-It relative gap on supported tokens: median "
                  f"{r['mi_rel_median']:.3f}, p90 {r['mi_rel_p90']:.3f}, max {r['mi_rel_max']:.3f}, "
                  f"{100*r['mi_rel_over_10pct']:.0f}% of tokens over 10%")
        return 0
    if args.stats:
        return cmd_stats(args)
    if args.convert:
        return cmd_convert(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Curation metrics for context selection — the quantum-sweep's top probe.

The quantum-connections sweep (§7.7E) found that the transferable part of the
entanglement/entropy intuition is *quantum-probability mathematics used
classically*: a set of units (chunks, tokens, contexts) can be lifted to a
trace-1 PSD density matrix, and its spectral quantities measure diversity and
effective dimensionality; the relational structure between chunks is measured
by bipartite mutual information; and a curation rule can score a unit by
COMI's Marginal Information Gain (relevance minus redundancy).

This module is deliberately representation-agnostic: it takes vectors (or
integer sequences) and returns the metrics.  The CLI at the bottom is the
first datapoint — lexical TF-IDF vectors and token-id MI, no model needed —
so the suite can be re-run on embeddings later without changing the math.

References (all in §7.7E): Vendi score / VNE diversity (Friedman & Dieng 2022
lineage; Max-VNE principle 2602.02117), bipartite-MI scaling / L2M (OpenReview
2026), COMI / MIG (2602.01719), and the H-DMAtt PSD caveat (naive symmetrized
attention is not a density matrix; use the Gram construction here).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter

import torch


# ----------------------------------------------------- density / spectra ---

def gram_density(vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Trace-1 PSD density matrix from row vectors (cosine Gram, normalised).

    ``vectors`` is ``(n, d)``; the result is ``(n, n)`` with trace 1.  A
    cosine-similarity Gram matrix of real vectors is PSD by construction, so
    the result is a valid density matrix — unlike a symmetrised attention
    matrix, which generally is not (H-DMAtt's caveat).  Zero rows (a chunk
    with no vocabulary overlap) are dropped rather than allowed to break PSD.
    """
    v = vectors.float()
    if v.dim() != 2:
        raise ValueError("vectors must be (n, d)")
    norms = v.norm(dim=-1, keepdim=True)
    keep = norms.squeeze(-1) > eps
    if not bool(keep.any()):
        raise ValueError("no non-zero vectors")
    u = v[keep] / norms[keep].clamp_min(eps)
    g = u @ u.t()
    g = (g + g.t()) * 0.5                      # exact symmetry
    n = g.shape[0]
    rho = g / n
    return rho


def spectral_stats(rho: torch.Tensor) -> dict:
    """Von Neumann entropy and friends of a density matrix.

    ``vne`` = -Tr(rho log rho) in nats; ``vendi`` = exp(vne) (the effective
    number of distinct units, 1..n); ``purity`` = Tr(rho^2) (1 = pure,
    1/n = maximally mixed); ``participation`` = 1 / Tr(rho^2) (the
    participation-ratio effective dimension).
    """
    if rho.dim() != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be square")
    lam = torch.linalg.eigvalsh(rho.double()).clamp_min(0.0)
    s = float(lam.sum())
    if s <= 0:
        raise ValueError("rho has no mass")
    lam = lam / s
    nz = lam[lam > 1e-12]
    vne = float(-(nz * nz.log()).sum())
    purity = float((lam ** 2).sum())
    return {"n": int(rho.shape[0]), "vne": vne, "vendi": math.exp(vne),
            "purity": purity, "participation": 1.0 / purity}


# ------------------------------------------------------------- mutual info ---

def bipartite_mi(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """Mutual information (nats) between two equal-length integer sequences.

    The L2M-style bipartite MI: pair ``x[i]`` with ``y[i]``, build the joint
    histogram, and subtract the product of the marginals.  Independent
    sequences give ~0; ``y = x`` gives H(x).
    """
    x = x.reshape(-1).long()
    y = y.reshape(-1).long()
    if x.numel() != y.numel():
        raise ValueError("x and y must have equal length")
    if x.numel() == 0:
        raise ValueError("empty sequences")
    xy = x * (int(y.max()) + 1) + y
    j = torch.bincount(xy).float()
    px = torch.bincount(x).float()
    py = torch.bincount(y).float()
    pj = j / j.sum()
    px = px / px.sum()
    py = py / py.sum()
    nz = pj > 0
    mi = (pj[nz] * (pj[nz].log() - eps)).sum()      # = -H(X, Y)
    # subtract H(X)+H(Y) via the marginals of the observed pairs only
    # (the joint's support is a subset; use the marginal entropies directly)
    hx = -(px[px > 0] * (px[px > 0].log() - eps)).sum()
    hy = -(py[py > 0] * (py[py > 0].log() - eps)).sum()
    return float(hx + hy + mi)                       # H(X) + H(Y) - H(X,Y)


def chunk_mi_curve(tokens: list[int], lengths: list[int] | None = None) -> list[dict]:
    """Bipartite MI between the two halves of a window, across window sizes.

    For each ``l`` in ``lengths``: take windows of ``2l`` tokens, split each
    into its two halves X and Y, and average ``I(X;Y)`` over non-overlapping
    windows.  A growing curve is the long-range-dependence signal the L2M
    condition is about.
    """
    lengths = lengths or [16, 32, 64, 128]
    out = []
    for l in lengths:
        span = 2 * l
        vals = []
        for i in range(0, len(tokens) - span + 1, span):
            x = torch.tensor(tokens[i:i + l])
            y = torch.tensor(tokens[i + l:i + span])
            vals.append(bipartite_mi(x, y))
        out.append({"l": l, "n_windows": len(vals),
                    "mi": sum(vals) / len(vals) if vals else float("nan")})
    return out


# -------------------------------------------------------------------- MIG ---

def mig_scores(relevance: torch.Tensor, similarity: torch.Tensor) -> torch.Tensor:
    """COMI's Marginal Information Gain: relevance minus redundancy.

    ``relevance`` is ``(n,)`` (unit-to-query); ``similarity`` is ``(n, n)``
    (unit-to-unit, diagonal ignored).  Redundancy is the mean similarity of a
    unit to the others; MIG is relevance minus that.
    """
    if similarity.dim() != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be square")
    n = similarity.shape[0]
    if relevance.numel() != n:
        raise ValueError("relevance length must match similarity")
    if n < 2:
        return relevance.float().clone()
    s = similarity.float().clone()
    s.fill_diagonal_(0.0)
    red = s.sum(-1) / (n - 1)
    return relevance.float() - red


# ---------------------------------------------------------------- lexical ---

_WORD = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]")


def tokenize(text: str) -> list[str]:
    """A small lexical tokenizer (no model, no external corpus)."""
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def chunkify(tokens: list[str], size: int) -> list[list[str]]:
    return [tokens[i:i + size] for i in range(0, len(tokens), size)
            if len(tokens[i:i + size]) == size]


def _vocab_and_df(chunks: list[list[str]]) -> tuple[dict, Counter]:
    dfs: Counter = Counter()
    for ch in chunks:
        dfs.update(set(ch))
    return {w: i for i, w in enumerate(sorted(dfs))}, dfs


def tfidf_vectors(chunks: list[list[str]], vocab: dict | None = None,
                  dfs: Counter | None = None) -> torch.Tensor:
    """TF-IDF rows over the chunk vocabulary (lexical, CPU-only)."""
    if vocab is None or dfs is None:
        vocab, dfs = _vocab_and_df(chunks)
    n = len(chunks)
    m = torch.zeros(n, len(vocab))
    for r, ch in enumerate(chunks):
        counts = Counter(ch)
        for w, c in counts.items():
            m[r, vocab[w]] = (1.0 + math.log(c)) * math.log((1.0 + n) / (1.0 + dfs[w]))
    return m


def tfidf_query(query_tokens: list[str], chunks: list[list[str]]) -> torch.Tensor:
    """The query row in the *corpus* vocabulary/IDF space (same dimensions).

    A query vector built from its own vocabulary cannot be compared with the
    chunk vectors (dimension mismatch) — the corpus projection is what makes
    relevance a cosine in the shared space.
    """
    vocab, dfs = _vocab_and_df(chunks)
    n = len(chunks)
    q = torch.zeros(len(vocab))
    for w, c in Counter(query_tokens).items():
        if w in vocab:
            q[vocab[w]] = (1.0 + math.log(c)) * math.log((1.0 + n) / (1.0 + dfs[w]))
    return q


# -------------------------------------------------------------------- CLI ---

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="text file to analyse")
    ap.add_argument("--chunk", type=int, default=64, help="tokens per chunk")
    ap.add_argument("--query", default="", help="query text for MIG ranking")
    ap.add_argument("--top", type=int, default=10, help="MIG rows to print")
    ap.add_argument("--out", default="", help="write the full JSON here")
    ap.add_argument("--mi-lengths", default="16,32,64,128")
    args = ap.parse_args()

    text = open(args.path, encoding="utf-8", errors="ignore").read()
    tokens = tokenize(text)
    chunks = chunkify(tokens, args.chunk)
    if len(chunks) < 4:
        raise SystemExit(f"only {len(chunks)} full chunks; use a longer file")

    vecs = tfidf_vectors(chunks)
    rho = gram_density(vecs)
    spec = spectral_stats(rho)

    # token-id MI curve (map words to ids)
    vocab = {w: i for i, w in enumerate(sorted(set(tokens)))}
    ids = [vocab[t] for t in tokens]
    lengths = [int(x) for x in args.mi_lengths.split(",") if x]
    mi_curve = chunk_mi_curve(ids, lengths)

    report = {
        "path": args.path,
        "n_tokens": len(tokens),
        "n_chunks": len(chunks),
        "chunk_size": args.chunk,
        "spectral": spec,
        "bipartite_mi": mi_curve,
    }

    if args.query:
        qt = tokenize(args.query)
        if qt:
            q = tfidf_query(qt, chunks)
            qn = q / q.norm().clamp_min(1e-8)
            vn = vecs / vecs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            rel = vn @ qn
            sim = vn @ vn.t()
            mig = mig_scores(rel, sim)
            order = torch.argsort(mig, descending=True).tolist()
            report["query"] = args.query
            report["mig_top"] = [
                {"chunk": int(i), "mig": float(mig[i]), "relevance": float(rel[i])}
                for i in order[:args.top]]

    print(json.dumps({k: v for k, v in report.items() if k != "mig_top"},
                     indent=2))
    if "mig_top" in report:
        print("MIG top:")
        for row in report["mig_top"]:
            print(f"  chunk {row['chunk']:4d}  mig {row['mig']:+.4f}  "
                  f"relevance {row['relevance']:.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

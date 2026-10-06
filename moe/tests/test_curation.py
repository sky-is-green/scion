"""CPU tests for the curation metrics (the quantum-sweep probe, §8.10).

The three properties the probe's claims rest on:
  1. ``gram_density`` produces a *valid* density matrix (PSD, trace 1) -- the
     H-DMAtt caveat is exactly that a naive symmetrised attention matrix is not
     one, so the construction is the load-bearing part;
  2. the spectral quantities behave at the extremes (orthogonal units are
     maximally diverse, identical units are pure) and the bipartite MI is zero
     for independent sequences and the source entropy for a copy;
  3. MIG ranks a relevant-but-redundant unit below a relevant-unique one, the
     whole point of subtracting redundancy.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import torch

MOE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOE))

import curation as cur  # noqa: E402


def test_gram_density_is_psd_and_trace_one():
    torch.manual_seed(0)
    v = torch.randn(12, 7)
    rho = cur.gram_density(v)
    assert rho.shape == (12, 12)
    assert abs(float(torch.trace(rho)) - 1.0) < 1e-5
    lam = torch.linalg.eigvalsh(rho.double())
    assert float(lam.min()) > -1e-8, "PSD violated"


def test_gram_density_drops_zero_rows():
    v = torch.randn(5, 4)
    v[2] = 0.0
    rho = cur.gram_density(v)
    assert rho.shape == (4, 4)


def test_spectral_stats_extremes():
    # orthogonal unit vectors: maximally mixed -> vne = ln n, purity = 1/n
    rho = cur.gram_density(torch.eye(6))
    s = cur.spectral_stats(rho)
    assert abs(s["vne"] - torch.log(torch.tensor(6.0)).item()) < 1e-5
    assert abs(s["vendi"] - 6.0) < 1e-3
    assert abs(s["purity"] - 1 / 6) < 1e-5
    assert abs(s["participation"] - 6.0) < 1e-3
    # identical units: pure -> vne 0, vendi 1
    rho1 = cur.gram_density(torch.ones(4, 3))
    s1 = cur.spectral_stats(rho1)
    assert s1["vne"] < 1e-5
    assert abs(s1["vendi"] - 1.0) < 1e-4
    assert s1["purity"] > 1 - 1e-6


def test_bipartite_mi_independent_is_zero_and_copy_is_entropy():
    torch.manual_seed(1)
    n, k = 4000, 4
    x = torch.randint(0, k, (n,))
    y = torch.randint(0, k, (n,))
    mi_ind = cur.bipartite_mi(x, y)
    assert abs(mi_ind) < 0.02, "independent sequences must give ~0 MI"
    mi_copy = cur.bipartite_mi(x, x)
    h = torch.log(torch.tensor(float(k))).item()          # uniform source
    assert abs(mi_copy - h) < 0.05, (mi_copy, h)
    # a partial copy sits between
    y2 = x.clone()
    y2[::2] = torch.randint(0, k, ((n + 1) // 2,))
    mi_part = cur.bipartite_mi(x, y2)
    assert 0.2 < mi_part < mi_copy


def test_bipartite_mi_handles_sparse_large_id_spaces():
    """Regression: the joint must use unique(), not bincount().

    The joint code space is |X|*|Y| cells; a real corpus has 10^4+ ids, and
    bincount would allocate ~10^8+ cells (the first wiki.test.raw run sat
    there for 23 minutes).  This call must return, finite, immediately.
    """
    torch.manual_seed(2)
    x = torch.randint(0, 100_000, (500,))
    y = torch.randint(0, 100_000, (500,))
    mi = cur.bipartite_mi(x, y)
    assert math.isfinite(mi)
    mi_copy = cur.bipartite_mi(x, x)
    assert abs(mi_copy - math.log(500.0)) < 0.05    # all samples distinct


def test_mig_rewards_a_relevant_unique_unit():
    rel = torch.tensor([1.0, 1.0, 1.0])
    sim = torch.tensor([[1.0, 0.95, 0.0],   # 0 and 1 are redundant twins
                        [0.95, 1.0, 0.0],
                        [0.0, 0.0, 1.0]])
    mig = cur.mig_scores(rel, sim)
    assert torch.allclose(mig[0], mig[1])          # twins are symmetric
    assert mig[0] < mig[2]                          # unique wins at equal relevance
    assert abs(float(mig[0]) - (1.0 - 0.95 / 2)) < 1e-6


def test_tokenize_chunkify_tfidf_shapes():
    toks = cur.tokenize("The cat sat. The cat ran!")
    assert "cat" in toks and "." in toks
    chunks = cur.chunkify(toks, 3)
    assert all(len(c) == 3 for c in chunks)
    vecs = cur.tfidf_vectors(chunks)
    assert vecs.shape[0] == len(chunks)
    assert vecs.shape[1] > 0


def test_cli_writes_a_report(tmp_path):
    # a vocabulary wider than the chunk: with every word in every chunk the
    # IDF is zero, the vectors are zero, and there is no valid density matrix
    words = [f"w{chr(97 + a)}{chr(97 + b)}" for a in range(8) for b in range(5)]
    text = " ".join(words[i % 40] for i in range(600))
    src = tmp_path / "corpus.txt"
    src.write_text(text)
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(MOE / "curation.py"), str(src),
         "--chunk", "32", "--query", "waa wab", "--out", str(out)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    rep = json.loads(out.read_text())
    assert rep["n_chunks"] == 18
    assert "spectral" in rep and "bipartite_mi" in rep
    assert len(rep["bipartite_mi"]) == 4
    assert len(rep["mig_top"]) == 10

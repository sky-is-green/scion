"""Tests for ``moe/gate_diff.py`` (pure JSON analysis, no torch)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

MOE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOE))

import gate_diff  # noqa: E402


def _gate(mean: float, worst: list[tuple[int, int]]) -> dict:
    return {
        "n_tokens": 4088,
        "mean": mean, "p99": mean + 1, "max": mean + 2,
        "student_entropy_nats": 8.0, "student_top1_prob": 0.1,
        "sharper_than_teacher": True,
        "worst_tokens": [{"token": 1, "window": w, "pos": p, "kld": mean}
                         for w, p in worst],
    }


def _write(tmp_path: Path, name: str, d: dict) -> str:
    p = tmp_path / f"kld-{name}.json"
    p.write_text(json.dumps(d))
    return str(p)


def test_persistence_and_new_spikes(tmp_path):
    shared = [(2, 183), (6, 181)]
    base = _gate(1.0, shared + [(2, 187)])
    # candidate keeps one shared token, fixes the other two, adds a new spike
    cand = _gate(0.5, [(2, 183), (7, 99)])
    out = gate_diff.main([_write(tmp_path, "armC", base),
                          _write(tmp_path, "kdw2", cand)])
    assert out == 0


def test_positions_reported_as_window_pos(tmp_path, capsys):
    base = _gate(1.0, [(2, 183), (6, 181)])
    cand = _gate(0.5, [(2, 183), (3, 501)])
    gate_diff.main([_write(tmp_path, "armC", base),
                    _write(tmp_path, "kdw2", cand)])
    text = capsys.readouterr().out
    assert "1/16 keep" in text
    assert "w2:183" in text          # the one shared worst token
    assert "w3:501" in text          # the new spike
    assert "w6:181" not in text      # fixed tokens are not listed as kept


def test_missing_worst_tokens_is_an_error(tmp_path):
    p = tmp_path / "kld-empty.json"
    p.write_text(json.dumps({"mean": 1.0}))
    try:
        gate_diff.load(str(p))
    except SystemExit as e:
        assert "worst_tokens" in str(e)
    else:
        raise AssertionError("expected SystemExit for a non-gate JSON")

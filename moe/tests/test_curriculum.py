"""CPU tests for the hard-window curriculum builder (no datasets/tokenizers)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

MOE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOE))

import curriculum as cur  # noqa: E402


def _gate(worst):
    return {"worst_tokens": [{"token": 1, "window": w, "pos": p, "kld": 9 - i}
                             for i, (w, p) in enumerate(worst)]}


def test_worst_window_ids_dedupes_worst_first():
    gate = _gate([(3, 1), (2, 5), (3, 9), (0, 2)])
    assert cur.worst_window_ids(gate, top=4) == [3, 2, 0]
    assert cur.worst_window_ids(gate, top=2) == [3, 2]


def test_worst_window_ids_rejects_non_gate_json():
    with pytest.raises(SystemExit):
        cur.worst_window_ids({"mean": 1.0}, top=4)


def test_build_rows_decodes_the_selected_windows():
    data = torch.arange(3 * 8).reshape(3, 8)
    rows = cur.build_rows(data, [2, 0], decode=lambda ids: f"win{ids[0]}")
    assert [r["text"] for r in rows] == ["win16", "win0"]


def test_build_rows_rejects_out_of_range_window():
    with pytest.raises(SystemExit):
        cur.build_rows(torch.zeros(2, 8), [5], decode=lambda ids: "x")


def test_gate_split_guard():
    assert cur.gate_split_is_contaminating("wikitext")
    assert cur.gate_split_is_contaminating("WikiText")
    assert not cur.gate_split_is_contaminating("fineweb")


def test_main_refuses_the_gate_split_before_loading_anything(tmp_path, capsys):
    gate = tmp_path / "kld-probe.json"
    gate.write_text(json.dumps(_gate([(0, 1)])))
    with pytest.raises(SystemExit) as e:
        cur.main([str(gate), "--split", "wikitext"])
    assert "contaminat" in str(e.value)


def test_main_writes_jsonl_with_force_and_injected_data(tmp_path, monkeypatch):
    """End-to-end with the dataset/tokenizer/loader stubbed out."""
    gate = tmp_path / "kld-probe.json"
    gate.write_text(json.dumps(_gate([(1, 4), (0, 7)])))
    out = tmp_path / "curric.jsonl"

    import types
    fake_tok = types.SimpleNamespace(decode=lambda ids: f"t{ids[0]}")
    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
                            from_pretrained=lambda *a, **k: fake_tok)))
    monkeypatch.setitem(sys.modules, "olmoe_proxy",
                        types.SimpleNamespace(windows=lambda *a, **k: torch.arange(2 * 8).reshape(2, 8)))
    monkeypatch.setitem(sys.modules, "qwen35_moe_proxy",
                        types.SimpleNamespace(MODEL="/nonexistent"))

    rc = cur.main([str(gate), "--split", "fineweb", "--top", "2", "--out", str(out)])
    assert rc == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["text"] for r in rows] == ["t8", "t0"]

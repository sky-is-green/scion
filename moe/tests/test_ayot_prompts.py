"""CPU tests for the AYOT prompt-set builder's pure filtering/cleaning pieces."""
import pytest

from build_ayot_prompts import COT_SUFFIX, FINEWEB_PROMPT, _fineweb_ok, clean


# ----------------------------------------------------------------- clean -----

def test_clean_preserves_line_structure_and_indentation():
    """The coding half is fenced Python; newlines AND leading indent must survive."""
    src = ("Write a convolution function:\n\n```python\ndef conv(a, b):\n"
           "    out = [0] * (len(a) + len(b) - 1)\n"
           "    for i, x in enumerate(a):\n"
           "        for j, y in enumerate(b):\n"
           "            out[i + j] += x * y\n"
           "    return out\n```\n\nExplain what the inner loop costs.")
    out = clean(src, 5000)
    assert out.startswith("Write a convolution function:\n\n```python")
    assert "\n    out = [0] * (len(a) + len(b) - 1)\n" in out
    assert "\n        for j, y in enumerate(b):\n" in out   # 8-space indent intact
    assert out.count("\n") == src.count("\n")


def test_clean_collapses_interior_runs_only():
    src = ("alpha    bravo\t\tcharlie\n\n\n\n\ndelta  echo foxtrot "
           "golf    hotel")
    out = clean(src, 5000)
    assert out == ("alpha bravo charlie\n\ndelta echo foxtrot golf hotel")


def test_clean_caps_blank_line_runs():
    src = "first paragraph of the passage\n\n\n\n\n\nsecond paragraph here"
    assert clean(src, 5000) == "first paragraph of the passage\n\nsecond paragraph here"


def test_clean_normalises_crlf():
    src = "first line of the passage\r\nsecond line of the passage"
    assert clean(src, 5000) == ("first line of the passage\n"
                                "second line of the passage")


def test_clean_length_floor():
    assert clean("x" * 39, 5000) is None
    assert clean("x" * 40, 5000) == "x" * 40


def test_clean_rejects_rubbish():
    assert clean("too short", 5000) is None
    assert clean("", 5000) is None
    assert clean(None, 5000) is None
    assert clean("   \n\n  ", 5000) is None


def test_clean_truncates_on_a_line_boundary():
    body = "\n".join(f"line {i} padding padding" for i in range(200))
    out = clean(body, 200)
    assert len(out) <= 200
    assert not out.endswith("padding paddi")   # no mid-word cut
    assert out == out.rstrip()


# ------------------------------------------------------------- fineweb ------

def test_fineweb_ok_requires_quality_and_length():
    assert _fineweb_ok({"int_score": 4, "token_count": 200})
    assert _fineweb_ok({"int_score": 5, "token_count": 120})
    assert not _fineweb_ok({"int_score": 3, "token_count": 500})
    assert not _fineweb_ok({"int_score": 5, "token_count": 10})
    # missing fields must be rejected, not crash
    assert not _fineweb_ok({})
    assert not _fineweb_ok({"int_score": None, "token_count": None})


# ------------------------------------------------------------ templates -----

def test_templates_interpolate_and_stay_balanced():
    assert "{passage}" in FINEWEB_PROMPT
    rendered = FINEWEB_PROMPT.format(passage="SOME PASSAGE")
    assert "SOME PASSAGE" in rendered
    assert "{" not in rendered and "}" not in rendered
    assert COT_SUFFIX.strip() and "{" not in COT_SUFFIX

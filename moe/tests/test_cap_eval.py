"""Tests for the capability harness (checkers, extraction, aggregation).

CPU-only, no model needed: hand-written good/bad completions are the positive
control.  The pipeline negative control (broken model scores the floor) is
the manual 4-layer run in the runlog.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import cap_eval
from cap_eval import check_completion, evaluate


def test_checkers_positive_and_negative():
    cases = [
        ({"type": "contains", "values": ["Paris"]}, "The answer is Paris.", True),
        ({"type": "contains", "values": ["Paris"]}, "Lyon is nice.", False),
        ({"type": "contains_any", "values": ["yes"]}, "YES!", True),
        ({"type": "contains_any", "values": ["no"]}, "YES!", False),
        ({"type": "exact", "value": "blue"}, "  BLUE\n", True),
        ({"type": "exact", "value": "blue"}, "blue green", False),
        ({"type": "numeric", "value": 391, "tol": 0.5}, "391", True),
        ({"type": "numeric", "value": 391, "tol": 0.5}, "answer: 391.0", True),
        ({"type": "numeric", "value": 391, "tol": 0.5}, "42", False),
        ({"type": "numeric", "value": 0.05, "tol": 0.005}, "0.05 dollars", True),
        ({"type": "regex", "pattern": r"1\s*,\s*2\s*,\s*3"}, "1, 2, 3", True),
        ({"type": "regex", "pattern": r"1\s*,\s*2\s*,\s*3"}, "1 2 3", False),
        ({"type": "python_output", "value": "42"}, "print(42)", True),
        ({"type": "python_output", "value": "42"}, "```python\nprint(41)\n```", False),
        ({"type": "python_output", "value": "42"}, "print(1/0)", False),
        ({"type": "python_value", "value": 42}, "20 + 22", True),
        ({"type": "python_value", "value": 42}, "21 + 22", False),
    ]
    for check, completion, want in cases:
        passed, _ = check_completion(check, completion)
        assert passed == want, (check, completion)
    with __import__("pytest").raises(ValueError):
        check_completion({"type": "nope"}, "x")


def test_tasks_file_valid():
    doc = json.loads((Path(cap_eval.__file__).parent / "cap_tasks.json").read_text())
    tasks = doc["tasks"]
    assert len(tasks) == 20
    ids = [t["id"] for t in tasks]
    assert len(set(ids)) == len(ids)
    cats = {t["category"] for t in tasks}
    assert {"factual", "math", "instruction", "code", "logic"} <= cats
    for t in tasks:
        assert t["prompt"] and t["check"]["type"]


def test_server_backend_runs_against_fake_http_server():
    """Single-load server path: health wait + /completion parsing."""
    import http.server
    import json as _json
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep pytest output clean
            pass

        def _send(self, obj):
            data = _json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send({"status": "ok"})
            else:
                self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = _json.loads(self.rfile.read(n))
            assert self.path == "/completion"
            assert req["temperature"] == 0.0
            self._send({"content": "yes" if "Paris" in req["prompt"] else "5"})

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        cap_eval.wait_health(url, timeout=5)
        tasks = [
            {"id": "a", "category": "factual", "prompt": "capital of Paris?",
             "check": {"type": "exact", "value": "yes"}},
            {"id": "b", "category": "math", "prompt": "2+3",
             "check": {"type": "numeric", "value": 5, "tol": 0.5}},
        ]
        rep = cap_eval.evaluate("", "model", tasks, 8, 30,
                                backend="server", server_url=url)
        assert rep["backend"] == "server"
        assert rep["ok"] == 2 and rep["accuracy"] == 1.0
        assert rep["results"][0]["completion"] == "yes"
        with __import__("pytest").raises(TimeoutError):
            cap_eval.wait_health("http://127.0.0.1:1", timeout=0.2)
    finally:
        srv.shutdown()
        srv.server_close()


def test_evaluate_aggregates_with_stubbed_backend():
    tasks = [
        {"id": "a", "category": "factual", "prompt": "p",
         "check": {"type": "exact", "value": "yes"}},
        {"id": "b", "category": "math", "prompt": "p",
         "check": {"type": "numeric", "value": 4, "tol": 0.5}},
    ]
    with patch.object(cap_eval, "run_task", side_effect=["yes", "5"]):
        rep = evaluate("cli", "model", tasks, 40, 600)
    assert rep["n"] == 2 and rep["ok"] == 1
    assert rep["by_category"]["factual"]["acc"] == 1.0
    assert rep["by_category"]["math"]["acc"] == 0.0
    # a throwing backend fails the task, not the suite.
    with patch.object(cap_eval, "run_task", side_effect=RuntimeError("boom")):
        rep = evaluate("cli", "model", tasks, 40, 600)
    assert rep["ok"] == 0 and all(r["error"] for r in rep["results"])

#!/usr/bin/env python3
"""Build the v2 pilot rollout corpus (hard-90 and easy-30 excluded).

Sources (public, downloaded in this workspace):
  reasoning  30 AIME + 70 MATH-500
  coding     40 MBPP + 30 HumanEval + 30 LiveCodeBench
  tools      100 synthetic tool-use requests with OpenAI tool schemas

The heldoutlab hard-90 prompts and the 30-task test split of
`hivebench/experiments/cascade/tasks-bench.json` (the easy-30) are normalised
and removed by exact prompt match.  The corpus text lives under v2/corpus/
(gitignored); a manifest with source counts and a sha256 is tracked.

Usage:
  python v2/scripts/build_rollout_corpus.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
HIVE = WORK / "hivebench"
HELDOUT = Path("/tmp/opencode/bonsai2-27b-test")
BENCH = HIVE / "data" / "benches"

os.environ.setdefault("BONSAI_ROUND", "2")
for p in (str(HELDOUT / "harness"), str(HELDOUT / "items_r2")):
    if p not in sys.path:
        sys.path.insert(0, p)


def norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def exclusions() -> set[str]:
    out = set()
    import run
    items, _ = run.load_items()
    for _, (family, item) in items.items():
        if family in ("reasoning", "coding", "tools"):
            out.add(norm(item.get("prompt", "")))
    tasks = json.loads((HIVE / "experiments" / "cascade" /
                        "tasks-bench.json").read_text())
    for t in tasks["tasks"]:
        if t.get("split") == "test":
            out.add(norm(t.get("prompt", "")))
    return out


def add(rows, exclude, *, rid, family, prompt, source, tools=None,
        reference="", meta=None):
    if not prompt or norm(prompt) in exclude:
        return False
    row = {"id": rid, "family": family, "prompt": prompt,
           "source": source, "tools": tools or [], "reference": reference}
    if meta:
        row["meta"] = meta
    rows.append(row)
    return True


def build_reasoning(rows, exclude):
    import pyarrow.parquet as pq
    from datasets import load_dataset

    aime = pq.read_table(BENCH / "aime" / "data" /
                         "train-00000-of-00001.parquet").to_pylist()
    n = 0
    for r in aime:
        if add(rows, exclude, rid=f"PR-{n:03d}", family="reasoning",
               prompt=r["problem"], source="aime",
               reference=str(r.get("answer", "")),
               meta={"aime_id": r.get("id")}):
            n += 1
    math = load_dataset("HuggingFaceH4/MATH-500", split="test",
                        streaming=True)
    m = 0
    for r in math:
        if m >= 70:
            break
        if add(rows, exclude, rid=f"PR-{n:03d}", family="reasoning",
               prompt=r["problem"], source="math-500",
               reference=str(r.get("answer", "")),
               meta={"subject": r.get("subject"), "level": r.get("level")}):
            n += 1
            m += 1
    return n


def build_coding(rows, exclude):
    mbpp = [json.loads(line) for line in
            open(BENCH / "mbpp.jsonl", encoding="utf-8")]
    n = 0
    for r in mbpp[:80]:
        prompt = r["text"].strip()
        if add(rows, exclude, rid=f"PC-{n:03d}", family="coding",
               prompt=prompt + "\n\nWrite the complete function in Python.",
               source="mbpp", reference=r.get("code", ""),
               meta={"task_id": r.get("task_id")}):
            n += 1
        if n >= 40:
            break
    he = [json.loads(line) for line in
          gzip.open(BENCH / "humaneval.jsonl.gz", "rt", encoding="utf-8")]
    m = 0
    for r in he:
        if add(rows, exclude, rid=f"PC-{n:03d}", family="coding",
               prompt=("Complete the following Python function.  Return only "
                       "the complete function.\n\n```python\n" +
                       r["prompt"] + "```"),
               source="humaneval", reference=r.get("canonical_solution", ""),
               meta={"task_id": r.get("task_id")}):
            n += 1
            m += 1
        if m >= 30:
            break
    lcb_path = BENCH / "livecodebench" / "test.jsonl"
    lcb = 0
    with open(lcb_path, encoding="utf-8") as f:
        for line in f:
            if lcb >= 30:
                break
            r = json.loads(line)
            if r.get("difficulty") not in ("medium", "hard"):
                continue
            prompt = (r["question_content"].strip() +
                      "\n\nWrite a complete Python solution that reads from "
                      "stdin and writes to stdout.")
            if add(rows, exclude, rid=f"PC-{n:03d}", family="coding",
                   prompt=prompt, source="livecodebench",
                   reference="", meta={"question_id": r.get("question_id"),
                                       "difficulty": r.get("difficulty")}):
                n += 1
                lcb += 1
    return n


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "unit": {"type": "string",
                                                  "enum": ["c", "f"]}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_flights", "description": "Search flights",
        "parameters": {"type": "object", "properties": {
            "origin": {"type": "string"}, "destination": {"type": "string"},
            "date": {"type": "string"}}, "required": ["origin",
                                                       "destination",
                                                       "date"]}}},
    {"type": "function", "function": {
        "name": "create_event", "description": "Create a calendar event",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "start": {"type": "string"},
            "duration_minutes": {"type": "integer"},
            "timezone": {"type": "string"}},
            "required": ["title", "start", "duration_minutes"]}}},
    {"type": "function", "function": {
        "name": "send_email", "description": "Send an email",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "subject": {"type": "string"},
            "body": {"type": "string"}}, "required": ["to", "subject",
                                                       "body"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write a text file",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "sql_query", "description": "Run a read-only SQL query",
        "parameters": {"type": "object", "properties": {
            "database": {"type": "string"}, "query": {"type": "string"}},
            "required": ["database", "query"]}}},
    {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate arithmetic",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "http_get", "description": "Fetch a URL",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "create_ticket", "description": "Create a support ticket",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "priority": {"type": "string",
                                                       "enum": ["low", "medium", "high"]},
            "details": {"type": "string"}},
            "required": ["title", "priority", "details"]}}},
]

TOOL_REQUESTS = [
    "What is the weather in Paris right now?  Use Celsius.",
    "What is the weather in New York in Fahrenheit?",
    "Find flights from Lisbon to Berlin on 2026-12-03.",
    "Find the cheapest flight from Oslo to Rome on 2026-11-20; if the date is ambiguous ask me first.",
    "Schedule a 45-minute project sync tomorrow at 10:00 Europe/Berlin.",
    "Create a calendar event called 'Dentist' on 2026-11-18 at 08:30 for 30 minutes.",
    "Email ada@example.com a one-line note that the report is ready.",
    "Draft an email to the team about the Friday deploy freeze, subject 'Deploy freeze'.",
    "Read /etc/hosts and summarize the entries.",
    "Read the file notes/todo.txt and list every unchecked item.",
    "Write 'status: ok' to /tmp/health.txt.",
    "Append the current date to /tmp/audit.log.",
    "How many users signed up last month?  The database is analytics.",
    "List the five newest orders in the sales database.",
    "Compute 18% of 2,450 and then add 37.5.",
    "Convert 72 Fahrenheit to Celsius using the calculator tool.",
    "Fetch https://example.com/status and report the HTTP status.",
    "Fetch the JSON at https://api.example.com/v1/health and extract the 'ok' field.",
    "Open a high-priority ticket: production checkout returns 500 for some users.",
    "Create a low-priority ticket for a typo in the docs.",
    "Which is warmer today, Madrid or Vienna?  Use metric units.",
    "I need to be in Zurich by 09:00 on 2026-12-01; find a flight from London.",
    "Move my 14:00 meeting to 16:00 on 2026-11-13; it is 30 minutes.",
    "Schedule a 2-hour workshop next Tuesday at 13:00 UTC.",
    "Send the quarterly numbers to finance@example.com and cc legal@example.com.",
    "Email me a reminder to renew the domain, subject 'Renewal'.",
    "Show me the contents of /var/log/app.log, last 20 lines only.",
    "Create /tmp/run.sh with a shebang and the command 'echo ready'.",
    "Count the rows in the events table for 2026-11-01.",
    "Find duplicate emails in the users table of the crm database.",
    "What is 2**32 divided by 1024?",
    "Compute the compound interest on 10,000 at 4.5% for 3 years.",
    "Check whether https://example.com/health returns 200.",
    "Fetch the page at https://example.com and tell me its title.",
    "Open a medium-priority ticket: the export button is slow for large accounts.",
    "Create a high-priority ticket for a data loss report and include the account id 8842.",
    "What is the weather in Tokyo?  If the city is ambiguous, ask which Tokyo.",
    "Find a flight from Paris to Montreal on 2026-12-24, preferably direct.",
    "Reschedule the 09:30 standup to 10:00 on 2026-11-17, still 15 minutes.",
    "Create an event 'Release' on 2026-12-10 at 18:00 UTC for 90 minutes.",
    "Send a short thank-you email to sam@example.com.",
    "Email the on-call list about the incident, subject 'Incident resolved'.",
    "Read /etc/os-release and report the distribution name.",
    "Write a JSON object {\"ok\": true} to /tmp/status.json.",
    "How many orders were refunded in the last 7 days?",
    "List the columns of the invoices table in the billing database.",
    "Calculate 1,234 * 5,678 and then take the square root.",
    "Convert 1.5 miles to kilometers using the calculator.",
    "Fetch https://example.com/robots.txt and summarize it.",
    "Fetch https://api.example.com/v1/version and report the version string.",
    "Create a high-priority ticket: login fails with SSO for one tenant.",
    "Open a low-priority ticket for a broken link in the help center.",
    "Compare the weather in Reykjavik and Helsinki in Celsius.",
    "Find a flight from Dublin to Barcelona on 2026-11-28; ask if the airport is ambiguous.",
    "Book a 30-minute '1:1' on 2026-11-19 at 15:00 Europe/London.",
    "Create an all-day event 'Holiday' on 2026-12-25.",
    "Email ops@example.com the subject 'Backup complete' and a one-line body.",
    "Send a reminder to the design team about the review, subject 'Design review'.",
    "Show the first 40 lines of /var/log/syslog.",
    "Create /tmp/hello.txt containing 'hello world'.",
    "Query the number of active subscriptions in the billing database.",
    "Find the top three products by revenue in the sales database.",
    "What is 15% of 3,200?",
    "Convert 0 Celsius to Fahrenheit.",
    "Check the status of https://example.com/health.",
    "Fetch https://example.com/version and report the build number.",
    "Create a medium-priority ticket for slow search results.",
    "Open a high-priority ticket for a payment webhook failure.",
    "What is the weather in Sydney in Celsius?",
    "Find a flight from Amsterdam to New York on 2026-12-15.",
    "Move the 11:00 design review to 15:30 on 2026-11-14, 45 minutes.",
    "Schedule a 60-minute planning session on 2026-11-21 at 09:00 UTC.",
    "Send an email to hr@example.com with the subject 'Policy update'.",
    "Draft a short email to the vendor asking for a revised quote.",
    "Read /proc/loadavg and report the one-minute load.",
    "Write the current timestamp to /tmp/timestamp.txt.",
    "How many sessions started on 2026-11-10 in the analytics database?",
    "List the slowest five queries from the ops database.",
    "Compute 99 * 99 * 99.",
    "Convert 100 kilometers to miles.",
    "Fetch https://example.com/api/status and report whether it is healthy.",
    "Fetch https://example.com/terms and count the paragraphs.",
    "Create a low-priority ticket for a missing favicon.",
    "Open a high-priority ticket for a broken checkout flow in the EU region.",
    "Which city is colder right now, Anchorage or Oslo?",
    "Find a direct flight from Madrid to Mexico City on 2026-12-20.",
    "Create a 25-minute 'Coffee chat' on 2026-11-16 at 16:00 Europe/Paris.",
    "Schedule a 2-hour incident review on 2026-11-22 at 14:00 UTC.",
    "Email the security team about the completed rotation, subject 'Key rotation'.",
    "Send a one-line status update to the project alias.",
    "Read /tmp/config.json and report the 'port' value.",
    "Write 'enabled' to /tmp/feature.flag.",
    "Count distinct users in the events table for 2026-11-01.",
    "Find the average order value in the sales database.",
    "Calculate 7! / 5!.",
    "Convert 32 Fahrenheit to Celsius and then to Kelvin.",
    "Fetch https://example.com/metrics and report the first metric name.",
    "Fetch https://example.com/healthz and report the status code.",
    "Create a high-priority ticket for a data pipeline delay.",
    "Open a medium-priority ticket for an intermittent 502 on the dashboard.",
]


def build_tools(rows, exclude):
    n = 0
    for i, req in enumerate(TOOL_REQUESTS):
        if n >= 100:
            break
        # a deterministic subset of schemas per request, always including the
        # one the request needs; the model must choose.
        tools = [TOOL_SCHEMAS[i % len(TOOL_SCHEMAS)],
                 TOOL_SCHEMAS[(i * 3 + 1) % len(TOOL_SCHEMAS)],
                 TOOL_SCHEMAS[(i * 7 + 2) % len(TOOL_SCHEMAS)]]
        # de-duplicate by function name
        seen, uniq = set(), []
        for t in tools:
            if t["function"]["name"] not in seen:
                seen.add(t["function"]["name"])
                uniq.append(t)
        if add(rows, exclude, rid=f"PT-{n:03d}", family="tools",
               prompt=req, source="synthetic-tools", tools=uniq):
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "v2/corpus/rollout-prompts.jsonl"))
    ap.add_argument("--manifest", default=str(REPO / "v2/rollout-corpus-manifest.json"))
    args = ap.parse_args()

    exclude = exclusions()
    rows = []
    nr = build_reasoning(rows, exclude)
    nc = build_coding(rows, exclude)
    nt = build_tools(rows, exclude)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "prompts": str(out.relative_to(REPO)),
        "sha256": digest,
        "counts": {"reasoning": nr, "coding": nc, "tools": nt,
                   "total": len(rows)},
        "sources": {
            "reasoning": ["aime (2022-2024 parquet)",
                          "HuggingFaceH4/MATH-500 test"],
            "coding": ["mbpp.jsonl", "humaneval.jsonl.gz",
                        "livecodebench/test.jsonl (medium+hard)"],
            "tools": ["synthetic-tools (this script)"],
        },
        "excluded": {
            "heldoutlab hard-90 prompts": "exact normalised match",
            "hivebench cascade tasks-bench test split (easy-30)":
                "exact normalised match",
        },
        "exclusion_set_size": len(exclude),
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

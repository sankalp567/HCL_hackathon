"""Grade the team's test set (eval/test_queries.csv) against a running backend.

    python eval/run_test_queries.py                      # backend at http://127.0.0.1:8765
    python eval/run_test_queries.py --url http://127.0.0.1:8000

Columns of the CSV: company_id (tcs | infy | hcl, several joined with '|'), query, value, page.
value: a number, 'company:number|...' for a comparison, or a company id for a 'which company' question.
page:  a PDF page, or 'company:page|...'. Pages are informational: one number can be printed on several pages.

A question is right when the answer contains every expected number (12.5 matches 12.51; crore values exact), or,
for 'which company', when the answer names only the expected company in its first or last sentence.
Writes eval/test_queries_results.md and .jsonl. Answers are asked fresh (use_cache: false).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
NAMES = {"tcs": "TCS", "infy": "Infosys", "hcl": "HCLTech"}
# years, citation marks and dates are not answer numbers
NOT_A_VALUE = re.compile(r"\bFY\s*'?\d{2,4}(?:\s*[-–]\s*\d{2,4})?|\b20\d{2}(?:\s*[-–]\s*\d{2,4})?\b|\[\d+\]|March 31,?", re.I)
COMPUTED = re.compile(r"combined|how much higher|gap between", re.I)


def ask(client: httpx.Client, url: str, question: str) -> dict:
    out: dict = {"final": None, "citations": [], "done": {}, "notices": []}
    with client.stream("POST", f"{url}/api/ask", json={"question": question, "use_cache": False}, timeout=180) as r:
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "final":
                    out["final"] = data
                elif event == "citations":
                    out["citations"] = data["citations"]
                elif event == "done":
                    out["done"] = data
                elif event == "notice":
                    out["notices"].append(data["message"])
    return out


def numbers(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", NOT_A_VALUE.sub(" ", text))]


def has(text: str, value: str) -> bool:
    v = float(value)
    tolerance = 0.051 if ("." in value or v < 100) else 0.5
    return any(abs(n - v) <= tolerance for n in numbers(text))


def split(field: str) -> dict[str, str] | None:
    return dict(p.split(":", 1) for p in field.split("|")) if ":" in field else None


def names_only(sentence: str, expected: str) -> bool:
    s = sentence.lower()
    return NAMES[expected].lower() in s and not any(n.lower() in s for k, n in NAMES.items() if k != expected)


def grade(row: dict, answer: str, status: str) -> tuple[str, bool]:
    value, multi = row["value"], split(row["value"])
    if multi:
        return "compare", status == "answered" and all(has(answer, v) for v in multi.values())
    if value in NAMES:
        sentences = [s for s in re.split(r"(?<=[.!?])\s|\n", answer.strip()) if s.strip()]
        return "which", status == "answered" and bool(sentences) and (
            names_only(sentences[0], value) or names_only(sentences[-1], value))
    kind = "computed" if COMPUTED.search(row["query"]) else "single"
    return kind, status == "answered" and has(answer, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--csv", type=Path, default=HERE / "test_queries.csv")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    results = []
    with httpx.Client() as client:
        for i, row in enumerate(rows, 1):
            r = ask(client, args.url, row["query"])
            status = r["done"].get("status", "error")
            answer = (r["final"] or {}).get("answer", "") if status == "answered" else ""
            kind, ok = grade(row, answer, status)
            cited = {(c["company"], c["page"]) for c in r["citations"]}
            pages = split(row["page"]) or {row["company_id"]: row["page"]}
            page_hit = all((NAMES[k], int(p)) in cited for k, p in pages.items() if k in NAMES)
            results.append({"n": i, "kind": kind, "query": row["query"], "expected": row["value"], "status": status,
                            "ok": ok, "page_hit": page_hit, "backend": r["done"].get("backend"),
                            "model": r["done"].get("model"), "latency_ms": r["done"].get("latency_ms"),
                            "answer": answer, "message": r["done"].get("message", ""), "cited": sorted(cited)})
            print(f"{i:2d} {'OK ' if ok else 'BAD'} {kind:8s} {status:13s} {row['query'][:70]}", flush=True)

    lines = ["# Team test set: results", "",
             f"Backend: `{args.url}` · questions: {len(results)} · run at {time.strftime('%Y-%m-%d %H:%M')}", "",
             "| Kind | Right | Expected page cited |", "|---|---|---|"]
    for kind in ("single", "compare", "which", "computed"):
        sel = [r for r in results if r["kind"] == kind]
        if sel:
            lines.append(f"| {kind} | {sum(r['ok'] for r in sel)}/{len(sel)} | {sum(r['page_hit'] for r in sel)}/{len(sel)} |")
    lines += [f"| **all** | **{sum(r['ok'] for r in results)}/{len(results)}** | {sum(r['page_hit'] for r in results)}/{len(results)} |",
              "", "| # | Kind | Question | Expected | Result | Answer |", "|---|---|---|---|---|---|"]
    for r in results:
        shown = (r["answer"] or r["message"]).replace("\n", " / ").replace("|", "\\|")[:160]
        lines.append(f"| {r['n']} | {r['kind']} | {r['query']} | {r['expected']} | {'✓' if r['ok'] else '✗'} {r['status']} | {shown} |")
    (HERE / "test_queries_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "test_queries_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8")
    print("\n".join(lines[4:4 + 2 + 5]))


if __name__ == "__main__":
    main()

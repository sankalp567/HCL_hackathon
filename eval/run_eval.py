"""Run the question set against a running backend and score answers and citations.

    python eval/run_eval.py                          # backend at http://127.0.0.1:8765, writes eval/results.md
    python eval/run_eval.py --url http://127.0.0.1:8000

Scores, per question type:
- status:    answered / not_found / out_of_scope as expected
- numbers:   every expected number (any listed spelling) appears in the final answer
- citations: for every sentence with numbers, each number is on one of the PDF pages the sentence cites,
             checked against the page's own text layer (pages table), not against our extracted facts
- terms:     a descriptive answer mentions the expected term
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MARK = re.compile(r"\[(\d+)\]")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
YEAR_EXPR = re.compile(r"\bFY\s*'?\d{2,4}(?:\s*[-–]\s*\d{2,4})?|\b(?:19|20)\d{2}(?:\s*[-–]\s*\d{2,4})?\b", re.I)


def norm(number: str) -> str:
    n = number.replace(",", "")
    return n.rstrip("0").rstrip(".") if "." in n else n


def digits_of(text: str) -> str:
    """Page text with spaces and commas removed, so '6, 17 437' and '1,30,144' match their numbers."""
    return re.sub(r"[\s,]", "", text)


def ask(client: httpx.Client, url: str, question: str) -> dict:
    result = {"tokens": "", "final": None, "citations": [], "done": None, "meta": None}
    started = time.monotonic()
    with client.stream("POST", f"{url}/api/ask", json={"question": question, "use_cache": False}, timeout=180) as response:
        event = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "token":
                    result["tokens"] += data["t"]
                elif event in ("final", "done", "meta"):
                    result[event] = data
                elif event == "citations":
                    result["citations"] = data["citations"]
    result["seconds"] = time.monotonic() - started
    return result


def check_citations(result: dict, page_text: dict) -> tuple[int, int, list[str]]:
    """(numeric claims checked, claims whose numbers are on a cited page, problems)."""
    by_n = {c["n"]: c for c in result["citations"]}
    sentences = (result["final"] or {}).get("sentences", [])
    checked = ok = 0
    problems = []
    for s in sentences:
        if s["status"] == "removed":
            continue
        numbers = {norm(m) for m in NUMBER.findall(YEAR_EXPR.sub(" ", MARK.sub(" ", s["text"])))}
        cited = [by_n[int(n)] for n in MARK.findall(s["text"]) if int(n) in by_n]
        if not numbers or not cited:
            continue
        checked += 1
        pages = "".join(digits_of(page_text.get((c["doc_id"], c["page"]), "")) for c in cited)
        missing = [n for n in numbers if n.replace(".", "") and n not in pages and n.split(".")[0] + "." + (n.split(".")[1] if "." in n else "") not in pages]
        if not missing:
            ok += 1
        else:
            problems.append(f"{s['text'][:80]}… → {', '.join(missing)} not on p. {[c['page'] for c in cited]}")
    return checked, ok, problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "facts.db")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    page_text = {(d, p): t for d, p, t in conn.execute("SELECT doc_id, page_no, text FROM pages")}
    questions = [json.loads(line) for line in (HERE / "questions.jsonl").read_text().splitlines() if line.strip()]
    rows, records = [], []
    totals = {"status": 0, "numbers": [0, 0], "terms": [0, 0], "claims": [0, 0]}
    latencies = []
    with httpx.Client() as client:
        for q in questions:
            r = ask(client, args.url, q["question"])
            status = (r["done"] or {}).get("status")
            answer = (r["final"] or {}).get("answer", "")
            status_ok = status == q["expect"]
            totals["status"] += status_ok
            numbers_ok = terms_ok = None
            if q.get("numbers"):
                found = {norm(m) for m in NUMBER.findall(answer)}
                numbers_ok = all(any(norm(alt) in found for alt in group) for group in q["numbers"])
                totals["numbers"][0] += 1
                totals["numbers"][1] += numbers_ok
            if q.get("terms"):
                terms_ok = any(t.lower() in answer.lower() for t in q["terms"])
                totals["terms"][0] += 1
                totals["terms"][1] += terms_ok
            checked, ok, problems = check_citations(r, page_text)
            totals["claims"][0] += checked
            totals["claims"][1] += ok
            latencies.append(r["seconds"])
            backend = (r["done"] or {}).get("backend")
            mark = lambda v: "–" if v is None else ("✓" if v else "✗")  # noqa: E731
            rows.append(f"| {q['id']} | {q['type']} | {q['question']} | {status} {mark(status_ok)} | {mark(numbers_ok)} "
                        f"| {mark(terms_ok)} | {ok}/{checked} | {backend} | {r['seconds']:.1f} |")
            records.append({"id": q["id"], "question": q["question"], "status": status, "status_ok": status_ok,
                            "numbers_ok": numbers_ok, "terms_ok": terms_ok, "claims_checked": checked, "claims_ok": ok,
                            "problems": problems, "answer": answer, "backend": backend, "seconds": round(r["seconds"], 2)})
            print(f"{q['id']} {status:13} status {mark(status_ok)} numbers {mark(numbers_ok)} terms {mark(terms_ok)} "
                  f"citations {ok}/{checked} {backend} {r['seconds']:.1f}s", *problems, sep="\n    ")

    n = len(questions)
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[max(0, int(round(0.95 * n)) - 1)]
    pct = lambda a, b: f"{a}/{b} ({100 * a / b:.0f}%)" if b else "–"  # noqa: E731
    summary = [
        "# Evaluation results", "",
        f"Backend: `{args.url}` · questions: {n} · run at {time.strftime('%Y-%m-%d %H:%M')}", "",
        "| Score | Result |", "|---|---|",
        f"| Expected outcome (answered / not found / out of scope) | {pct(totals['status'], n)} |",
        f"| Numeric answers with the expected numbers | {pct(*reversed(totals['numbers']))} |",
        f"| Descriptive answers with the expected term | {pct(*reversed(totals['terms']))} |",
        f"| Numeric claims whose numbers are on a cited PDF page | {pct(*reversed(totals['claims']))} |",
        f"| Latency p50 / p95 | {p50:.1f} s / {p95:.1f} s |", "",
        "Citations are checked against each page's own text layer, independent of facts.db.", "",
        "| ID | Type | Question | Status | Numbers | Terms | Cited numbers on page | Answered by | Seconds |",
        "|---|---|---|---|---|---|---|---|---|", *rows,
    ]
    (HERE / "results.md").write_text("\n".join(summary) + "\n")
    (HERE / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    print("\n".join(summary[4:11]))


if __name__ == "__main__":
    main()

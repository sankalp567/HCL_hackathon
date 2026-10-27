"""Backend tests on a small fixture facts.db built from extract/schema.sql (no GPU, no network)."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import ROOT, get_settings
from app.llm import OpenAICompatLLM
from app.main import create_app
from app.qa import parse_citations

PASSAGES = [
    ("Infosys_FY2025-26", "Infosys", 1, "text", "Talent",
     "Voluntary attrition was 14.1% for fiscal 2026, down from 14.6% a year ago. We hired 15,000 graduates."),
    ("TCS_FY2025-26", "TCS", 1, "text", "Financial capital",
     "TCS revenue from operations grew 4.6% in FY 2026 on the back of large deals in North America."),
    ("HCLTech_FY2025-26", "HCLTech", 1, "text", "AI",
     "HCLTech increased its GenAI investments through AI Force and AI Foundry, training 1,50,000 employees."),
]


def build_fixture(root: Path) -> Path:
    """facts.db with three documents, one fact, three passages; a one-page Infosys PDF in root/reports."""
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Voluntary attrition was 14.1% for fiscal 2026.")
    pdf.save(reports / "Infosys_FY2025-26.pdf")
    db = root / "facts.db"
    conn = sqlite3.connect(db)
    conn.executescript((ROOT / "extract" / "schema.sql").read_text())
    for doc_id, company, title in [("TCS_FY2025-26", "TCS", "Annual Report 2025-26"),
                                   ("Infosys_FY2025-26", "Infosys", "Integrated Annual Report 2025-26"),
                                   ("HCLTech_FY2025-26", "HCLTech", "Integrated Annual Report 2025-26")]:
        conn.execute("INSERT INTO documents VALUES (?, ?, ?, 'FY2025-26', ?, 'x', 1, NULL, '2026-10-27')",
                     (doc_id, company, title, f"{doc_id}.pdf"))
    conn.execute(
        """INSERT INTO facts (company, metric, fy, value, unit, basis, label_in_report, source_kind, source_rank,
                              element_id, doc_id, page_no, x0, y0, x1, y1, quote, in_text_layer, agrees_with, status, extractor)
           VALUES ('Infosys', 'attrition', 'FY2025-26', 14.1, '%', 'consolidated', 'Voluntary attrition', 'table', 3,
                   'Infosys_FY2025-26#/tables/1', 'Infosys_FY2025-26', 1, 72, 60, 300, 76,
                   'Voluntary attrition | FY 2026: 14.1%', 1, 1, 'corroborated', 'fixture')""")
    conn.executemany("INSERT INTO passages (doc_id, company, page_no, kind, section, text) VALUES (?, ?, ?, ?, ?, ?)",
                     PASSAGES)
    conn.execute("INSERT INTO passages_fts(passages_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()
    return db


@pytest.fixture()
def settings(tmp_path):
    db = build_fixture(tmp_path)
    return replace(get_settings(), facts_db=db, reports_dir=tmp_path / "reports", log_dir=tmp_path / "logs",
                   llm_backend="extractive")


def events_of(response) -> list[tuple[str, dict]]:
    out = []
    for block in response.text.split("\n\n"):
        if not block.strip():
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def ask(client, question):
    response = client.post("/api/ask", json={"question": question})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return events_of(response)


def done_of(events):
    assert events[-1][0] == "done"
    return events[-1][1]


class FakeLLM:
    name = "fake"

    def __init__(self, pieces):
        self.pieces = pieces

    def stream(self, system, user):
        yield from self.pieces


def test_out_of_scope_streams_no_tokens(settings):
    events = ask(TestClient(create_app(settings)), "Should I buy Infosys shares?")
    assert done_of(events)["status"] == "out_of_scope"
    assert not [e for e, _ in events if e == "token"]


def test_nothing_matching_is_not_found(settings):
    done = done_of(ask(TestClient(create_app(settings)), "What colour is the moon?"))
    assert (done["status"], done["gate"]) == ("not_found", "evidence")


def test_future_year_is_not_found(settings):
    done = done_of(ask(TestClient(create_app(settings)), "What was TCS's revenue in FY2029-30?"))
    assert (done["status"], done["gate"]) == ("not_found", "fy")


def test_answerable_streams_tokens_then_cited_pages(settings):
    events = ask(TestClient(create_app(settings)), "What was Infosys's voluntary attrition in FY2025-26?")
    kinds = [e for e, _ in events]
    assert kinds[0] == "meta" and "token" in kinds and kinds[-2:] == ["citations", "done"]
    assert kinds.index("token") < kinds.index("citations")
    answer = "".join(d["t"] for e, d in events if e == "token")
    assert "14.1%" in answer
    citations = events[-2][1]["citations"]
    assert all(c["page"] == 1 and c["company"] == "Infosys" and c["pdf"] == "Infosys_FY2025-26.pdf" for c in citations)
    assert all(c["verified"] for c in citations)
    assert done_of(events)["status"] == "answered"
    assert (settings.log_dir / "qa.jsonl").read_text().count("\n") == 1


def test_model_not_found_is_never_shown(settings):
    client = TestClient(create_app(settings, llm=FakeLLM(["NOT", "_FO", "UND"])))
    events = ask(client, "What was Infosys's voluntary attrition in FY2025-26?")
    assert not [e for e, _ in events if e == "token"]
    assert (done_of(events)["status"], done_of(events)["gate"]) == ("not_found", "model")


def test_answer_without_citation_is_not_found(settings):
    client = TestClient(create_app(settings, llm=FakeLLM(["Attrition was low this year, says the report."])))
    done = done_of(ask(client, "What was Infosys's voluntary attrition in FY2025-26?"))
    assert (done["status"], done["gate"]) == ("not_found", "no_citation")


def test_parse_citations_maps_markers_and_checks_numbers():
    sources = [
        {"n": 1, "company": "TCS", "report": "AR", "pdf": "t.pdf", "doc_id": "t", "page": 5, "kind": "text",
         "text": "TCS had 6,07,354 employees at the end of FY 2026.", "bbox": None},
        {"n": 2, "company": "Infosys", "report": "AR", "pdf": "i.pdf", "doc_id": "i", "page": 9, "kind": "fact",
         "text": "Infosys headcount FY2025-26: 3,10,000", "bbox": [1, 2, 3, 4]},
    ]
    answer = "TCS: 6,07,354 employees in FY2025-26 [1].\nInfosys: 3,17,788 employees [2]. Unknown [7]."
    cites = {c["n"]: c for c in parse_citations(answer, sources)}
    assert set(cites) == {1, 2}                      # [7] has no source and is dropped
    assert cites[1]["page"] == 5 and cites[1]["verified"] is True
    assert cites[2]["bbox"] == [1, 2, 3, 4] and cites[2]["verified"] is False   # 3,17,788 is not in source 2


def test_vllm_backend_streams_chunks(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True and body["temperature"] == 0 and body["model"] == "qwen3-4b"
        assert request.headers["authorization"] == "Bearer secret"
        chunks = [{"choices": [{"delta": {"content": "Voluntary attrition was 14.1% "}}]},
                  {"choices": [{"delta": {"content": "in FY2025-26 [1]."}}]}]
        sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse.encode())

    llm = OpenAICompatLLM("http://ramanujan:8001/v1", "qwen3-4b", api_key="secret",
                          transport=httpx.MockTransport(handler))
    events = ask(TestClient(create_app(settings, llm=llm)), "What was Infosys's voluntary attrition in FY2025-26?")
    answer = "".join(d["t"] for e, d in events if e == "token")
    assert answer == "Voluntary attrition was 14.1% in FY2025-26 [1]."
    assert done_of(events)["status"] == "answered" and done_of(events)["backend"] == "vllm"


def test_vllm_unreachable_falls_back_to_extractive(settings):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    llm = OpenAICompatLLM("http://127.0.0.1:8001/v1", "qwen3-4b", transport=httpx.MockTransport(refuse))
    events = ask(TestClient(create_app(settings, llm=llm)), "What was Infosys's voluntary attrition in FY2025-26?")
    assert [d for e, d in events if e == "notice"]
    assert [e for e, _ in events if e == "token"]
    assert (done_of(events)["status"], done_of(events)["backend"]) == ("answered", "extractive")


def test_page_image_and_pdf_are_served_only_for_known_reports(settings):
    client = TestClient(create_app(settings))
    png = client.get("/api/page/Infosys_FY2025-26/1.png?x0=72&y0=60&x1=300&y1=76")
    assert png.status_code == 200 and png.content.startswith(b"\x89PNG")
    assert client.get("/api/page/Unknown/1.png").status_code == 404
    assert client.get("/api/page/Infosys_FY2025-26/5.png").status_code == 404
    assert client.get("/api/page/TCS_FY2025-26/1.png").status_code == 404      # listed, but no file on disk
    assert client.get("/reports/Infosys_FY2025-26.pdf").headers["content-type"] == "application/pdf"
    assert client.get("/reports/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_metrics_health_and_frontend(settings):
    client = TestClient(create_app(settings))
    facts = client.get("/api/metrics?metric=attrition").json()["facts"]
    assert facts[0]["company"] == "Infosys" and facts[0]["page_no"] == 1
    assert client.get("/api/health").json()["facts"] == 1
    home = client.get("/")
    assert home.status_code == 200 and "annual report" in home.text.lower()


def test_overview_facts_evidence_and_search(settings):
    client = TestClient(create_app(settings))
    overview = client.get("/api/overview").json()
    assert overview["companies"] == ["TCS", "Infosys", "HCLTech"] and overview["years"] == ["FY2025-26"]
    facts = client.get("/api/facts?company=Infosys").json()["facts"]
    assert facts[0]["is_best"] is True and facts[0]["bbox"] == [72, 60, 300, 76]
    card = client.get(f"/api/evidence?fact_id={facts[0]['fact_id']}").json()
    assert card["page"] == 1 and card["pdf"] == "Infosys_FY2025-26.pdf" and "14.1" in card["quote"]
    assert client.get("/api/evidence").status_code == 400
    assert client.get("/api/evidence?fact_id=999").status_code == 404
    results = client.get("/api/search?q=voluntary attrition").json()["results"]
    assert results[0]["company"] == "Infosys" and "[" in results[0]["snippet"]
    passage = client.get(f"/api/evidence?passage_id={results[0]['passage_id']}").json()
    assert passage["kind"] == "text" and passage["table"] is None


def test_company_toggle_scopes_only_questions_without_a_company(settings):
    client = TestClient(create_app(settings))
    scoped = client.post("/api/ask", json={"question": "What was voluntary attrition in FY2025-26?",
                                           "companies": ["Infosys"]})
    assert events_of(scoped)[0][1]["understood"]["companies"] == ["Infosys"]
    named = client.post("/api/ask", json={"question": "What was TCS revenue?", "companies": ["Infosys"]})
    assert events_of(named)[0][1]["understood"]["companies"] == ["TCS"]   # the question wins over the toggle

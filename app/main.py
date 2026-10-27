"""HTTP API and the static frontend.

Laptop, no GPU:   uvicorn app.main:app --port 8000                      (LLM_BACKEND=extractive, the default)
GPU server:       LLM_BACKEND=transformers uvicorn app.main:app --host 127.0.0.1 --port 8000
vLLM elsewhere:   LLM_BACKEND=vllm LLM_BASE_URL=http://127.0.0.1:8001/v1 uvicorn app.main:app --port 8000

SSE contract of POST /api/ask ({"question": str, "companies"?: [str]}), one JSON object per event:
  meta       {query_id, understood: {companies, metric, fy, named}, sources: [{n, company, report, pdf, doc_id, page, kind, score}]}
  token      {t}                      a piece of the answer text, with [n] citation markers
  final      {answer, sentences: [{text, status: supported|repaired|uncited|removed}]}
                                      the checked answer; the UI replaces the streamed text with it
  citations  {citations: [{n, company, report, pdf, doc_id, page, kind, quote, bbox, verified, fact_id, passage_id}]}
  notice     {message}                e.g. the model was unavailable and the extractive backend answered
  error      {message}
  done       {status: answered|not_found|out_of_scope|error, message, latency_ms, backend, gate}
"""
from __future__ import annotations

import json
import logging
import threading
from contextlib import closing

import pymupdf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ROOT, Settings, get_settings
from app.db import connect
from app.evidence import fact_evidence, passage_evidence, search_snippets
from app.llm import LLM, TransformersLLM, get_llm
from app.qa import answer_events

WEB_DIR = ROOT / "web"
log = logging.getLogger("app")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    companies: list[str] | None = None   # default scope when the question names no company (UI toggle)
    use_cache: bool = True               # False: answer afresh (the evaluation measures the pipeline)
    history: list[str] | None = None     # earlier questions of this chat, oldest first (follow-ups)


def with_bbox(row) -> dict:
    d = dict(row)
    box = [d.pop(k) for k in ("x0", "y0", "x1", "y1")]
    d["bbox"] = box if None not in box else None
    return d


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(settings: Settings | None = None, llm: LLM | None = None) -> FastAPI:
    settings = settings or get_settings()
    llm = llm or get_llm(settings)
    app = FastAPI(title="IT Annual Reports Q&A", version="0.1")

    if isinstance(llm, TransformersLLM):  # load the weights in the background so the first question is fast
        def preload() -> None:
            try:
                llm.load()
            except Exception:
                log.exception("model preload failed; questions will use the extractive fallback")
        threading.Thread(target=preload, daemon=True).start()

    def db():
        try:
            return closing(connect(settings.facts_db))
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/health")
    def health() -> dict:
        info = {"status": "ok", "llm_backend": llm.name, "facts_db": str(settings.facts_db)}
        try:
            with db() as conn:
                for table in ("documents", "facts", "passages"):
                    info[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        except HTTPException as exc:
            info |= {"status": "degraded", "problem": exc.detail}
        return info

    @app.get("/api/overview")
    def overview() -> dict:
        with db() as conn:
            reports = [dict(r) for r in conn.execute(
                "SELECT company, doc_id, title, file AS pdf, fy, n_pages AS pages FROM documents ORDER BY rowid")]
            metric_rows = [dict(r) for r in conn.execute("SELECT metric, label, unit FROM metric_dictionary ORDER BY rowid")]
            years = [r[0] for r in conn.execute("SELECT DISTINCT fy FROM facts ORDER BY fy")]
            counts = {name: conn.execute(sql).fetchone()[0] for name, sql in (
                ("facts", "SELECT count(*) FROM facts"), ("passages", "SELECT count(*) FROM passages"),
                ("tables", "SELECT count(*) FROM elements WHERE kind = 'table'"), ("pages", "SELECT count(*) FROM pages"))}
        model = {"vllm": settings.llm_model, "transformers": settings.llm_model_path.name}.get(llm.name)
        return {"companies": [r["company"] for r in reports], "reports": reports, "metrics": metric_rows,
                "years": years, "llm_backend": llm.name, "model": model, "counts": counts}

    @app.post("/api/ask")
    def ask(req: AskRequest) -> StreamingResponse:
        events = (sse(event, data) for event, data in answer_events(req.question, settings, llm, req.companies,
                                                                          req.use_cache, req.history))
        return StreamingResponse(events, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/metrics")
    def metrics(metric: str = Query("revenue", pattern=r"^[a-z_]{1,40}$")) -> dict:
        """The one chosen value per company x year (view best_facts), as the chart plots it."""
        with db() as conn:
            info = conn.execute("SELECT label, unit FROM metric_dictionary WHERE metric = ?", (metric,)).fetchone()
            rows = conn.execute(
                """SELECT f.fact_id, f.company, f.fy, f.value, f.unit, f.basis, f.page_no, f.doc_id, d.file AS pdf,
                          d.title AS report, f.status, f.quote, f.label_in_report, f.source_kind, f.source_rank,
                          f.agrees_with, f.x0, f.y0, f.x1, f.y1
                   FROM best_facts f JOIN documents d ON d.doc_id = f.doc_id
                   WHERE f.metric = ? ORDER BY f.company, f.fy""", (metric,)).fetchall()
        return {"metric": metric, "label": info["label"] if info else metric, "unit": info["unit"] if info else None,
                "facts": [with_bbox(r) for r in rows]}

    @app.get("/api/facts")
    def facts(company: str | None = None, metric: str | None = None, fy: str | None = None) -> dict:
        """Every extracted fact (not only the chosen ones), for the Extracted Data view."""
        sql = """SELECT f.fact_id, f.company, f.metric, f.fy, f.value, f.unit, f.basis, f.page_no, f.doc_id,
                        d.file AS pdf, d.title AS report, f.status, f.quote, f.label_in_report, f.source_kind,
                        f.source_rank, f.agrees_with, f.conflicts_with, f.x0, f.y0, f.x1, f.y1,
                        (b.fact_id IS NOT NULL) AS is_best
                 FROM facts f JOIN documents d ON d.doc_id = f.doc_id
                 LEFT JOIN best_facts b ON b.fact_id = f.fact_id WHERE 1 = 1"""
        params = []
        for column, value in (("f.company", company), ("f.metric", metric), ("f.fy", fy)):
            if value:
                sql += f" AND {column} = ?"
                params.append(value)
        with db() as conn:
            rows = conn.execute(sql + " ORDER BY f.metric, f.company, f.fy, f.source_rank", params).fetchall()
        return {"facts": [with_bbox(r) | {"is_best": bool(r["is_best"])} for r in rows]}

    @app.get("/api/evidence")
    def evidence(fact_id: int | None = None, passage_id: int | None = None) -> dict:
        if (fact_id is None) == (passage_id is None):
            raise HTTPException(400, "give exactly one of fact_id or passage_id")
        with db() as conn:
            card = fact_evidence(conn, fact_id) if fact_id is not None else passage_evidence(conn, passage_id)
        if card is None:
            raise HTTPException(404, "no such evidence")
        return card

    @app.get("/api/search")
    def search(q: str = Query(min_length=1, max_length=200), company: str | None = None) -> dict:
        with db() as conn:
            return {"results": search_snippets(conn, q, company)}

    def pdf_for(doc_id: str | None = None, file: str | None = None):
        """Resolve a PDF only through the documents table, never from a raw request path."""
        with db() as conn:
            row = conn.execute("SELECT file, n_pages FROM documents WHERE doc_id = ? OR file = ?",
                               (doc_id, file)).fetchone()
        if row is None:
            raise HTTPException(404, "unknown report")
        path = (settings.reports_dir / row["file"]).resolve()
        if path.parent != settings.reports_dir.resolve() or not path.is_file():
            raise HTTPException(404, "report file not available")
        return path, row["n_pages"]

    @app.get("/api/page/{doc_id}/{page_no}.png")
    def page_image(doc_id: str, page_no: int, x0: float | None = None, y0: float | None = None,
                   x1: float | None = None, y1: float | None = None) -> Response:
        path, n_pages = pdf_for(doc_id=doc_id)
        if not 1 <= page_no <= n_pages:
            raise HTTPException(404, "page out of range")
        with pymupdf.open(path) as pdf:
            page = pdf[page_no - 1]
            if None not in (x0, y0, x1, y1):
                page.draw_rect(pymupdf.Rect(x0, y0, x1, y1).normalize() + (-3, -3, 3, 3),
                               color=(0.95, 0.66, 0.0), fill=(1.0, 0.84, 0.42), fill_opacity=0.35, width=1.5)
            png = page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes("png")
        return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.get("/reports/{file}")
    def report_pdf(file: str) -> FileResponse:
        path, _ = pdf_for(file=file)
        return FileResponse(path, media_type="application/pdf")

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()

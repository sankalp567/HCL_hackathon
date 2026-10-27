"""HTTP API and the static frontend.

Laptop, no GPU:   uvicorn app.main:app --port 8000                      (LLM_BACKEND=extractive, the default)
GPU server:       LLM_BACKEND=transformers uvicorn app.main:app --host 127.0.0.1 --port 8000
vLLM elsewhere:   LLM_BACKEND=vllm LLM_BASE_URL=http://127.0.0.1:8001/v1 uvicorn app.main:app --port 8000

SSE contract of POST /api/ask ({"question": str}), one JSON object per event:
  meta       {query_id, understood: {companies, metric, fy, named}, sources: [{n, company, report, pdf, doc_id, page, kind, score}]}
  token      {t}                      a piece of the answer text, with [n] citation markers
  citations  {citations: [{n, company, report, pdf, doc_id, page, kind, quote, bbox, verified}]}
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
from app.llm import LLM, TransformersLLM, get_llm
from app.qa import answer_events

WEB_DIR = ROOT / "web"
log = logging.getLogger("app")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


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

    @app.post("/api/ask")
    def ask(req: AskRequest) -> StreamingResponse:
        events = (sse(event, data) for event, data in answer_events(req.question, settings, llm))
        return StreamingResponse(events, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/metrics")
    def metrics(metric: str = Query("revenue", pattern=r"^[a-z_]{1,40}$")) -> dict:
        with db() as conn:
            rows = conn.execute(
                """SELECT company, fy, value, unit, basis, page_no, doc_id, status, quote, source_rank, agrees_with,
                          label_in_report, x0, y0, x1, y1
                   FROM best_facts WHERE metric = ? ORDER BY company, fy""", (metric,)).fetchall()
        return {"metric": metric, "facts": [dict(r) for r in rows]}

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

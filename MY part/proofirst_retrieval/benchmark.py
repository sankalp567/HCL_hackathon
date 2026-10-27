"""Benchmark runner: executes experiment configs A-F (+ variants) on one corpus/qrels set and writes JSON, CSV, Markdown.

Usage (synthetic fixture):  python -m proofirst_retrieval.benchmark --label synthetic
Usage (real index, once Member 2's collection + Member 3's qrels exist):
    python -m proofirst_retrieval.benchmark --label real --queries real_queries.jsonl \
        --chroma-path PATH --collection NAME --embedder my_pkg.embed:embed_query [--field-map map.json]
Every number in the outputs is computed here; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import json
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .bm25 import BM25Retriever
from .contracts import Candidate, FieldMap
from .dataset import load_corpus, load_queries
from .dense import ChromaDenseRetriever, HashingDenseRetriever, load_corpus_from_chroma
from .metrics import mean_ignoring_none, numeric_evidence_metrics, percentile, query_metrics
from .pipeline import PipelineConfig, RetrievalPipeline
from .query import QueryProcessor
from .rerank import LexicalFeatureReranker, cross_encoder_status

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ALIASES = {"Acme": ["acme", "acme infotech"], "Borealis": ["borealis", "borealis software"],
                   "Cygnus": ["cygnus", "cygnus digital"]}
RANK_METRICS = ["recall@5", "recall@10", "precision@5", "mrr", "ndcg@5", "ndcg@10"]
NUM_METRICS = ["source_page_hit", "company_year_hit", "metric_table_hit", "row_evidence_hit",
               "company_coverage", "company_full_coverage", "scope_match@1"]
NUM_K = 5
STAGES = ["analysis", "dense", "bm25", "fusion", "rerank", "total"]


class Blocked(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- experiment matrix
def build_experiments(reranker) -> list[dict]:
    base = dict(k_candidates=20, final_k=10)
    ex = []
    def add(eid, name, purpose, **kw):
        ex.append({"id": eid, "name": name, "purpose": purpose, "cfg": PipelineConfig(name=eid, **{**base, **kw})})
    hyb = dict(use_dense=True, use_bm25=True)
    flt = dict(use_filters=True, per_entity=True, soft_boost=0.1, company_quota=True)
    for alias in (False, True):
        s = "+alias" if alias else ""
        if not alias:
            add("A", "Dense only", "baseline for dense retrieval", use_dense=True, use_bm25=False)
        add(f"B{s}", f"BM25 only{' (alias-assisted)' if alias else ''}", "lexical retrieval", use_dense=False, use_bm25=True,
            alias_expansion=alias)
        add(f"C{s}", "Dense + BM25 + RRF", "hybrid vs single retrievers", alias_expansion=alias, **hyb)
        add(f"D{s}", "Hybrid + RRF + rerank (stand-in heuristic)", "value of the rerank stage",
            alias_expansion=alias, reranker=reranker, **hyb)
        add(f"E{s}", "Hybrid + RRF + metadata filters", "company/year/scope filtering", alias_expansion=alias, **hyb, **flt)
        add(f"F{s}", "Hybrid + RRF + filters + rerank (stand-in heuristic)", "full pipeline",
            alias_expansion=alias, reranker=reranker, **hyb, **flt)
    for k in (10, 30):
        add(f"C@{k}", f"Hybrid, {k} candidates/retriever", "candidate-pool sweep", k_candidates=k, **hyb)
        add(f"E@{k}", f"Hybrid + filters, {k} candidates/retriever", "candidate-pool sweep", k_candidates=k, **hyb, **flt)
    return ex


def blocked_experiments() -> list[dict]:
    ok, why = cross_encoder_status()
    ce = f"cross-encoder not run: {why}"
    return [
        {"id": "D_ce", "name": "Hybrid + RRF + cross-encoder (MiniLM)", "status": "blocked", "reason": ce},
        {"id": "F_ce", "name": "Hybrid + RRF + filters + cross-encoder (MiniLM)", "status": "blocked", "reason": ce},
        {"id": "D_qwen3rr", "name": "Hybrid + Qwen3-Reranker-0.6B", "status": "blocked",
         "reason": "adapter not implemented and not attempted (torch/transformers not installed; no model downloaded)"},
        {"id": "real_chroma", "name": "All experiments on the real ChromaDB collection + manual qrels", "status": "blocked",
         "reason": "collection, metadata schema and query embedder from Member 2 and manual labels from Member 3 are not available; "
                   "real-mode code path exists but has never been executed against a real collection"},
        {"id": "A_instr", "name": "Dense with vs without Qwen3 query instruction", "status": "blocked",
         "reason": "needs Member 2's real Qwen3-Embedding-0.6B query embedder; the hashing stand-in has no instruction mode"},
    ]


# ---------------------------------------------------------------------------------------------- evaluation
def evaluate(pipe: RetrievalPipeline, cfg: PipelineConfig, queries: list[dict], repeats: int = 5) -> dict:
    per_query = []
    samples = {s: [] for s in STAGES}
    counts_acc: dict[str, list[int]] = {}
    for q in queries:
        pipe.run(q["question"], cfg)  # warm-up
        res = None
        for _ in range(repeats):
            res = pipe.run(q["question"], cfg)
            for s in STAGES:
                samples[s].append(res.timings_ms.get(s, 0.0))
        ranked = [c.chunk_id for c in res.candidates]
        qrels = q.get("qrels") or {}
        top = [{**c.metadata, "chunk_id": c.chunk_id} for c in res.candidates]
        m = query_metrics(ranked, qrels)
        n = numeric_evidence_metrics(top, q.get("expected"), NUM_K) if qrels else {k: None for k in NUM_METRICS}
        for k, v in res.counts.items():
            counts_acc.setdefault(k, []).append(v)
        per_query.append({"qid": q["qid"], "category": q["category"], "question": q["question"], "ranked": ranked,
                          "template": q.get("template", q["qid"]), "status": res.status,
                          "metrics": m, "numeric": n, "limitations": res.limitations, "intent": res.analysis.intent,
                          "reranker_status": res.reranker_status})
    ans = [p for p in per_query if p["category"] != "unanswerable"]
    agg, ns = {}, {}
    for k in RANK_METRICS:
        agg[k], ns[k] = mean_ignoring_none(p["metrics"][k] for p in ans)
    for k in NUM_METRICS:
        agg[k], ns[k] = mean_ignoring_none(p["numeric"][k] for p in ans)
    # template-macro: average within each query template first, so 12 near-identical queries count once, not 12 times
    tmpl_macro = {}
    for k in ("recall@10", "mrr", "ndcg@10"):
        per_t: dict[str, list[float]] = {}
        for p in ans:
            if p["metrics"][k] is not None:
                per_t.setdefault(p["template"], []).append(p["metrics"][k])
        tmpl_macro[k] = (sum(sum(v) / len(v) for v in per_t.values()) / len(per_t)) if per_t else None
    groups = {}
    for gname, pred in (("legacy_templates", lambda p: not str(p["template"]).startswith("H")),
                        ("handwritten", lambda p: str(p["template"]).startswith("H"))):
        sub = [p for p in ans if pred(p)]
        groups[gname] = {"n": len(sub), **{k: mean_ignoring_none(p["metrics"][k] for p in sub)[0] for k in ("recall@10", "mrr", "ndcg@10")}}
    unans = [p for p in per_query if p["category"] == "unanswerable"]
    status_counts = {}
    for p in per_query:
        status_counts[p["status"]] = status_counts.get(p["status"], 0) + 1
    unans_with_candidates = sum(1 for p in unans if p["ranked"])
    cats = sorted({p["category"] for p in ans})
    by_cat = {c: {k: mean_ignoring_none(p["metrics"][k] for p in ans if p["category"] == c)[0]
                  for k in ("recall@10", "mrr", "ndcg@10")} | {"n": sum(p["category"] == c for p in ans)} for c in cats}
    lat = {}
    for s in STAGES:
        v = samples[s]
        lat[s] = {"p50_ms": percentile(v, 50), "p95_ms": percentile(v, 95), "mean_ms": float(np.mean(v)), "n": len(v)}
    return {"metrics": agg, "metric_n": ns, "by_category": by_cat, "latency": lat, "template_macro": tmpl_macro,
            "n_templates": len({p["template"] for p in ans}), "by_group": groups, "status_counts": status_counts,
            "unanswerable_returning_candidates": f"{unans_with_candidates}/{len(unans)}",
            "counts_mean": {k: float(np.mean(v)) for k, v in counts_acc.items()},
            "n_answerable": len(ans), "n_unanswerable_excluded": len(per_query) - len(ans), "per_query": per_query}


def paired_bootstrap(a: list[float | None], b: list[float | None], n_boot: int = 2000, seed: int = 7) -> dict | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 5:
        return None
    d = np.array([y - x for x, y in pairs])
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], "n": len(d)}


def hardware() -> dict:
    h = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0]}
    if sys.platform == "darwin":
        try:
            h["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            h["ram_gb"] = round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 2**30, 1)
        except Exception:
            pass
    h["gpu_used"] = False
    return h


# ---------------------------------------------------------------------------------------------- builders
def build_synthetic(corpus_path: Path):
    chunks = load_corpus(corpus_path)
    lex = BM25Retriever(chunks)
    dense = HashingDenseRetriever(chunks)
    qp = QueryProcessor(company_aliases=FIXTURE_ALIASES)
    info = {"kind": "synthetic fixture (fictional companies, invented numbers)", "chunks": len(chunks),
            "dense": "HashingDenseRetriever (1024-d signed hashing of words+char-trigrams): STAND-IN, not Qwen3-Embedding-0.6B",
            "lexical": "BM25Retriever (own implementation, k1=1.5 b=0.75, unigram+bigram, crude plural stemmer)",
            "vector_store": "none (in-memory numpy)", "company_aliases": "fixture registry (Acme/Borealis/Cygnus)"}
    return RetrievalPipeline(lex, dense, chunks, qp), chunks, info


def build_real(a: argparse.Namespace):
    try:
        import chromadb  # type: ignore
    except Exception as e:
        raise Blocked(f"chromadb not installed: {e}")
    fm = FieldMap(json.loads(Path(a.field_map).read_text()) if a.field_map else {})
    mod, fn = a.embedder.split(":")
    embed = getattr(importlib.import_module(mod), fn)
    col = chromadb.PersistentClient(path=a.chroma_path).get_collection(a.collection)
    chunks = load_corpus_from_chroma(col, fm)
    pipe = RetrievalPipeline(BM25Retriever(chunks), ChromaDenseRetriever(col, embed, fm), chunks, QueryProcessor())
    info = {"kind": "real annual-report index", "chunks": len(chunks), "dense": f"Chroma collection '{a.collection}' via {a.embedder}",
            "lexical": "BM25Retriever over chunks read from the same collection", "vector_store": a.chroma_path}
    return pipe, chunks, info


# ---------------------------------------------------------------------------------------------- main
def run(label: str, corpus: Path, queries_path: Path, out_dir: Path, docs_dir: Path, repeats: int, args=None) -> dict:
    corpus_ids = None
    if label == "synthetic":
        pipe, chunks, info = build_synthetic(corpus)
    else:
        pipe, chunks, info = build_real(args)
    corpus_ids = {c.chunk_id for c in chunks}
    queries = load_queries(queries_path, corpus_ids)
    ans_q = [q for q in queries if q["qrels"]]
    an = [pipe.qp.analyze(q["question"]) for q in ans_q]
    explicit = {"company": sum(bool(a.companies) for a in an), "fiscal_year": sum(bool(a.periods) for a in an),
                "scope": sum(bool(a.scope) for a in an), "latest_wording": sum(a.relative_period == "latest" for a in an),
                "cross_company_wording": sum(a.cross_company for a in an), "n_answerable": len(ans_q)}
    templates = {q.get("template", q["qid"]) for q in queries}
    ex = build_experiments(LexicalFeatureReranker())
    if label == "synthetic":   # never let a stand-in pass for the real model in a table
        for e in ex:
            if e["cfg"].use_dense:
                e["name"] += " [dense = hashing stand-in]"
    results, per_query_rows = [], []
    for e in ex:
        r = evaluate(pipe, e["cfg"], queries, repeats)
        comps = [n for n, on in (("dense", e["cfg"].use_dense), ("bm25", e["cfg"].use_bm25), ("rrf", e["cfg"].use_dense and e["cfg"].use_bm25),
                                 ("metadata_filters", e["cfg"].use_filters), ("per_entity_retrieval", e["cfg"].per_entity),
                                 ("alias_expansion", e["cfg"].alias_expansion),
                                 ("rerank:" + (e["cfg"].reranker.name if e["cfg"].reranker else ""), e["cfg"].reranker is not None)) if on]
        results.append({"id": e["id"], "name": e["name"], "purpose": e["purpose"], "status": "ran", "config": e["cfg"].describe(),
                        "components_executed": comps, **{k: v for k, v in r.items() if k != "per_query"}})
        for p in r["per_query"]:
            per_query_rows.append({"experiment": e["id"], "qid": p["qid"], "category": p["category"], "intent": p["intent"],
                                   **{k: p["metrics"][k] for k in RANK_METRICS}, **{k: p["numeric"][k] for k in NUM_METRICS},
                                   "template": p["template"], "status": p["status"], "top3": "|".join(p["ranked"][:3]), "limitations": "; ".join(p["limitations"])})
        e["per_query"] = r["per_query"]
    by_id = {r["id"]: i for i, r in enumerate(results)}
    ex_by = {e["id"]: e for e in ex}
    def series(eid, k):
        return [p["metrics"][k] for p in ex_by[eid]["per_query"] if p["category"] != "unanswerable"]
    paired = []
    for eid in by_id:
        if eid == "C":
            continue
        for k in ("ndcg@10", "mrr"):
            res = paired_bootstrap(series("C", k), series(eid, k))
            if res:
                paired.append({"experiment": eid, "baseline": "C", "metric": k, **res})
    out = {
        "meta": {"label": label, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "version": __version__,
                 "hardware": hardware(), "corpus": info, "dataset": {"queries_file": str(queries_path.name), "n_queries": len(queries),
                 "n_answerable": sum(1 for q in queries if q["qrels"]), "n_unanswerable": sum(1 for q in queries if not q["qrels"]),
                 "n_templates": len(templates), "n_handwritten": sum(1 for q in queries if str(q.get("template", "")).startswith("H")),
                 "answerable_queries_stating": explicit,
                 "categories": {c: sum(1 for q in queries if q["category"] == c) for c in sorted({q['category'] for q in queries})},
                 "provenance": sorted({q["provenance"] for q in queries})},
                 "rrf": {"k": 60, "rank_base": 1, "formula": "sum_lists 1/(k+rank)"}, "repeats_per_query": repeats, "numeric_k": NUM_K,
                 "real_data_benchmark": "BLOCKED" if label == "synthetic" else "ran"},
        "experiments": results, "blocked": blocked_experiments(), "paired_vs_C": paired,
    }
    out_dir.mkdir(parents=True, exist_ok=True); docs_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_benchmark_results.json").write_text(json.dumps(out, indent=2, default=str))
    _write_csv(out_dir / "retrieval_benchmark_results.csv", results)
    _write_per_query(out_dir / "retrieval_benchmark_per_query.csv", per_query_rows)
    from .report import write_report
    write_report(out, ex, docs_dir / "RETRIEVAL_EVALUATION.md")
    return out


def _write_csv(path: Path, results: list[dict]):
    cols = ["experiment", "name", "status", *RANK_METRICS, *NUM_METRICS, "p50_total_ms", "p95_total_ms", "p50_dense_ms",
            "p50_bm25_ms", "p50_fusion_ms", "p50_rerank_ms", "ndcg@10_template_macro", "mrr_template_macro", "unanswerable_returning_candidates", "mean_union", "mean_after_dedupe", "mean_reranked_pool", "mean_returned"]
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            L, C = r["latency"], r["counts_mean"]
            w.writerow([r["id"], r["name"], r["status"], *[_f(r["metrics"][k]) for k in RANK_METRICS + NUM_METRICS],
                        _f(L["total"]["p50_ms"]), _f(L["total"]["p95_ms"]), _f(L["dense"]["p50_ms"]), _f(L["bm25"]["p50_ms"]),
                        _f(L["fusion"]["p50_ms"]), _f(L["rerank"]["p50_ms"]), _f(r["template_macro"]["ndcg@10"]), _f(r["template_macro"]["mrr"]),
                        r["unanswerable_returning_candidates"], _f(C.get("union_before_dedupe")),
                        _f(C.get("after_dedupe")), _f(C.get("reranked_pool")), _f(C.get("returned"))])


def _write_per_query(path: Path, rows: list[dict]):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: _f(v) for k, v in r.items()})


def _f(v):
    return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else v)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--corpus", default=str(ROOT / "data" / "synthetic_corpus.jsonl"))
    ap.add_argument("--queries", default=str(ROOT / "data" / "synthetic_queries.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    ap.add_argument("--docs-dir", default=str(ROOT / "docs"))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--chroma-path"); ap.add_argument("--collection"); ap.add_argument("--embedder"); ap.add_argument("--field-map")
    a = ap.parse_args(argv)
    try:
        out = run(a.label, Path(a.corpus), Path(a.queries), Path(a.out_dir), Path(a.docs_dir), a.repeats, a)
    except Blocked as e:
        print(f"BLOCKED: {e}", file=sys.stderr); return 2
    print(f"wrote {len(out['experiments'])} experiments to {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

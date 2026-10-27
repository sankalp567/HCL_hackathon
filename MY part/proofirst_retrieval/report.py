"""Markdown report generated from benchmark output. All tables and the winner are computed from the run."""
from __future__ import annotations

from pathlib import Path

from .metrics import mean_ignoring_none

LAT_GATE_MS = 100.0           # acceptable p95 total retrieval latency on the local box (interactive chat budget)
TIE_EPS = 0.01                # composite differences below this are reported as inconclusive
COMPOSITE = ["recall@10", "mrr", "ndcg@10", "source_page_hit"]
MAIN = ["A", "B", "C", "D", "E", "F"]


def _v(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def composite(r):
    vals = [r["metrics"][k] for k in COMPOSITE if r["metrics"][k] is not None]
    return sum(vals) / len(vals) if vals else None


def _table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def pick_winner(results):
    ok = [r for r in results if r["status"] == "ran" and (r["latency"]["total"]["p95_ms"] or 0) <= LAT_GATE_MS and composite(r) is not None]
    ok.sort(key=lambda r: (-composite(r), r["latency"]["total"]["p50_ms"]))
    if not ok:
        return None, None, False
    best = ok[0]
    near = [r for r in ok[1:] if composite(best) - composite(r) < TIE_EPS]
    if near:
        simplest = min([best, *near], key=lambda r: (r["config"]["reranker"] != "none", r["config"]["use_filters"] is False,
                                                     r["latency"]["total"]["p50_ms"]))
        return simplest, [best, *near], True
    return best, [best], False


def write_report(out: dict, ex: list[dict], path: Path):
    meta, R = out["meta"], {r["id"]: r for r in out["experiments"]}
    syn = meta["label"] == "synthetic"
    L = []
    w = L.append
    banner = ("**DATA LABEL: SYNTHETIC.** Every number below was measured on a small fixture of *fictional* companies with *invented* "
              "figures and labels derived by construction. It validates the pipeline mechanics and compares configurations on a "
              "synthetic task. It does **not** measure retrieval quality on the real TCS / Infosys / HCLTech annual reports. "
              "The dense retriever is a hashing **stand-in**, not Qwen3-Embedding-0.6B, and the reranker is a lexical-feature "
              "heuristic, not a cross-encoder. **Real-data benchmark: BLOCKED** (inputs listed in section 12)."
              if syn else "**DATA LABEL: REAL annual-report index and manually checked labels.**")
    w("# ProofFirst AI — Retrieval Evaluation (Member 4)\n")
    w(f"> {banner}\n")
    w(f"Generated {meta['generated_at']} by `python -m proofirst_retrieval.benchmark --label {meta['label']}` (code version {meta['version']}). "
      f"Machine-readable: `artifacts/retrieval_benchmark_results.json`, `.csv`, `retrieval_benchmark_per_query.csv`.\n")

    w("## 1. Experimental setup\n")
    h = meta["hardware"]
    w(f"- Hardware: {h.get('cpu', h['machine'])}, {h.get('ram_gb', '?')} GB RAM, {h['platform']}, Python {h['python']}, GPU used: {h['gpu_used']}.")
    c = meta["corpus"]
    w(f"- Dense: {c['dense']}.\n- Lexical: {c['lexical']}.\n- Vector store: {c['vector_store']}.")
    w(f"- Fusion: RRF, k={meta['rrf']['k']}, ranks **{meta['rrf']['rank_base']}-based**, score = `{meta['rrf']['formula']}`; in-list repeats count once; "
      f"ties broken by fusion score, then best single-list rank, then first appearance.")
    w(f"- Latency: {meta['repeats_per_query']} timed repeats per query after one warm-up, wall-clock `perf_counter`, single process, no GPU. "
      f"p50/p95 are reported only if the sample has at least 20 timings.\n")

    w("## 2. Corpus and test dataset\n")
    d = meta["dataset"]
    w(f"- Corpus: {c['kind']}; **{c['chunks']} chunks**. Provenance of labels: {', '.join(d['provenance'])}.")
    w(f"- Queries: {d['n_queries']} ({d['n_answerable']} answerable, {d['n_unanswerable']} unanswerable and excluded from rank metrics). "
      f"Categories: " + ", ".join(f"{k}={v}" for k, v in d["categories"].items()) + ".")
    ex_ = d["answerable_queries_stating"]; na = ex_["n_answerable"]
    w(f"- Query independence: **{d['n_templates']} distinct templates** across {d['n_queries']} queries ({d['n_handwritten']} hand-written, one template each; "
      f"the rest are legacy loop templates repeated across companies). Section 5.6 reports template-macro scores (each template counts once) next to the per-query means.")
    w(f"- Explicitness of the answerable queries: company named in {ex_['company']}/{na}, fiscal year in {ex_['fiscal_year']}/{na}, scope in {ex_['scope']}/{na}; "
      f"'latest' wording in {ex_['latest_wording']}, cross-company wording without names in {ex_['cross_company_wording']}.")
    if syn:
        w("- **Disclosed biases:** (1) queries that name company/year/scope make the hard filters in E/F act like an oracle for those queries, so their gain is an "
          "**upper bound** (the unnamed / latest / omitted-scope queries above are the ones that test routing instead); "
          "(2) legacy templates are repeated across three companies and labels are derived from chunk ids; hand-written queries were written by the same author as the aliases and the heuristic; "
          "(3) the alias registry in `query.py` contains the exact paraphrase terms used in some queries ('net sales', 'bottom line', 'workforce'), so alias gains are inflated; "
          "(4) the hashing 'dense' stand-in is word + char-trigram based, i.e. effectively lexical, so hybrid gains from A+B fusion are NOT evidence of semantic/lexical complementarity; "
          "(5) historical queries give grade 2 to the other-year chunk (comparative column), which a hard fiscal-year filter excludes, so E/F recall on that category is lowered by label design, not by a retrieval defect; "
          "(6) E/F also apply a x1.1 soft boost for `content_type=table` on numeric intents and use `company_quota`; "
          "(7) 'latest' is resolved from index metadata, which on this fixture has two report years per company.")
        w("- Further design bias: table-row chunks do not contain the company name (as in real annual-report tables), so unfiltered "
          "retrieval cannot tell companies apart on table rows; the controlled alias registry and the paraphrase queries were written by "
          "the same author, which favours alias expansion.\n")

    w("## 3. Tested configurations\n")
    rows = []
    for r in out["experiments"]:
        cfg = r["config"]
        soft = f"x{1 + cfg['soft_boost']:.1f} for content_type=table (soft pref.)" if cfg["use_filters"] and cfg["soft_boost"] else "-"
        rows.append([r["id"], r["name"], ", ".join(x for x in r["components_executed"]), cfg["k_candidates"], cfg["rerank_pool"] if cfg["reranker"] != "none" else "-", soft])
    w(_table(rows, ["ID", "Name", "Components actually executed", "cand/retriever", "rerank pool", "soft boost"]))
    w("\nNot run (recorded, not substituted):\n")
    w(_table([[b["id"], b["name"], b["reason"]] for b in out["blocked"]], ["ID", "Configuration", "Reason"]))
    w("")

    w("## 4. Metric definitions\n")
    w("- Relevance: chunk-level graded qrels (1-3); grade>0 counts as relevant for binary metrics (the fixture only uses grades 2 and 3). Ranked list = final top-10 chunks.\n"
      "- Recall@K = relevant in top K / all relevant. Precision@K = relevant in top K / K (denominator fixed at K). MRR = 1/rank of first relevant (0 if absent).\n"
      "- nDCG@K: gain 2^grade-1, discount log2(position+1), ideal = the labelled grades sorted descending, truncated to K. Queries with no relevant chunk return *undefined* and are excluded (not scored 0 or 1).\n"
      f"- Numerical-evidence metrics use top-{meta['numeric_k']}: source-page hit (expected doc+PDF page present), company+year hit, metric/table hit (expected table id), "
      "row evidence hit (expected row chunk), company coverage (share of requested companies present; multi-company queries only), full coverage, scope match@1. "
      "`company_year_hit` is computed for single-company queries only; coverage metrics for multi-company queries only; `source_page_hit` accepts *any* expected page, so it is lenient and saturates easily. A metric is *n/a* if the labels it needs are absent. These measure **retrieval**, not whether a parsed number is financially correct.\n")

    def main_table(ids, title):
        w(title + "\n")
        rows = []
        for i in ids:
            if i in R:
                m = R[i]["metrics"]
                rows.append([i, R[i]["name"]] + [_v(m[k]) for k in ["recall@5", "recall@10", "precision@5", "mrr", "ndcg@5", "ndcg@10"]])
        w(_table(rows, ["ID", "Config", "R@5", "R@10", "P@5", "MRR", "nDCG@5", "nDCG@10"]))
        w("")
        rows = []
        for i in ids:
            if i in R:
                m = R[i]["metrics"]
                rows.append([i] + [_v(m[k]) for k in ["source_page_hit", "company_year_hit", "metric_table_hit", "row_evidence_hit", "company_coverage", "company_full_coverage", "scope_match@1"]])
        w(_table(rows, ["ID", "page hit", "company+year hit", "metric/table hit", "row hit", "company coverage", "full coverage", "scope@1"]))
        w("")

    w("## 5. Results\n")
    main_table(MAIN, "### 5.1 Main matrix (A-F; lexical queries without alias expansion)")
    main_table(["B+alias", "C+alias", "D+alias", "E+alias", "F+alias"], "### 5.2 Alias-assisted variants")
    main_table(["C@10", "C", "C@30", "E@10", "E", "E@30"], "### 5.3 Candidate-pool sweep (per retriever; C and E at 10/20/30)")

    w("### 5.4 Per-category recall@10 / MRR (main matrix)\n")
    cats = sorted({k for i in MAIN for k in R[i]["by_category"]})
    rows = [[i] + [f"{_v(R[i]['by_category'].get(c, {}).get('recall@10'), 2)} / {_v(R[i]['by_category'].get(c, {}).get('mrr'), 2)}" for c in cats] for i in MAIN]
    w(_table(rows, ["ID"] + [f"{c} (n={R['A']['by_category'][c]['n']})" for c in cats]))
    w("")

    w("### 5.5 Paired bootstrap vs hybrid baseline C (95% CI of mean difference, 2000 resamples)\n")
    rows = [[p["experiment"], p["metric"], f"{p['mean_diff']:+.3f}", f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]", p["n"],
             "yes" if (p["ci95"][0] > 0 or p["ci95"][1] < 0) else "no"] for p in out["paired_vs_C"] if p["experiment"] in MAIN + ["B+alias", "C+alias", "E+alias", "F+alias"]]
    w(_table(rows, ["Config", "Metric", "Mean diff vs C", "95% CI", "n", "CI excludes 0"]))
    w("")

    w("### 5.6 Template-macro scores, legacy vs hand-written queries\n")
    rows = []
    for i in MAIN + ["E+alias", "D+alias"]:
        r = R[i]; tm = r["template_macro"]; g = r["by_group"]
        rows.append([i, _v(r["metrics"]["ndcg@10"]), _v(tm["ndcg@10"]), _v(r["metrics"]["mrr"]), _v(tm["mrr"]),
                     f"{_v(g['legacy_templates']['ndcg@10'])} (n={g['legacy_templates']['n']})", f"{_v(g['handwritten']['ndcg@10'])} (n={g['handwritten']['n']})"])
    w(_table(rows, ["ID", "nDCG@10 per-query", "nDCG@10 template-macro", "MRR per-query", "MRR template-macro", "nDCG@10 legacy templates", "nDCG@10 hand-written"]))
    w("\nTemplate-macro averages inside each template first, so a template repeated across companies counts once. Prefer it when judging generalisation.\n")
    w("### 5.7 Retrieval status and unanswerable queries\n")
    rows = [[i, ", ".join(f"{k}={v}" for k, v in sorted(R[i]["status_counts"].items())), R[i]["unanswerable_returning_candidates"]] for i in MAIN]
    w(_table(rows, ["ID", "status counts (all queries)", "unanswerable queries that still returned candidates"]))
    w("\nNo relevance policy is configured in any benchmark run (no threshold is invented), so dense configurations return candidates even for unanswerable questions; "
      "`status=no_evidence` only appears when nothing at all is retrieved. Thresholds must be calibrated on real labelled data.\n")

    w("## 6. Latency and candidate counts\n")
    rows = []
    for i in MAIN + ["C@30", "E@30"]:
        r = R[i]; Lt = r["latency"]; Cn = r["counts_mean"]; cf = r["config"]
        st = lambda stage, on: _v(Lt[stage]["p50_ms"], 3) if on else "n/a (not run)"
        rows.append([i, st("dense", cf["use_dense"]), st("bm25", cf["use_bm25"]), _v(Lt["fusion"]["p50_ms"], 3) if (cf["use_dense"] and cf["use_bm25"]) else "n/a",
                     st("rerank", cf["reranker"] != "none"), _v(Lt["total"]["p50_ms"], 3), _v(Lt["total"]["p95_ms"], 3), f"{Cn.get('union_before_dedupe', 0):.1f}",
                     f"{Cn.get('after_dedupe', 0):.1f}", f"{Cn.get('reranked_pool', 0):.1f}", f"{Cn.get('returned', 0):.1f}"])
    w(_table(rows, ["ID", "dense p50 ms", "bm25 p50 ms", "fusion p50 ms", "rerank p50 ms", "total p50 ms", "total p95 ms", "candidates in", "after dedupe", "reranked", "returned"]))
    w("\nLatencies are **indicative only**: one benchmark run, no run-to-run variance, repeats of one query are correlated (effective sample is about the number of queries, not queries x repeats), and E/F p95 is driven by the multi-company queries that run per-entity retrieval. They are for a ~130-chunk in-memory corpus. They show relative stage cost only and **will not extrapolate** to a ChromaDB collection with thousands of chunks and a 0.6B embedder.\n")

    # failure cases
    w("## 7. Failure cases and ranking differences\n")
    pq = {e["id"]: {p["qid"]: p for p in e["per_query"]} for e in ex}
    def worst(eid, n=5):
        ps = [p for p in pq[eid].values() if p["category"] != "unanswerable" and p["metrics"]["mrr"] is not None]
        return sorted(ps, key=lambda p: (p["metrics"]["mrr"], p["metrics"]["recall@10"]))[:n]
    w("**Hardest queries for the unfiltered hybrid (C)** (lowest MRR):\n")
    rows = [[p["qid"], p["category"], p["question"], _v(p["metrics"]["mrr"], 2), ", ".join(p["ranked"][:2])] for p in worst("C")]
    w(_table(rows, ["qid", "category", "question", "MRR", "top-2 chunk ids"]))
    w("\n**Hardest queries for F (filters + rerank)**:\n")
    rows = [[p["qid"], p["category"], p["question"], _v(p["metrics"]["mrr"], 2), ", ".join(p["ranked"][:2]), "; ".join(p["limitations"]) or "-"] for p in worst("F")]
    w(_table(rows, ["qid", "category", "question", "MRR", "top-2 chunk ids", "limitations reported"]))
    diffs = []
    for qid, pc in pq["C"].items():
        pf = pq["F"][qid]
        if pc["metrics"]["mrr"] is None:
            continue
        dlt = pf["metrics"]["mrr"] - pc["metrics"]["mrr"]
        if abs(dlt) > 1e-9:
            diffs.append((dlt, qid, pc, pf))
    diffs.sort(key=lambda x: x[0])
    w(f"\n**Ranking differences C -> F**: {sum(1 for x in diffs if x[0] > 0)} queries improved MRR, {sum(1 for x in diffs if x[0] < 0)} got worse, "
      f"{len(pq['C']) - len(diffs)} unchanged/undefined.\n")
    show = diffs[:3] + diffs[-3:] if len(diffs) > 6 else diffs
    if show:
        w(_table([[q, pc["question"], _v(pc["metrics"]["mrr"], 2), _v(pf["metrics"]["mrr"], 2), ", ".join(pc["ranked"][:1]), ", ".join(pf["ranked"][:1])]
                  for _, q, pc, pf in show], ["qid", "question", "MRR C", "MRR F", "C top-1", "F top-1"]))
    lim = sorted({x for p in pq["F"].values() for x in p["limitations"]})
    if lim:
        w("\nLimitations the pipeline reported at run time (filters it refused to apply, missing evidence):\n")
        w("\n".join(f"- {x}" for x in lim[:8]))
    w("")

    # winner
    w("## 8. Best configuration\n")
    best, group, tie = pick_winner(out["experiments"])
    rows = sorted(out["experiments"], key=lambda r: -(composite(r) or 0))[:6]
    w(f"Selection rule (fixed before looking at results): composite = mean({', '.join(COMPOSITE)}); candidate must have p95 total latency <= {LAT_GATE_MS:.0f} ms; "
      f"configurations within {TIE_EPS} composite are treated as tied; among ties prefer the one with no reranker, then lower p50 latency "
      f"(latency differences of a fraction of a millisecond are timer noise and are NOT used as evidence). `source_page_hit` is saturated on this fixture and mildly favours filtered configs.\n")
    w(_table([[r["id"], r["name"], _v(composite(r)), _v(r["latency"]["total"]["p50_ms"], 3)] for r in rows], ["ID", "Config", "Composite", "p50 total ms"]))
    if best:
        w(f"\n**Default on this synthetic fixture: `{best['id']}` ({best['name']}).**")
        if tie:
            from .benchmark import paired_bootstrap
            top = max(group, key=composite)
            ex_by = {e["id"]: e for e in ex}
            for other in group:
                if other["id"] == top["id"]:
                    continue
                sa = [p["metrics"]["ndcg@10"] for p in ex_by[other["id"]]["per_query"] if p["category"] != "unanswerable"]
                sb = [p["metrics"]["ndcg@10"] for p in ex_by[top["id"]]["per_query"] if p["category"] != "unanswerable"]
                bs = paired_bootstrap(sa, sb)
                if bs:
                    w(f"\n- `{top['id']}` minus `{other['id']}` on nDCG@10: {bs['mean_diff']:+.3f}, 95% CI [{bs['ci95'][0]:+.3f}, {bs['ci95'][1]:+.3f}].")
            w(f"\n**Inconclusive:** {', '.join(r['id'] for r in group)} are statistically indistinguishable on this fixture; `{best['id']}` is named the "
              "default only because it is the simplest of the tied set (no reranker where possible). Latency does not discriminate: every config is far under 1 ms here.")
        nores = [r for r in out["experiments"] if r["config"]["reranker"] == "none" and composite(r) is not None
                 and (r["latency"]["total"]["p95_ms"] or 0) <= LAT_GATE_MS]
        if nores and best["config"]["reranker"] != "none":
            from .benchmark import paired_bootstrap
            alt = max(nores, key=composite)
            ex_by = {e["id"]: e for e in ex}
            sa = [p["metrics"]["ndcg@10"] for p in ex_by[alt["id"]]["per_query"] if p["category"] != "unanswerable"]
            sb = [p["metrics"]["ndcg@10"] for p in ex_by[best["id"]]["per_query"] if p["category"] != "unanswerable"]
            bs = paired_bootstrap(sa, sb)
            if bs:
                w(f"\nBest configuration **without** a reranker: `{alt['id']}` (composite {_v(composite(alt))}). `{best['id']}` minus `{alt['id']}` on nDCG@10: "
                  f"{bs['mean_diff']:+.3f}, 95% CI [{bs['ci95'][0]:+.3f}, {bs['ci95'][1]:+.3f}]. The winner uses the stand-in heuristic reranker, which was tuned against this "
                  "fixture's failures (section 9); treat the reranker's advantage as unproven for real reports.")
        per = {}
        for k in ["recall@10", "mrr", "ndcg@10", "source_page_hit", "company_full_coverage"]:
            cand = [r for r in out["experiments"] if r["metrics"][k] is not None]
            top = max(cand, key=lambda r: r["metrics"][k])
            ties = [r["id"] for r in cand if abs(r["metrics"][k] - top["metrics"][k]) < 1e-9]
            per[k] = f"{k}: {_v(top['metrics'][k])} ({', '.join(ties[:6])}{'...' if len(ties) > 6 else ''})"
        w("\nBest per metric: " + "; ".join(per.values()) + ".\n")
    if syn:
        w("This selection is a statement about the **synthetic** task only; it is not a recommendation for the real reports until section 12 is done.\n")

    # reranker
    w("## 9. Does reranking pay for itself?\n")
    rows = []
    for a, b in (("C", "D"), ("E", "F")):
        da = R[b]["metrics"]; ca = R[a]["metrics"]
        rows.append([f"{a} -> {b}"] + [f"{da[k] - ca[k]:+.3f}" for k in ["recall@10", "mrr", "ndcg@10", "source_page_hit"]] +
                    [f"{R[b]['latency']['rerank']['p50_ms']:.3f}", f"{R[b]['latency']['total']['p50_ms'] - R[a]['latency']['total']['p50_ms']:+.3f}"])
    w(_table(rows, ["Pair", "Δ R@10", "Δ MRR", "Δ nDCG@10", "Δ page hit", "rerank p50 ms", "Δ total p50 ms"]))
    sig = [p for p in out["paired_vs_C"] if p["experiment"] in ("D", "F") and p["metric"] == "ndcg@10"]
    w("\n" + " ".join(f"{p['experiment']} vs C on nDCG@10: {p['mean_diff']:+.3f}, CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]." for p in sig))
    w("\n**Caveat on D/F:** the stand-in heuristic scores chunks partly on company/scope agreement taken from the query analysis (only for constraints the "
      "retriever did *not* enforce), so its gain over C mixes 'reranking' with 'metadata-aware scoring' (E gets the same signal through filters). "
      "An earlier version also scored down chunks that lacked literals already guaranteed by hard filters (e.g. 'FY2025-26'), which pushed correct risk-narrative "
      "chunks out of the top 10 (archived v1 results: `artifacts/v1_synthetic/`). It now removes enforced-constraint spans before scoring (`rerank.relevance_text`). "
      "That fix was written after inspecting failures on this fixture, so the heuristic's advantage here is partly fitted to it. "
      "Whether a learned cross-encoder helps is an untested hypothesis.")
    w("\nThe reranker here is a deterministic heuristic, so its cost (microseconds-milliseconds) says nothing about a 22M-param MiniLM or a 0.6B "
      "Qwen3-Reranker, which are orders of magnitude slower per pair. The rerank decision for the real system **cannot be made from these results**; "
      "rerun D/F with the real cross-encoder on the real labelled set.\n")

    w("## 10. Trade-offs and limitations\n")
    w("- Small synthetic fixture (see section 2): CIs in 5.5 are wide; treat differences smaller than the CI as noise.\n"
      "- Filters are only as good as metadata: a hard filter excludes chunks whose field is missing; filters on fields absent from the whole index, or on values "
      "that are not report years, are *not applied* and the limitation is returned to the caller (see section 7).\n"
      "- Relevance scores (BM25, cosine, RRF, rerank) never prove that a number is right. Downstream numeric validation is a separate task.\n"
      "- Alias expansion is a controlled, hand-written registry; it can help paraphrases it knows and cannot help ones it does not. Wrong aliases could pull confusable metrics, so `revenue from operations`, `total income` and `other income` are separate entries.\n")

    w("## 11. Label\n")
    w("**Results: SYNTHETIC only.** No measurement in this file comes from real annual-report data.\n" if syn else "**Results: REAL.**\n")

    w("## 12. What is needed to unblock the real-data benchmark\n")
    w("1. **Member 2:** persistent ChromaDB path + collection name; the metadata keys on each chunk (map to canonical names in `FieldMap`: company, fiscal_year, doc_id, pdf_page_index, printed_page, content_type, scope, table_id, table_title, headers, row_label, row_index, unit); the collection's distance metric.\n"
      "2. **Member 2:** an importable `embed_query(text) -> list[float]` using the *same* Qwen3-Embedding-0.6B revision, dimension and query-instruction setting as indexing.\n"
      "3. **Member 3:** 30+ manually checked queries in the JSONL schema of `proofirst_retrieval/dataset.py` (qrels pointing at real chunk ids; `provenance: manual:<name>`), including at least a few unanswerable and multi-company ones.\n"
      "4. A decision on installing `chromadb`, `torch`, `sentence-transformers` (not present in this environment) and downloading a cross-encoder; then run: "
      "`python -m proofirst_retrieval.benchmark --label real --queries real_queries.jsonl --chroma-path PATH --collection NAME --embedder pkg.mod:embed_query`.\n"
      "*(The real-mode code path is implemented but has not been executed against a real Chroma collection.)*\n")
    path.write_text("\n".join(L))

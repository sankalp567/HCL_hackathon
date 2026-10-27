"""Vector search over the RAG/ chunks versus our keyword retrieval, on the team's test set.

Two steps, because embedding needs a GPU and our retrieval needs facts.db:

    python eval/compare_retrieval.py export   # -> eval/retrieval_compare/{docs.jsonl,queries.json}
    # on a GPU host, in a venv with torch + transformers:
    python eval/compare_retrieval.py embed eval/retrieval_compare   # -> ranked.json
    python eval/compare_retrieval.py score    # prints both retrievers' hit rates

The chunks are read from RAG/data/chunks/*.jsonl and re-embedded with the same model and query prompt the
ChromaDB index was built with (Qwen/Qwen3-Embedding-0.6B); the index's own pickle is never loaded.
A retriever scores a hit when, for a (question, company) pair, the expected PDF page is among its results, or the
expected number appears in their text (Indian or Western digit grouping).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from contextlib import closing
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "retrieval_compare"
NAMES = {"tcs": "TCS", "infy": "Infosys", "hcl": "HCLTech"}
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
TOP_K = 4   # per company, like our PER_COMPANY passage quota


def export() -> None:
    OUT.mkdir(exist_ok=True)
    with (OUT / "docs.jsonl").open("w", encoding="utf-8") as fh:
        for path in sorted((ROOT / "RAG" / "data" / "chunks").glob("*.jsonl")):
            for line in path.open(encoding="utf-8"):
                c = json.loads(line)
                fh.write(json.dumps({"id": c["id"], "company": c["metadata"]["company"], "pages": c["metadata"]["pages"],
                                     "text": c["text"], "embed": c["embed_text"]}, ensure_ascii=False) + "\n")
    queries = []
    for row in csv.DictReader((HERE / "test_queries.csv").open(encoding="utf-8")):
        pages = dict(p.split(":") for p in row["page"].split("|")) if ":" in row["page"] else {row["company_id"]: row["page"]}
        values = dict(p.split(":") for p in row["value"].split("|")) if ":" in row["value"] else {k: row["value"] for k in pages}
        queries.append({"query": row["query"], "companies": [NAMES[k] for k in row["company_id"].split("|")],
                        "pages": {NAMES[k]: int(p) for k, p in pages.items()},
                        "values": {NAMES[k]: v for k, v in values.items() if v not in NAMES}})
    (OUT / "queries.json").write_text(json.dumps(queries, ensure_ascii=False), encoding="utf-8")
    print(f"exported {sum(1 for _ in (OUT / 'docs.jsonl').open())} chunks and {len(queries)} queries to {OUT}")


def embed_and_rank(data: Path) -> None:
    import time

    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    docs = [json.loads(line) for line in (data / "docs.jsonl").open(encoding="utf-8")]
    queries = json.loads((data / "queries.json").read_text(encoding="utf-8"))
    path = Path(snapshot_download(EMBED_MODEL))
    prompt = json.loads((path / "config_sentence_transformers.json").read_text())["prompts"]["query"]
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModel.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()

    @torch.no_grad()
    def embed(texts: list[str], batch: int = 32) -> torch.Tensor:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = torch.empty(len(texts), model.config.hidden_size)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            b = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
            out[idx] = F.normalize(model(**b).last_hidden_state[:, -1].float(), dim=-1).cpu()   # last-token pooling
        return out

    t0 = time.time()
    D, Q = embed([d["embed"] for d in docs]), embed([prompt + q["query"] for q in queries])
    results = []
    for qi, q in enumerate(queries):
        for company in q["companies"]:
            idx = torch.tensor([i for i, d in enumerate(docs) if d["company"] == company])
            top = (D[idx] @ Q[qi]).topk(10)
            results.append({"q": qi, "company": company,
                            "top": [{"id": docs[idx[j]]["id"], "pages": docs[idx[j]]["pages"], "score": round(float(s), 4)}
                                    for s, j in zip(top.values.tolist(), top.indices.tolist())]})
    (data / "ranked.json").write_text(json.dumps({"model": EMBED_MODEL, "prompt": prompt, "seconds": round(time.time() - t0, 1),
                                                  "results": results}))
    print(f"embedded {len(docs)} chunks + {len(queries)} queries in {time.time() - t0:.1f}s")


def pages_of(pages: str) -> set[int]:
    first, _, last = str(pages).partition("-")
    return set(range(int(first), int(last or first) + 1))


def spellings(value: str) -> set[str]:
    """'267021' -> {'267,021', '2,67,021'}; decimals and small numbers as written."""
    if "." in value or float(value) < 1000:
        return {value}
    s = str(int(float(value)))
    head, tail, groups = s[:-3], s[-3:], []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return {f"{int(s):,}", ",".join(([head] if head else []) + groups + [tail])}


def value_in(texts: list[str], value: str | None) -> bool:
    return value is not None and any(
        re.search(rf"(?<![\d,.]){re.escape(s)}(?![\d,]|\.\d)", t) for s in spellings(value) for t in texts)


def score() -> None:
    sys.path.insert(0, str(ROOT))
    from app.db import connect
    from app.retrieval import build_sources

    docs = {d["id"]: d for d in map(json.loads, (OUT / "docs.jsonl").open(encoding="utf-8"))}
    queries = json.loads((OUT / "queries.json").read_text(encoding="utf-8"))
    ranked = json.loads((OUT / "ranked.json").read_text())["results"]
    hits = {"vectors": [0, 0], "keywords": [0, 0]}
    only = {"vectors": [], "keywords": []}
    with closing(connect(ROOT / "data" / "facts.db")) as conn:
        for r in ranked:
            q, company = queries[r["q"]], r["company"]
            page, value = q["pages"].get(company), q["values"].get(company)
            top = r["top"][:TOP_K]
            v_page = any(page in pages_of(t["pages"]) for t in top)
            v_value = value_in([docs[t["id"]]["text"] for t in top], value)
            _, sources = build_sources(conn, q["query"], [company])
            own = [s for s in sources if s["company"] == company]
            k_page = any(s["page"] == page for s in own)
            k_value = value_in([s["text"] for s in own], value)
            hits["vectors"][0] += v_page
            hits["vectors"][1] += v_value
            hits["keywords"][0] += k_page
            hits["keywords"][1] += k_value
            if v_value and not k_value:
                only["vectors"].append((r["q"] + 1, company))
            if k_value and not v_value:
                only["keywords"].append((r["q"] + 1, company))
    n = len(ranked)
    print(f"{n} (question, company) pairs; vectors: top {TOP_K} chunks per company; keywords: our sources for that company")
    print(f"  expected page retrieved : vectors {hits['vectors'][0]}/{n}, keywords {hits['keywords'][0]}/{n}")
    print(f"  expected value retrieved: vectors {hits['vectors'][1]}/{n}, keywords {hits['keywords'][1]}/{n}")
    print(f"  value found only by vectors : {only['vectors']}")
    print(f"  value found only by keywords: {only['keywords']}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "score"
    if step == "export":
        export()
    elif step == "embed":
        embed_and_rank(Path(sys.argv[2]) if len(sys.argv) > 2 else OUT)
    else:
        score()

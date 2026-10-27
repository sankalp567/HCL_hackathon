"""Step 3: embed chunks into a persistent Chroma collection via LangChain.

Reads data/chunks/*.jsonl, writes data/vectorstore/ (Chroma, cosine) and
data/vectorstore/embedding_info.json (model + settings; retrieval must load the same model).

  * Documents are embedded from `embed_text` (company/FY/part/headings + chunk text)
  * Queries must be embedded with the model's "query" prompt - get_embeddings() does this,
    so retrieval code should build its embeddings from here
  * The collection is rebuilt from scratch on every run, so it always matches the chunk files

Usage:  .venv-gpu\\Scripts\\python pipeline\\embed_chunks.py [--test]
"""
import argparse
import json
import sys
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHUNKS_DIR, COLLECTION, EMBED_MODEL, EMBED_QUERY_PROMPT, VECTOR_DIR

BATCH = 64
sys.stdout.reconfigure(encoding="utf-8")  # Windows console can't print ₹ otherwise


def get_embeddings(device=None):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {"device": device}
    if device == "cuda":
        model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        query_encode_kwargs={"normalize_embeddings": True, "prompt_name": EMBED_QUERY_PROMPT},
    )


def get_vectorstore(embeddings=None):
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(VECTOR_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_chunks():
    docs = []
    for path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                # Chroma stores page_content (what was embedded); raw text rides along for display
                meta = dict(c["metadata"], text=c["text"])
                docs.append(Document(id=c["id"], page_content=c["embed_text"], metadata=meta))
    return docs


def smoke_test(store):
    queries = [
        "What was the employee attrition rate?",
        "Consolidated revenue from operations for the year",
        "How is the company investing in generative AI?",
    ]
    for company in ("HCLTech", "TCS", "Infosys"):
        for q in queries:
            hits = store.similarity_search_with_relevance_scores(q, k=2, filter={"company": company})
            print(f"\n[{company}] {q}")
            for doc, score in hits:
                m = doc.metadata
                print(f"   {score:.3f}  {m['chunk_id']} | pages {m['pages']} | {m['chunk_type']} | {m['n_tokens']} tokens")
                print(f"          {m['text'][:140].replace(chr(10), ' ')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="only run retrieval smoke test on the existing store")
    args = ap.parse_args()

    embeddings = get_embeddings()
    if args.test:
        smoke_test(get_vectorstore(embeddings))
        return

    docs = load_chunks()
    if not docs:
        raise SystemExit(f"No chunks in {CHUNKS_DIR}; run chunk_reports.py first")
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids)), "duplicate chunk ids"

    store = get_vectorstore(embeddings)
    store.reset_collection()
    t = time.perf_counter()
    for i in range(0, len(docs), BATCH):
        store.add_documents(docs[i:i + BATCH], ids=ids[i:i + BATCH])
        print(f"\rembedded {min(i + BATCH, len(docs))}/{len(docs)}", end="", flush=True)
    elapsed = time.perf_counter() - t
    print(f"\nembedded {len(docs)} chunks in {elapsed:.0f}s")

    dim = len(embeddings.embed_query("dimension check"))
    info = {
        "model": EMBED_MODEL,
        "dimension": dim,
        "query_prompt": EMBED_QUERY_PROMPT,
        "normalized": True,
        "distance": "cosine",
        "collection": COLLECTION,
        "chunks": len(docs),
        "embedded_field": "embed_text",
        "seconds": round(elapsed, 1),
    }
    (VECTOR_DIR / "embedding_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    smoke_test(store)


if __name__ == "__main__":
    main()

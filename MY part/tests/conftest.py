import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proofirst_retrieval.contracts import Candidate  # noqa: E402


def mk(cid, text="t", **md):
    return Candidate(chunk_id=cid, text=text, metadata=md)


@pytest.fixture(scope="session")
def fixture_pipe():
    from proofirst_retrieval.benchmark import build_synthetic
    pipe, chunks, info = build_synthetic(ROOT / "data" / "synthetic_corpus.jsonl")
    return pipe, chunks

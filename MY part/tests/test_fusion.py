import pytest
from conftest import mk
from proofirst_retrieval.fusion import rrf_fuse, dedupe_ranked


def L(*ids):
    return [mk(i) for i in ids]


def test_rrf_known_example_k60_rank_is_one_based():
    a, b = L("x", "y", "z"), L("y", "x", "w")
    out = rrf_fuse([a, b], ["dense", "bm25"], k=60)
    s = {c.chunk_id: c.fusion_score for c in out}
    assert s["x"] == pytest.approx(1 / 61 + 1 / 62)
    assert s["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert s["z"] == pytest.approx(1 / 63)
    assert s["w"] == pytest.approx(1 / 63)
    assert out[0].chunk_id == "x"  # x and y tie on score; x first by best-rank then first-seen


def test_rrf_small_k_changes_weighting():
    out = rrf_fuse([L("a", "b"), L("b", "a")], k=0)
    assert {c.chunk_id: c.fusion_score for c in out} == {"a": pytest.approx(1 + 0.5), "b": pytest.approx(0.5 + 1)}


def test_empty_lists():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
    out = rrf_fuse([[], L("a")], ["dense", "bm25"])
    assert [c.chunk_id for c in out] == ["a"] and out[0].ranks == {"bm25": 1}


def test_duplicate_within_list_counts_once_best_rank():
    out = rrf_fuse([L("a", "b", "a")], k=60)
    assert len(out) == 2
    assert out[0].chunk_id == "a" and out[0].fusion_score == pytest.approx(1 / 61)


def test_dedupe_ranked_preserves_order():
    assert [c.chunk_id for c in dedupe_ranked(L("a", "b", "a", "c", "b"))] == ["a", "b", "c"]


def test_duplicate_across_lists_merged_and_provenance_preserved():
    c1 = mk("a", "txt", company="Acme", page=3); c1.scores = {"dense": -0.2, "dense_distance": 0.2}
    c2 = mk("a", "txt", company="Acme", page=3); c2.scores = {"bm25": 7.5}
    out = rrf_fuse([[c1], [c2, mk("b")]], ["dense", "bm25"])
    a = next(c for c in out if c.chunk_id == "a")
    assert a.ranks == {"dense": 1, "bm25": 1}
    assert a.scores["dense"] == -0.2 and a.scores["bm25"] == 7.5
    assert a.metadata == {"company": "Acme", "page": 3} and a.text == "txt"
    assert a.fusion_score == pytest.approx(2 / 61)


def test_inputs_not_mutated():
    c = mk("a"); c.ranks = {"dense": 9}
    rrf_fuse([[c]], ["dense"])
    assert c.fusion_score is None and c.ranks == {"dense": 9}


def test_stable_deterministic_ordering_on_ties():
    runs = [[c.chunk_id for c in rrf_fuse([L("a", "b"), L("c", "d")])] for _ in range(5)]
    assert all(r == runs[0] for r in runs)
    assert runs[0] == ["a", "c", "b", "d"]


def test_weights_and_validation():
    out = rrf_fuse([L("a"), L("b")], weights=[1.0, 3.0])
    assert out[0].chunk_id == "b"
    with pytest.raises(ValueError):
        rrf_fuse([L("a")], names=["x", "y"])
    with pytest.raises(ValueError):
        rrf_fuse([L("a")], k=-1)


def test_top_n():
    assert len(rrf_fuse([L("a", "b", "c")], top_n=2)) == 2


def test_top_n_zero_returns_nothing():
    assert rrf_fuse([L("a", "b")], top_n=0) == []

"""Tests for the RagTrace IR: coercion, accessors, and round-trip serialization."""

from raglens.ir import SCHEMA_VERSION, Attribution, Chunk, RagTrace, Stage


def test_chunk_from_raw_string():
    c = Chunk.from_raw("hello world", 0)
    assert c.text == "hello world"
    assert c.id.startswith("c0_")


def test_chunk_from_raw_dict_variants():
    c = Chunk.from_raw({"content": "x", "score": 0.5, "source": "a.md"}, 1)
    assert c.text == "x" and c.score == 0.5 and c.source == "a.md"
    c2 = Chunk.from_raw({"page_content": "y", "metadata": {"k": 1}}, 2)
    assert c2.text == "y" and c2.metadata == {"k": 1}


def test_chunk_from_raw_document_like():
    class Doc:
        page_content = "doc text"
        metadata = {"src": "z"}

    c = Chunk.from_raw(Doc(), 0)
    assert c.text == "doc text" and c.metadata == {"src": "z"}


def test_chunk_id_deterministic():
    assert Chunk.from_raw("same", 0).id == Chunk.from_raw("same", 0).id


def _sample_trace() -> RagTrace:
    t = RagTrace(query_id="q1", query="hi", timestamp="2026-01-01T00:00:00Z")
    chunks = [Chunk("c0_a", "alpha", source="a"), Chunk("c1_b", "beta", source="b")]
    t.stages.append(Stage("retrieval", {"query": "hi", "k": 2, "candidates": [c.to_dict() for c in chunks]}, 12.0))
    t.stages.append(Stage("context_assembly", {"final_chunk_ids": ["c0_a", "c1_b"], "tokens": 5, "truncated": False}))
    t.stages.append(Stage("generation", {"model": "m", "answer": "alpha", "cost_usd": 0.01}, 30.0))
    t.answer = "alpha"
    t.attributions.append(Attribution("c0_a", 0.9))
    return t


def test_accessors():
    t = _sample_trace()
    assert t.stage("generation").data["model"] == "m"
    assert [c.id for c in t.retrieved_chunks()] == ["c0_a", "c1_b"]
    assert [c.id for c in t.final_chunks()] == ["c0_a", "c1_b"]
    assert t.total_latency_ms() == 42.0
    assert t.total_cost_usd() == 0.01


def test_roundtrip_serialization():
    t = _sample_trace()
    d = t.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    t2 = RagTrace.from_dict(d)
    assert t2.to_dict() == d
    assert t2.query == "hi"
    assert len(t2.stages) == 3
    assert t2.attributions[0].chunk_id == "c0_a"

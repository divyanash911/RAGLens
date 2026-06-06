"""Tests for capture frontends (decorators, wrap, run) and attribution backend."""

import pytest

import raglens
from raglens import trace
from raglens.attribution import AttributionError, attribute, lexical_similarity


# A deterministic toy pipeline where exactly one chunk is decisive.
DOCS = {
    "key": "The capital of France is Paris.",
    "noise1": "Bananas are yellow.",
    "noise2": "The ocean is large and blue.",
}


def _retrieve(query):
    return [{"text": t, "source": s} for s, t in DOCS.items()]


def _generate(query, chunks):
    # Answers "Paris" only if the decisive chunk is present; else "unknown".
    for ch in chunks:
        text = ch["text"] if isinstance(ch, dict) else str(ch)
        if "Paris" in text:
            return "Paris"
    return "unknown"


def test_decorator_capture_records_stages():
    r = trace.retriever(_retrieve)
    g = trace.generator(_generate)
    with raglens.capture(config={"embedding_model": "toy"}) as cap:
        chunks = r("capital of France?")
        ans = g("capital of France?", chunks)
    assert ans == "Paris"
    assert cap.trace.stage("retrieval") is not None
    assert cap.trace.stage("generation") is not None
    assert cap.trace.answer == "Paris"
    assert cap.trace.config_fingerprint is not None


def test_decorators_are_passthrough_without_capture():
    r = trace.retriever(_retrieve)
    # Outside a capture, behaves like the original function and records nothing.
    assert len(r("q")) == 3
    assert raglens.active_capture() is None


def test_attribution_identifies_decisive_chunk():
    pipe = raglens.wrap(retrieve=_retrieve, generate=_generate, config={"x": 1})
    cap = pipe.run("capital of France?")
    attrs = cap.attribute().trace.attributions
    assert len(attrs) == 3
    by_chunk = {a.chunk_id: a for a in attrs}
    # The chunk containing "Paris" should have importance ~1.0; noise ~0.0.
    final = cap.trace.final_chunks()
    paris_id = next(c.id for c in final if "Paris" in c.text)
    noise_ids = [c.id for c in final if "Paris" not in c.text]
    assert by_chunk[paris_id].score > 0.5
    for nid in noise_ids:
        assert by_chunk[nid].score < 0.5


def test_run_helper_with_plain_callables():
    cap = raglens.run("capital of France?", _retrieve, _generate, config={"x": 1})
    assert cap.trace.answer == "Paris"
    cap.attribute()
    assert any(a.score > 0.5 for a in cap.trace.attributions)


def test_attribution_requires_generator_handle():
    # Build a trace by hand with no runtime generator → attribution must error.
    cap = raglens.Capture(query="q", config={})
    cap.trace.answer = "x"
    with pytest.raises(AttributionError):
        attribute(cap)


def test_lexical_similarity_bounds():
    assert lexical_similarity("a b c", "a b c") == 1.0
    assert lexical_similarity("", "") == 1.0
    assert lexical_similarity("a", "") == 0.0
    assert 0.0 <= lexical_similarity("a b c", "a b x") <= 1.0


def test_top_chunks_ordering():
    pipe = raglens.wrap(retrieve=_retrieve, generate=_generate)
    cap = pipe.run("capital of France?")
    cap.attribute()
    ranked = raglens.top_chunks(cap, n=3)
    scores = [rc.score for rc in ranked]
    assert scores == sorted(scores, reverse=True)

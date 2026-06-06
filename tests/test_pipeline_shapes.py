"""Tests that the decorators handle the *variety* of real RAG code shapes:
instance methods, custom argument names, keyword calls, async, extra kwargs,
and the manual escape hatch for non-decomposable pipelines.
"""

import asyncio

import pytest

import raglens
from raglens import trace

DOCS = [
    {"text": "The capital of France is Paris.", "source": "geo"},
    {"text": "Bananas are yellow.", "source": "noise"},
]


def _decisive(chunks):
    for ch in chunks:
        text = ch["text"] if isinstance(ch, dict) else str(ch)
        if "Paris" in text:
            return "Paris"
    return "unknown"


def test_instance_method_query_not_confused_with_self():
    """Regression: self must be skipped so the query isn't bound to the instance."""

    class Pipe:
        @trace.retriever
        def retrieve(self, query):
            return DOCS

        @trace.generator
        def generate(self, query, chunks):
            return _decisive(chunks)

    p = Pipe()
    with raglens.capture(config={"x": 1}) as cap:
        chunks = p.retrieve("capital of France?")
        ans = p.generate("capital of France?", chunks)

    assert ans == "Paris"
    # The recorded query must be the real query string, not the Pipe instance.
    assert cap.trace.query == "capital of France?"
    assert cap.trace.stage("retrieval").data["query"] == "capital of France?"


def test_custom_argument_names():
    @trace.retriever(query="question")
    def retrieve(question, top_k=2):
        return DOCS[:top_k]

    @trace.generator(query="question", chunks="context")
    def generate(question, context, temperature=0.0):
        return _decisive(context)

    with raglens.capture() as cap:
        chunks = retrieve("capital of France?")
        generate("capital of France?", chunks)

    cap.attribute()
    assert cap.trace.answer == "Paris"
    # Attribution must still find the decisive chunk despite the extra
    # `temperature` kwarg — regenerate preserves it.
    paris_id = next(c.id for c in cap.trace.final_chunks() if "Paris" in c.text)
    by = {a.chunk_id: a.score for a in cap.trace.attributions}
    assert by[paris_id] > 0.5


def test_keyword_invocation():
    @trace.generator
    def generate(query, chunks):
        return _decisive(chunks)

    @trace.retriever
    def retrieve(query):
        return DOCS

    with raglens.capture() as cap:
        chunks = retrieve(query="capital of France?")
        generate(query="capital of France?", chunks=chunks)

    assert cap.trace.query == "capital of France?"
    assert cap.trace.answer == "Paris"


def test_async_pipeline():
    @trace.retriever
    async def retrieve(query):
        await asyncio.sleep(0)
        return DOCS

    @trace.generator
    async def generate(query, chunks):
        await asyncio.sleep(0)
        return _decisive(chunks)

    async def run():
        with raglens.capture() as cap:
            chunks = await retrieve("capital of France?")
            await generate("capital of France?", chunks)
        return cap

    cap = asyncio.run(run())
    assert cap.trace.answer == "Paris"
    # Attribution re-runs the async generator synchronously.
    cap.attribute()
    paris_id = next(c.id for c in cap.trace.final_chunks() if "Paris" in c.text)
    by = {a.chunk_id: a.score for a in cap.trace.attributions}
    assert by[paris_id] > 0.5


def test_manual_escape_hatch_for_nondecomposable_pipeline():
    """A monolithic answer() function instrumented by hand, with attribution."""

    def answer(query, chunks):
        return _decisive(chunks)

    with raglens.capture(config={"x": 1}) as cap:
        chunks = DOCS
        result = answer("capital of France?", chunks)
        trace.record_retrieval("capital of France?", chunks)
        trace.record_generation(
            "capital of France?",
            chunks,
            result,
            regenerate=lambda new: answer("capital of France?", new),
        )

    cap.attribute()
    assert cap.trace.answer == "Paris"
    assert len(cap.trace.attributions) == 2


def test_generator_without_chunk_arg_degrades_gracefully():
    """If no chunk argument can be identified, capture/doc still work; attribution
    raises a clear, actionable error rather than guessing."""

    @trace.generator
    def generate(prompt):  # pre-assembled prompt, no separable chunks
        return "some answer"

    with raglens.capture() as cap:
        generate("a fully assembled prompt with context baked in")

    assert cap.trace.answer == "some answer"
    with pytest.raises(raglens.AttributionError):
        cap.attribute()

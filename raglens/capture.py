"""Capture — the runtime object frontends populate and backends consume.

A ``Capture`` is created for one query. While it is the *active* capture (tracked
via a ``ContextVar`` so it is async/thread-safe), the ``@trace.retriever`` and
``@trace.generator`` decorators and the auto-instrumentation hooks append stages
to it. When the query finishes, the ``Capture`` holds:

  - ``trace``: the serializable ``RagTrace`` (what gets written to JSONL), and
  - runtime handles (the live ``generate`` callable, the exact chunk objects that
    were passed to it, the original answer) that are *not* serialized but are
    needed by counterfactual attribution, which must re-run generation.

This separation is why attribution is computed in-process at capture time rather
than from a static trace file: the causal re-generation needs the live callable.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional

from .fingerprint import config_fingerprint
from .ir import Chunk, RagTrace, Stage

# The currently active capture, if any. None means tracing is a no-op (so the
# user's decorated functions still run unchanged outside a capture — NFR1.4).
_active: ContextVar[Optional["Capture"]] = ContextVar("raglens_active_capture", default=None)


def active_capture() -> Optional["Capture"]:
    """Return the capture currently being recorded into, or None."""
    return _active.get()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Capture:
    """Accumulates one query's stages into a RagTrace plus runtime handles."""

    def __init__(self, query: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.trace = RagTrace(
            query_id=uuid.uuid4().hex,
            query=query or "",
            timestamp=_now_iso(),
            config=config,
            config_fingerprint=config_fingerprint(config) if config else None,
        )
        # Runtime-only handles for attribution (never serialized).
        # ``_regenerate`` is a closure that re-runs the user's generator with a
        # *substituted* chunk list while preserving every other argument it was
        # originally called with (self, prompt, temperature, ...). This is what
        # makes attribution robust to arbitrary generator signatures.
        self._regenerate: Optional[Callable[[List[Any]], Any]] = None
        self._final_chunks_raw: List[Any] = []
        self._final_chunks: List[Chunk] = []

    # ------------------------------------------------------------------ #
    # Frontend-facing recording API (called by decorators / wrap / instrument).
    # ------------------------------------------------------------------ #
    def set_query(self, query: str) -> None:
        if query and not self.trace.query:
            self.trace.query = query

    def record_query_transform(self, original: str, transformed: str, latency_ms: Optional[float] = None) -> None:
        self.trace.stages.append(
            Stage(kind="query_transform", data={"original": original, "transformed": transformed}, latency_ms=latency_ms)
        )

    def record_retrieval(self, query: str, raw_chunks: List[Any], latency_ms: Optional[float] = None) -> List[Chunk]:
        self.set_query(query)
        chunks = [Chunk.from_raw(c, i) for i, c in enumerate(raw_chunks)]
        self.trace.stages.append(
            Stage(
                kind="retrieval",
                data={"query": query, "k": len(chunks), "candidates": [c.to_dict() for c in chunks]},
                latency_ms=latency_ms,
            )
        )
        return chunks

    def record_generation(
        self,
        query: str,
        raw_chunks: List[Any],
        answer: str,
        *,
        regenerate: Optional[Callable[[List[Any]], Any]] = None,
        model: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Record the generation stage.

        ``regenerate`` is an optional closure that, given a new chunk list,
        re-runs the user's generator with that list substituted for the original
        chunks (preserving all other call arguments). It is what counterfactual
        attribution uses; if it is None, attribution will be unavailable for this
        trace but capture/documentation still work.
        """
        self.set_query(query)
        chunks = [Chunk.from_raw(c, i) for i, c in enumerate(raw_chunks)]
        self._final_chunks = chunks
        self._final_chunks_raw = list(raw_chunks)
        self._regenerate = regenerate

        # Record context assembly (which chunks actually reached the generator).
        self.trace.stages.append(
            Stage(
                kind="context_assembly",
                data={
                    "final_chunk_ids": [c.id for c in chunks],
                    "tokens": _approx_tokens(" ".join(c.text for c in chunks)),
                    "truncated": False,
                },
            )
        )
        self.trace.stages.append(
            Stage(
                kind="generation",
                data={
                    "model": model,
                    "answer": answer,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "cost_usd": None,
                    "logprobs": None,
                },
                latency_ms=latency_ms,
            )
        )
        self.trace.answer = answer

    def record_llm_usage(
        self,
        *,
        model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Attach LLM token/cost usage to the generation stage (FR1.5).

        Called by the SDK auto-instrumentation. If a generation stage does not
        exist yet (LLM call happened inside the generator before it returned),
        the usage is buffered onto a placeholder stage that ``record_generation``
        will merge into.
        """
        gen = self.trace.stage("generation")
        if gen is None:
            gen = Stage(kind="generation", data={})
            self.trace.stages.append(gen)
        d = gen.data
        if model is not None:
            d["model"] = model
        if prompt_tokens is not None:
            d["prompt_tokens"] = (d.get("prompt_tokens") or 0) + prompt_tokens
        if completion_tokens is not None:
            d["completion_tokens"] = (d.get("completion_tokens") or 0) + completion_tokens
        if cost_usd is not None:
            d["cost_usd"] = (d.get("cost_usd") or 0.0) + cost_usd
        if latency_ms is not None and gen.latency_ms is None:
            gen.latency_ms = latency_ms

    # ------------------------------------------------------------------ #
    # Backend conveniences.
    # ------------------------------------------------------------------ #
    def attribute(self, **kwargs: Any) -> "Capture":
        """Run counterfactual attribution and store results on the trace.

        Thin wrapper around :func:`raglens.attribution.attribute` so users can do
        ``cap.attribute()`` fluently. Requires that generation was captured via a
        re-callable generator (decorator or ``wrap``).
        """
        from .attribution import attribute as _attribute

        _attribute(self, **kwargs)
        return self

    def save(self, path: str, append: bool = True) -> "Capture":
        """Persist this capture's trace to JSONL (FR1.3)."""
        from .writer import append_trace, write_traces

        if append:
            append_trace(path, self.trace)
        else:
            write_traces(path, [self.trace], append=False)
        return self


@contextmanager
def capture(query: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Iterator[Capture]:
    """Context manager that makes a fresh ``Capture`` active for its body.

    Example::

        with raglens.capture(config=cfg) as cap:
            chunks = retrieve(q)          # @trace.retriever records into cap
            answer = generate(q, chunks)  # @trace.generator records into cap
        cap.attribute().save("traces.jsonl")
    """
    cap = Capture(query=query, config=config)
    token = _active.set(cap)
    try:
        yield cap
    finally:
        _active.reset(token)


def run(
    query: str,
    retrieve: Callable[..., Any],
    generate: Callable[..., Any],
    *,
    config: Optional[Dict[str, Any]] = None,
    query_transform: Optional[Callable[[str], str]] = None,
) -> Capture:
    """Run a retrieve→generate pipeline once under a capture and return it.

    ``retrieve`` and ``generate`` may be plain (undecorated) callables; this
    function records the stages itself, so it works whether or not the user used
    the decorators. The generator is invoked as ``generate(query, chunks)``; for
    generators with other signatures, decorate them with ``@trace.generator`` (it
    binds arguments by name) or pass a ``wrap``ped pipeline.
    """
    cap = Capture(query=query, config=config)
    token = _active.set(cap)
    try:
        q = query
        if query_transform is not None:
            t0 = time.perf_counter()
            q = query_transform(query)
            cap.record_query_transform(query, q, latency_ms=(time.perf_counter() - t0) * 1000)

        # Retrieve (record even if the callable is undecorated).
        t0 = time.perf_counter()
        raw_chunks = list(retrieve(q))
        retr_ms = (time.perf_counter() - t0) * 1000
        if cap.trace.stage("retrieval") is None:
            cap.record_retrieval(q, raw_chunks, latency_ms=retr_ms)

        # Generate.
        t0 = time.perf_counter()
        answer = generate(q, raw_chunks)
        gen_ms = (time.perf_counter() - t0) * 1000
        if cap.trace.stage("generation") is None:
            cap.record_generation(
                q,
                raw_chunks,
                _as_text(answer),
                regenerate=lambda new_chunks, _q=q: generate(_q, new_chunks),
                latency_ms=gen_ms,
            )
        else:
            # Generator was decorated; it already installed a regenerate closure.
            cap._final_chunks_raw = cap._final_chunks_raw or list(raw_chunks)
        return cap
    finally:
        _active.reset(token)


def _as_text(answer: Any) -> str:
    """Coerce a generator's return value into answer text."""
    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        return answer.get("answer") or answer.get("text") or answer.get("content") or str(answer)
    return str(answer)


def _approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used when no tokenizer is present."""
    return max(1, len(text) // 4) if text else 0

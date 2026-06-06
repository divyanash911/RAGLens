"""Counterfactual chunk attribution (FR1.6) — the headline explainability backend.

For each retrieved chunk we *remove it from the context and re-run generation*,
then measure how much the answer changed. A large change means the chunk was
causally important to the answer; no change means it was inert (or redundant).

This is faithful by construction: the score is produced by an actual
counterfactual re-generation, not self-reported by the model (which hallucinates
citations) and not an attention/gradient proxy. It is the ablation idea behind
ContextCite, kept deliberately simple for v1. Cost is O(num_chunks) extra
generations per query; v2 reduces this via grouped/batched ablation (FR2.8).

The similarity comparator is pluggable. The default is dependency-free lexical
similarity (``difflib``); users with embeddings can pass a semantic comparator.
"""

from __future__ import annotations

import difflib
from typing import Any, Callable, List, Optional

from .capture import Capture, _as_text
from .ir import Attribution, Chunk

# A comparator maps (original_answer, ablated_answer) -> similarity in [0, 1].
Comparator = Callable[[str, str], float]


def lexical_similarity(a: str, b: str) -> float:
    """Token-aware lexical similarity in [0, 1] using difflib (stdlib, no deps)."""
    a = (a or "").strip()
    b = (b or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


def attribute(
    capture: Capture,
    *,
    comparator: Optional[Comparator] = None,
    include_ablated_answer: bool = False,
    max_chunks: Optional[int] = None,
    groundedness: bool = True,
) -> List[Attribution]:
    """Compute and attach counterfactual attributions to ``capture.trace``.

    Args:
        capture: a Capture whose generation was recorded with a re-callable
            generator (via the decorators, ``wrap``, or ``raglens.run``).
        comparator: similarity function; defaults to :func:`lexical_similarity`.
            Importance score is ``1 - similarity(original, ablated)``.
        include_ablated_answer: store each ablated answer on the Attribution
            (useful for debugging; larger traces).
        max_chunks: cap the number of chunks ablated (cost control).
        groundedness: also run ONE no-context counterfactual (regenerate with zero
            chunks) and record a groundedness score in ``trace.diagnostics``.
            ``groundedness = 1 - similarity(answer, answer_without_any_context)``;
            a low value means the answer barely depends on retrieval (parametric /
            hallucination risk). Costs one extra generation.

    Returns:
        The list of :class:`raglens.ir.Attribution` (also stored on the trace).

    Raises:
        AttributionError: if there is no re-callable generator to ablate against.
    """
    comparator = comparator or lexical_similarity
    regenerate = capture._regenerate
    if regenerate is None:
        raise AttributionError(
            "Counterfactual attribution needs a way to re-run generation. Use "
            "@trace.generator, raglens.wrap(...), or raglens.run(...) — or pass a "
            "`regenerate` closure to raglens.trace.record_generation(...) — so the "
            "generator can be re-invoked with ablated chunks. (If your generator "
            "takes a pre-assembled prompt rather than a chunk argument, name the "
            "chunk argument via @trace.generator(chunks='...').)"
        )

    chunks = capture._final_chunks
    raw_chunks = capture._final_chunks_raw
    original_answer = capture.trace.answer
    if original_answer is None:
        raise AttributionError("No original answer recorded; cannot measure ablation deltas.")
    if not chunks:
        return []

    n = len(raw_chunks)
    limit = n if max_chunks is None else min(max_chunks, n)

    attributions: List[Attribution] = []
    for i in range(limit):
        ablated_raw = [c for j, c in enumerate(raw_chunks) if j != i]
        ablated_answer = _as_text(_regenerate(regenerate, ablated_raw))
        sim = comparator(original_answer, ablated_answer)
        importance = max(0.0, min(1.0, 1.0 - sim))
        attributions.append(
            Attribution(
                chunk_id=chunks[i].id,
                score=round(importance, 6),
                method="counterfactual_ablation",
                ablated_similarity=round(sim, 6),
                ablated_answer=ablated_answer if include_ablated_answer else None,
            )
        )

    capture.trace.attributions = attributions

    # Groundedness probe: how much does the answer rely on retrieval at all?
    if groundedness:
        try:
            no_ctx = _as_text(_regenerate(regenerate, []))
            g = max(0.0, min(1.0, 1.0 - comparator(original_answer, no_ctx)))
            capture.trace.diagnostics["groundedness"] = round(g, 6)
            capture.trace.diagnostics["answer_without_context"] = no_ctx
        except Exception:
            # Some generators can't run with empty context; skip rather than fail.
            pass

    return attributions


def _regenerate(regenerate: Callable[[List[Any]], Any], chunks: List[Any]) -> Any:
    """Re-invoke the user's generator with an ablated chunk set.

    We re-run *without* an active capture so the ablation generations do not
    pollute the trace being explained. Async generators are supported by running
    the returned coroutine to completion (best effort).
    """
    import asyncio
    import inspect

    from .capture import _active

    token = _active.set(None)
    try:
        result = regenerate(chunks)
        if inspect.iscoroutine(result):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(result)
            raise AttributionError(
                "Cannot run attribution on an async generator from inside a running "
                "event loop. Call attribute() from synchronous code, or precompute "
                "ablations yourself."
            )
        return result
    finally:
        _active.reset(token)


def top_chunks(trace_or_capture: Any, n: int = 3) -> List["RankedChunk"]:
    """Return the ``n`` most influential chunks (by attribution score)."""
    trace = getattr(trace_or_capture, "trace", trace_or_capture)
    by_id = {c.id: c for c in trace.final_chunks()}
    ranked = sorted(trace.attributions, key=lambda a: a.score, reverse=True)
    out: List[RankedChunk] = []
    for a in ranked[:n]:
        out.append(RankedChunk(chunk=by_id.get(a.chunk_id), attribution=a))
    return out


class RankedChunk:
    """A chunk paired with its attribution, for presentation."""

    def __init__(self, chunk: Optional[Chunk], attribution: Attribution):
        self.chunk = chunk
        self.attribution = attribution

    @property
    def score(self) -> float:
        return self.attribution.score


class AttributionError(RuntimeError):
    """Raised when counterfactual attribution cannot be computed."""

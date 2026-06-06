"""Decorators frontend (Tier 1): ``@trace.retriever`` / ``@trace.generator``.

The lowest-friction way to instrument a hand-rolled pipeline (FR1.1): add two
decorators to functions you already have. Outside an active capture they are
pure pass-through, so decorated functions behave identically in untraced paths.

Designed for the *variety* of ways people actually write RAG code:

- **Arguments are bound by signature**, not by fragile positional guessing — so
  instance/class methods (``self``/``cls`` are skipped), keyword calls, and
  defaulted parameters all resolve correctly.
- **Argument names are configurable** when the heuristics don't fit::

      @trace.retriever(query="question")
      @trace.generator(query="question", chunks="context")

- **Async functions are supported** (``async def`` retrievers/generators).
- For attribution, the generator decorator captures a *regenerate closure* that
  re-invokes your function with the chunk argument swapped out while preserving
  every other argument (prompt, temperature, ``self``, …). Attribution therefore
  works regardless of the generator's signature, not only for ``(query, chunks)``.

If your pipeline doesn't decompose into a retriever and a generator at all (e.g.
a single ``answer(query)`` function, or a declarative chain), use ``raglens.wrap``
or the manual escape hatch at the bottom of this module
(:func:`record_retrieval` / :func:`record_generation`).
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, List, Optional, Tuple

from .capture import active_capture, _as_text

# Heuristic parameter names, tried in order when no explicit name is given.
_QUERY_NAMES = ("query", "question", "q", "prompt", "input", "text", "user_input", "search")
_CHUNK_NAMES = ("chunks", "context", "contexts", "documents", "docs", "passages", "sources", "retrieved", "results")


# --------------------------------------------------------------------------- #
# Argument binding (signature-aware, self/cls-skipping).
# --------------------------------------------------------------------------- #
def _positional_params(sig: inspect.Signature) -> List[inspect.Parameter]:
    """Parameters that can take a value, with a leading self/cls dropped."""
    params = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if params and params[0].name in ("self", "cls"):
        params = params[1:]
    return params


def _resolve_name(
    bound: inspect.BoundArguments,
    params: List[inspect.Parameter],
    candidates: Tuple[str, ...],
    explicit: Optional[str],
    position: int,
) -> Optional[str]:
    """Resolve which parameter holds a value, returning its *name* (or None).

    Resolution order: explicit name → known candidate names present in the call →
    the parameter at ``position`` among the positional params (self-skipped).
    """
    if explicit is not None:
        return explicit
    for name in candidates:
        if name in bound.arguments:
            return name
    if len(params) > position:
        return params[position].name
    return None


def _bind(fn: Callable[..., Any], args: tuple, kwargs: dict) -> Tuple[inspect.Signature, inspect.BoundArguments]:
    sig = inspect.signature(fn)
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    return sig, bound


def _value(bound: inspect.BoundArguments, name: Optional[str]) -> Any:
    if name is None:
        return None
    return bound.arguments.get(name)


# --------------------------------------------------------------------------- #
# Retriever decorator.
# --------------------------------------------------------------------------- #
def _make_retriever(fn: Callable[..., Any], query_arg: Optional[str]) -> Callable[..., Any]:
    is_async = inspect.iscoroutinefunction(fn)

    def _record(args: tuple, kwargs: dict, result: Any, latency_ms: float) -> None:
        cap = active_capture()
        if cap is None:
            return
        sig, bound = _bind(fn, args, kwargs)
        params = _positional_params(sig)
        qname = _resolve_name(bound, params, _QUERY_NAMES, query_arg, position=0)
        query = _as_text(_value(bound, qname)) if qname else ""
        cap.record_retrieval(query, list(result), latency_ms=latency_ms)

    if is_async:
        @functools.wraps(fn)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            if active_capture() is None:
                return await fn(*args, **kwargs)
            t0 = _now()
            result = await fn(*args, **kwargs)
            _record(args, kwargs, result, (_now() - t0) * 1000)
            return result

        awrapper.__raglens_role__ = "retriever"  # type: ignore[attr-defined]
        return awrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if active_capture() is None:
            return fn(*args, **kwargs)
        t0 = _now()
        result = fn(*args, **kwargs)
        _record(args, kwargs, result, (_now() - t0) * 1000)
        return result

    wrapper.__raglens_role__ = "retriever"  # type: ignore[attr-defined]
    return wrapper


def retriever(fn: Optional[Callable[..., Any]] = None, *, query: Optional[str] = None) -> Callable[..., Any]:
    """Mark a function as the pipeline's retriever and capture its output.

    Usable bare (``@trace.retriever``) or parametrized
    (``@trace.retriever(query="question")``). The wrapped function must return an
    iterable of chunks (strings, dicts, or Document-like objects — see
    :meth:`raglens.ir.Chunk.from_raw`).
    """
    if fn is None:
        return lambda f: _make_retriever(f, query)
    return _make_retriever(fn, query)


# --------------------------------------------------------------------------- #
# Generator decorator.
# --------------------------------------------------------------------------- #
def _make_generator(fn: Callable[..., Any], query_arg: Optional[str], chunks_arg: Optional[str]) -> Callable[..., Any]:
    is_async = inspect.iscoroutinefunction(fn)

    def _record(args: tuple, kwargs: dict, result: Any, latency_ms: float) -> None:
        cap = active_capture()
        if cap is None:
            return
        sig, bound = _bind(fn, args, kwargs)
        params = _positional_params(sig)
        qname = _resolve_name(bound, params, _QUERY_NAMES, query_arg, position=0)
        cname = _resolve_name(bound, params, _CHUNK_NAMES, chunks_arg, position=1)
        query = _as_text(_value(bound, qname)) if qname else ""
        chunks_val = _value(bound, cname)
        chunks = list(chunks_val) if chunks_val is not None else []

        # Build a regenerate closure: re-run fn with the chunk arg replaced and
        # everything else (self, prompt, temperature, ...) preserved. If we could
        # not identify a chunk argument, attribution is unavailable (regenerate
        # stays None) but capture/documentation still work.
        regenerate = None
        if cname is not None:
            def regenerate(new_chunks: List[Any], _a=args, _k=kwargs, _c=cname) -> Any:
                _, b = _bind(fn, _a, _k)
                b.arguments[_c] = new_chunks
                return fn(*b.args, **b.kwargs)

        cap.record_generation(
            query, chunks, _as_text(result), regenerate=regenerate, latency_ms=latency_ms
        )

    if is_async:
        @functools.wraps(fn)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            if active_capture() is None:
                return await fn(*args, **kwargs)
            t0 = _now()
            result = await fn(*args, **kwargs)
            _record(args, kwargs, result, (_now() - t0) * 1000)
            return result

        awrapper.__raglens_role__ = "generator"  # type: ignore[attr-defined]
        return awrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if active_capture() is None:
            return fn(*args, **kwargs)
        t0 = _now()
        result = fn(*args, **kwargs)
        _record(args, kwargs, result, (_now() - t0) * 1000)
        return result

    wrapper.__raglens_role__ = "generator"  # type: ignore[attr-defined]
    return wrapper


def generator(
    fn: Optional[Callable[..., Any]] = None,
    *,
    query: Optional[str] = None,
    chunks: Optional[str] = None,
) -> Callable[..., Any]:
    """Mark a function as the pipeline's generator and capture its output.

    Usable bare (``@trace.generator``) or parametrized
    (``@trace.generator(query="question", chunks="context")``). The decorator
    binds arguments by signature, so it handles methods and custom argument
    names; for counterfactual attribution it re-invokes the function with the
    chunk argument swapped out, preserving all other arguments.
    """
    if fn is None:
        return lambda f: _make_generator(f, query, chunks)
    return _make_generator(fn, query, chunks)


# --------------------------------------------------------------------------- #
# Optional query-transform decorator.
# --------------------------------------------------------------------------- #
def query_transform(fn: Optional[Callable[..., Any]] = None, *, query: Optional[str] = None) -> Callable[..., Any]:
    """Mark a query-rewriting step (records a ``query_transform`` stage)."""

    def _make(f: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cap = active_capture()
            if cap is None:
                return f(*args, **kwargs)
            sig, bound = _bind(f, args, kwargs)
            params = _positional_params(sig)
            qname = _resolve_name(bound, params, _QUERY_NAMES, query, position=0)
            original = _as_text(_value(bound, qname)) if qname else ""
            t0 = _now()
            result = f(*args, **kwargs)
            cap.record_query_transform(original, _as_text(result), latency_ms=(_now() - t0) * 1000)
            return result

        wrapper.__raglens_role__ = "query_transform"  # type: ignore[attr-defined]
        return wrapper

    return _make if fn is None else _make(fn)


# --------------------------------------------------------------------------- #
# Manual escape hatch — for pipelines that don't fit the decorator model.
# --------------------------------------------------------------------------- #
def record_retrieval(query: str, chunks: List[Any]) -> None:
    """Manually record a retrieval stage into the active capture (no-op if none)."""
    cap = active_capture()
    if cap is not None:
        cap.record_retrieval(query, list(chunks))


def record_generation(
    query: str,
    chunks: List[Any],
    answer: Any,
    *,
    regenerate: Optional[Callable[[List[Any]], Any]] = None,
    model: Optional[str] = None,
) -> None:
    """Manually record a generation stage. Pass ``regenerate`` to enable attribution.

    ``regenerate(new_chunks) -> answer`` should re-run your generation with a
    different chunk set; supplying it is what lets counterfactual attribution run
    for fully custom pipelines.
    """
    cap = active_capture()
    if cap is not None:
        cap.record_generation(query, list(chunks), _as_text(answer), regenerate=regenerate, model=model)


def _now() -> float:
    import time

    return time.perf_counter()

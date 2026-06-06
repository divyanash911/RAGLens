"""``raglens.wrap`` adapter (FR1.2) — instrument without editing source.

Some users would rather not annotate their functions (they may not own the
source, or prefer to keep instrumentation external). ``wrap`` lets them point at
existing callables and get a runnable, traced pipeline object.

Example::

    pipe = raglens.wrap(retrieve=my_search, generate=my_answer,
                        config={"embedding_model": "bge-small"})
    cap = pipe.run("What is RAG?")
    cap.attribute().save("traces.jsonl")
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .capture import Capture, run as _run


class Pipeline:
    """A thin, traced facade over a user's retrieve/generate callables."""

    def __init__(
        self,
        retrieve: Callable[..., Any],
        generate: Callable[..., Any],
        *,
        query_transform: Optional[Callable[[str], str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if retrieve is None or generate is None:
            raise ValueError("wrap() requires both `retrieve` and `generate` callables")
        self.retrieve = retrieve
        self.generate = generate
        self.query_transform = query_transform
        self.config = config or {}

    def run(self, query: str, *, config: Optional[Dict[str, Any]] = None) -> Capture:
        """Execute the pipeline once for ``query`` and return the Capture."""
        return _run(
            query,
            self.retrieve,
            self.generate,
            config=config if config is not None else self.config,
            query_transform=self.query_transform,
        )


def wrap(
    pipeline: Optional[Any] = None,
    *,
    retrieve: Optional[Callable[..., Any]] = None,
    generate: Optional[Callable[..., Any]] = None,
    query_transform: Optional[Callable[[str], str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Build a traced :class:`Pipeline` from existing callables.

    Either pass ``retrieve=`` and ``generate=`` directly, or pass a ``pipeline``
    object that exposes ``retrieve``/``search`` and ``generate``/``answer``
    methods and let ``wrap`` discover them.
    """
    if pipeline is not None:
        retrieve = retrieve or _discover(pipeline, ("retrieve", "search", "get_chunks"))
        generate = generate or _discover(pipeline, ("generate", "answer", "complete"))
    if retrieve is None or generate is None:
        raise ValueError(
            "wrap() needs `retrieve` and `generate` callables (pass them explicitly "
            "or provide a pipeline object exposing retrieve/search and generate/answer)"
        )
    return Pipeline(retrieve, generate, query_transform=query_transform, config=config)


def _discover(obj: Any, names: tuple) -> Optional[Callable[..., Any]]:
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn
    return None

"""RAGLens — a portable software-engineering layer for RAG pipelines.

Drop-in explainability and auto-documentation for *your* RAG pipeline — no
framework lock-in. See ``proposal.md`` for the full design.

Quick start::

    import raglens
    from raglens import trace

    @trace.retriever
    def retrieve(query): ...

    @trace.generator
    def generate(query, chunks): ...

    with raglens.capture(config={"embedding_model": "bge-small"}) as cap:
        chunks = retrieve("What is RAG?")
        answer = generate("What is RAG?", chunks)

    cap.attribute().save("traces.jsonl")
    raglens.write_datasheet("datasheet.md", [cap.trace])
"""

from __future__ import annotations

from . import trace  # noqa: F401  (the `@trace.retriever` / `@trace.generator` namespace)
from .attribution import AttributionError, attribute, lexical_similarity, top_chunks
from .capture import Capture, active_capture, capture, run
from .datasheet import render_html, render_markdown, write_datasheet
from .fingerprint import config_fingerprint
from .insights import (
    Insight,
    aggregate_metrics,
    compute_metrics,
    generate_fleet_insights,
    generate_insights,
)
from .instrument import auto_instrument, patch_anthropic, patch_openai, set_pricing
from .ir import SCHEMA_VERSION, Attribution, Chunk, RagTrace, Stage
from .wrap import Pipeline, wrap
from .writer import append_trace, iter_traces, read_traces, write_traces

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    # IR
    "RagTrace",
    "Stage",
    "Chunk",
    "Attribution",
    # frontends
    "trace",
    "wrap",
    "Pipeline",
    "capture",
    "run",
    "Capture",
    "active_capture",
    # backends
    "attribute",
    "lexical_similarity",
    "top_chunks",
    "AttributionError",
    "render_markdown",
    "render_html",
    "write_datasheet",
    "compute_metrics",
    "generate_insights",
    "aggregate_metrics",
    "generate_fleet_insights",
    "Insight",
    # config / instrumentation
    "config_fingerprint",
    "auto_instrument",
    "patch_openai",
    "patch_anthropic",
    "set_pricing",
    # io
    "write_traces",
    "append_trace",
    "read_traces",
    "iter_traces",
]

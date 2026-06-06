"""RagTrace — the pipeline-agnostic intermediate representation (IR).

This module defines the stable, serializable contract that every frontend
(decorators, ``wrap``, auto-instrumentation) populates and every backend
(attribution, datasheet, export) consumes. Keeping this schema stable and
versioned is the whole point of the project: frontends and backends evolve
independently around it.

Design rules for this file:
- Everything here must be JSON-serializable via ``to_dict`` / ``from_dict``.
- No third-party imports (NFR1.2 portability, NFR1.3 local-first).
- Bump ``SCHEMA_VERSION`` on any breaking change to the on-disk shape.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# On-disk schema version. v2 will add migration support keyed off this.
SCHEMA_VERSION = "0.1"


def _chunk_id(text: str, index: int) -> str:
    """Deterministic id for a chunk: stable across runs with identical text."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"c{index}_{digest}"


@dataclass
class Chunk:
    """A single retrieved unit of context."""

    id: str
    text: str
    score: Optional[float] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any, index: int) -> "Chunk":
        """Coerce whatever the user's retriever returned into a ``Chunk``.

        Accepts plain strings, dicts (``text``/``content``/``page_content`` keys,
        optional ``score``/``source``/``metadata``/``id``), or objects exposing a
        ``page_content`` attribute (LangChain-style ``Document``).
        """
        if isinstance(raw, Chunk):
            return raw
        if isinstance(raw, str):
            return cls(id=_chunk_id(raw, index), text=raw)
        if isinstance(raw, dict):
            text = raw.get("text") or raw.get("content") or raw.get("page_content") or ""
            return cls(
                id=str(raw.get("id") or _chunk_id(text, index)),
                text=text,
                score=raw.get("score"),
                source=raw.get("source"),
                metadata=raw.get("metadata", {}) or {},
            )
        # Object with a page_content / text attribute (duck-typed, e.g. a
        # LangChain Document). Promote a metadata "source" to the source field.
        text = getattr(raw, "page_content", None) or getattr(raw, "text", None) or str(raw)
        meta = dict(getattr(raw, "metadata", {}) or {})
        return cls(
            id=_chunk_id(text, index),
            text=text,
            score=getattr(raw, "score", None),
            source=meta.get("source"),
            metadata=meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        return cls(
            id=d["id"],
            text=d.get("text", ""),
            score=d.get("score"),
            source=d.get("source"),
            metadata=d.get("metadata", {}) or {},
        )


@dataclass
class Stage:
    """One stage of the pipeline.

    Rather than a class per stage type (which would complicate serialization in
    v1), a stage carries a ``kind`` discriminator and a free-form ``data`` payload
    whose shape is documented per kind below. This keeps the IR open to new stage
    kinds without a schema break.

    kinds and their ``data`` keys:
      - ``query_transform``: ``{original, transformed}``
      - ``retrieval``: ``{query, k, candidates: [Chunk...]}``
      - ``rerank``: ``{query, candidates: [Chunk...]}``
      - ``context_assembly``: ``{final_chunk_ids: [...], tokens, truncated}``
      - ``generation``: ``{model, answer, prompt_tokens, completion_tokens,
                           cost_usd, logprobs}``
    """

    kind: str
    data: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "latency_ms": self.latency_ms, "data": self.data}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Stage":
        return cls(kind=d["kind"], data=d.get("data", {}), latency_ms=d.get("latency_ms"))


@dataclass
class Attribution:
    """A counterfactual attribution score for one chunk.

    ``score`` is the measured importance of the chunk: how much the generated
    answer changed when this chunk was removed (1.0 = answer fully changed,
    0.0 = answer unchanged). It is *causal* — produced by actually re-running
    generation without the chunk — not self-reported by the model.
    """

    chunk_id: str
    score: float
    method: str = "counterfactual_ablation"
    ablated_similarity: Optional[float] = None
    ablated_answer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "method": self.method,
            "ablated_similarity": self.ablated_similarity,
            "ablated_answer": self.ablated_answer,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Attribution":
        return cls(
            chunk_id=d["chunk_id"],
            score=d["score"],
            method=d.get("method", "counterfactual_ablation"),
            ablated_similarity=d.get("ablated_similarity"),
            ablated_answer=d.get("ablated_answer"),
        )


@dataclass
class RagTrace:
    """The full record of one query through a RAG pipeline."""

    query_id: str
    query: str
    timestamp: str
    config_fingerprint: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    answer: Optional[str] = None
    stages: List[Stage] = field(default_factory=list)
    attributions: List[Attribution] = field(default_factory=list)
    # Free-form derived signals that require extra computation at capture time
    # (e.g. groundedness from a no-context counterfactual). Backends read this;
    # it is intentionally schema-open so new diagnostics don't break the IR.
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------------ #
    # Convenience accessors used by backends.
    # ------------------------------------------------------------------ #
    def stage(self, kind: str) -> Optional[Stage]:
        """Return the first stage of ``kind``, or None."""
        for s in self.stages:
            if s.kind == kind:
                return s
        return None

    def retrieved_chunks(self) -> List[Chunk]:
        """Chunks from the retrieval stage (raw candidates)."""
        s = self.stage("retrieval")
        if not s:
            return []
        return [Chunk.from_dict(c) for c in s.data.get("candidates", [])]

    def final_chunks(self) -> List[Chunk]:
        """Chunks that were actually passed to generation, in order.

        Falls back to retrieved chunks if no explicit context-assembly stage.
        """
        retrieved = {c.id: c for c in self.retrieved_chunks()}
        ca = self.stage("context_assembly")
        if ca and ca.data.get("final_chunk_ids"):
            ordered = []
            for cid in ca.data["final_chunk_ids"]:
                if cid in retrieved:
                    ordered.append(retrieved[cid])
            if ordered:
                return ordered
        return list(retrieved.values())

    def total_latency_ms(self) -> float:
        return sum(s.latency_ms or 0.0 for s in self.stages)

    def total_cost_usd(self) -> float:
        gen = self.stage("generation")
        return float(gen.data.get("cost_usd") or 0.0) if gen else 0.0

    # ------------------------------------------------------------------ #
    # Serialization.
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            "query": self.query,
            "timestamp": self.timestamp,
            "config_fingerprint": self.config_fingerprint,
            "config": self.config,
            "answer": self.answer,
            "stages": [s.to_dict() for s in self.stages],
            "attributions": [a.to_dict() for a in self.attributions],
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RagTrace":
        return cls(
            query_id=d["query_id"],
            query=d.get("query", ""),
            timestamp=d.get("timestamp", ""),
            config_fingerprint=d.get("config_fingerprint"),
            config=d.get("config", {}) or {},
            answer=d.get("answer"),
            stages=[Stage.from_dict(s) for s in d.get("stages", [])],
            attributions=[Attribution.from_dict(a) for a in d.get("attributions", [])],
            diagnostics=d.get("diagnostics", {}) or {},
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

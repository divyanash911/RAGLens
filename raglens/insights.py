"""Analytics backend: observability metrics + auto-generated insights.

Two layers, both derived *faithfully* from the IR (no invented numbers):

1. ``compute_metrics(trace)`` — structured observability: per-stage latency
   breakdown, token/context budget, chunk-length stats, retrieval-score stats,
   chunk redundancy/overlap, attribution distribution, and groundedness.

2. ``generate_insights(trace)`` — thresholded, actionable findings derived from
   those metrics (e.g. "answer appears ungrounded", "60% of retrieved chunks were
   inert", "chunks 2 and 3 are 80% redundant"). Each insight carries a severity
   and a concrete recommendation.

Everything here is read-only over a ``RagTrace``; the only signal that costs an
extra model call (groundedness) is computed once in the attribution backend and
stored in ``trace.diagnostics``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any, Dict, List, Optional

from .ir import RagTrace

# Severity ranking for ordering/printing.
SEVERITY_ORDER = {"critical": 0, "warn": 1, "info": 2, "good": 3}
SEVERITY_MARK = {"critical": "🔴", "warn": "🟠", "info": "🔵", "good": "🟢"}


@dataclass
class Insight:
    """One auto-generated finding about a trace (or a fleet of traces)."""

    severity: str  # critical | warn | info | good
    title: str
    detail: str
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"severity": self.severity, "title": self.title, "detail": self.detail, "recommendation": self.recommendation}

    def line(self) -> str:
        mark = SEVERITY_MARK.get(self.severity, "•")
        s = f"{mark} {self.title} — {self.detail}"
        if self.recommendation:
            s += f"  → {self.recommendation}"
        return s


# --------------------------------------------------------------------------- #
# Small text helpers (dependency-free).
# --------------------------------------------------------------------------- #
def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _token_set(text: str) -> set:
    return {w.lower().strip(".,;:!?\"'()[]") for w in (text or "").split() if w.strip()}


def _jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _pct(part: float, whole: float) -> float:
    return (100.0 * part / whole) if whole else 0.0


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #
def stage_latency_breakdown(trace: RagTrace) -> List[Dict[str, Any]]:
    """Per-stage latency with share of end-to-end (sorted slowest first)."""
    total = trace.total_latency_ms()
    rows = [
        {"stage": s.kind, "latency_ms": round(s.latency_ms, 2), "pct": round(_pct(s.latency_ms, total), 1)}
        for s in trace.stages
        if s.latency_ms
    ]
    return sorted(rows, key=lambda r: r["latency_ms"], reverse=True)


def chunk_length_stats(trace: RagTrace) -> Dict[str, Any]:
    chunks = trace.final_chunks()
    if not chunks:
        return {}
    chars = [len(c.text) for c in chunks]
    toks = [_approx_tokens(c.text) for c in chunks]
    return {
        "n_chunks": len(chunks),
        "chars_min": min(chars),
        "chars_mean": round(mean(chars), 1),
        "chars_max": max(chars),
        "context_tokens_est": sum(toks),
    }


def retrieval_score_stats(trace: RagTrace) -> Dict[str, Any]:
    scores = [c.score for c in trace.retrieved_chunks() if c.score is not None]
    if not scores:
        return {}
    return {
        "min": round(min(scores), 4),
        "mean": round(mean(scores), 4),
        "max": round(max(scores), 4),
        "spread": round(max(scores) - min(scores), 4),
    }


def redundancy(trace: RagTrace) -> Dict[str, Any]:
    """Pairwise lexical overlap among final chunks (chunk redundancy, FR2.1)."""
    chunks = trace.final_chunks()
    if len(chunks) < 2:
        return {}
    worst_pair = None
    worst = 0.0
    overlaps = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            jac = _jaccard(chunks[i].text, chunks[j].text)
            overlaps.append(jac)
            if jac > worst:
                worst, worst_pair = jac, (chunks[i].id, chunks[j].id)
    return {
        "max_overlap": round(worst, 3),
        "mean_overlap": round(mean(overlaps), 3),
        "worst_pair": worst_pair,
    }


def attribution_stats(trace: RagTrace, inert_threshold: float = 0.05) -> Dict[str, Any]:
    attrs = trace.attributions
    if not attrs:
        return {}
    scores = sorted((a.score for a in attrs), reverse=True)
    n_inert = sum(1 for s in scores if s < inert_threshold)
    return {
        "n": len(scores),
        "mean": round(mean(scores), 4),
        "max": round(scores[0], 4),
        "second": round(scores[1], 4) if len(scores) > 1 else None,
        "n_inert": n_inert,
        "n_useful": len(scores) - n_inert,
        "useful_fraction": round(_pct(len(scores) - n_inert, len(scores)) / 100.0, 3),
    }


def compute_metrics(trace: RagTrace) -> Dict[str, Any]:
    """Bundle all observability metrics for one trace."""
    return {
        "query_id": trace.query_id,
        "config_fingerprint": trace.config_fingerprint,
        "latency": {
            "total_ms": round(trace.total_latency_ms(), 2),
            "by_stage": stage_latency_breakdown(trace),
        },
        "cost_usd": trace.total_cost_usd(),
        "chunks": chunk_length_stats(trace),
        "retrieval_scores": retrieval_score_stats(trace),
        "redundancy": redundancy(trace),
        "attribution": attribution_stats(trace),
        "groundedness": trace.diagnostics.get("groundedness"),
        "answer_len_chars": len(trace.answer or ""),
    }


# --------------------------------------------------------------------------- #
# Insight generation (single trace).
# --------------------------------------------------------------------------- #
def generate_insights(trace: RagTrace) -> List[Insight]:
    """Derive thresholded, actionable insights for one trace."""
    out: List[Insight] = []
    m = compute_metrics(trace)

    # --- Groundedness (hallucination / parametric risk) ------------------- #
    g = m["groundedness"]
    if g is not None:
        if g < 0.15:
            out.append(Insight(
                "critical", "Answer appears ungrounded",
                f"removing ALL retrieved context changed the answer by only {g:.0%} — "
                "the model answered largely from parametric memory, not retrieval",
                "verify retrieval relevance; this is a hallucination / stale-answer risk",
            ))
        elif g < 0.45:
            out.append(Insight(
                "warn", "Weak grounding",
                f"the answer is only {g:.0%} dependent on retrieved context",
                "check whether the right documents are being retrieved",
            ))
        else:
            out.append(Insight(
                "good", "Answer is grounded in retrieval",
                f"removing all context changed the answer by {g:.0%}",
            ))

    # --- Retrieval efficiency / inert chunks ------------------------------ #
    a = m["attribution"]
    if a:
        if a["n"] >= 2 and a["n_inert"] / a["n"] >= 0.5:
            out.append(Insight(
                "warn", "Over-retrieval",
                f"{a['n_inert']} of {a['n']} retrieved chunks had ~no effect on the answer "
                f"(importance < 0.05)",
                f"consider lowering k or adding a reranker; only {a['n_useful']} chunk(s) mattered",
            ))
        # Single point of failure: one chunk decisive, rest inert.
        if a["n"] >= 2 and a["max"] >= 0.9 and (a["second"] or 0) < 0.1:
            out.append(Insight(
                "info", "Answer hinges on a single chunk",
                f"top chunk importance is {a['max']:.2f} while the next is {a['second']:.2f}",
                "retrieval is brittle here — if that one chunk were missed, the answer would change",
            ))

    # --- Redundancy ------------------------------------------------------- #
    r = m["redundancy"]
    if r and r.get("max_overlap", 0) >= 0.55:
        pair = r.get("worst_pair")
        pair_str = f" ({pair[0]} ↔ {pair[1]})" if pair else ""
        out.append(Insight(
            "warn", "Redundant retrieved chunks",
            f"two chunks share {r['max_overlap']:.0%} of their tokens{pair_str}",
            "deduplicate or reduce chunk_overlap to reclaim context budget",
        ))

    # --- Latency hotspot -------------------------------------------------- #
    by_stage = m["latency"]["by_stage"]
    if by_stage and m["latency"]["total_ms"] > 0:
        top = by_stage[0]
        if top["pct"] >= 60:
            out.append(Insight(
                "info", f"{top['stage']} dominates latency",
                f"{top['stage']} is {top['pct']:.0f}% of end-to-end latency "
                f"({top['latency_ms']:.0f} ms of {m['latency']['total_ms']:.0f} ms)",
                "optimize this stage first if latency matters",
            ))

    # --- Context budget --------------------------------------------------- #
    ch = m["chunks"]
    if ch and ch.get("context_tokens_est", 0) >= 3000:
        out.append(Insight(
            "info", "Large context window",
            f"~{ch['context_tokens_est']} context tokens across {ch['n_chunks']} chunks",
            "large prompts raise cost/latency; prune if many chunks are inert",
        ))

    return sorted(out, key=lambda i: SEVERITY_ORDER.get(i.severity, 9))


# --------------------------------------------------------------------------- #
# Fleet-level (multiple traces) — a taste of v2 observability.
# --------------------------------------------------------------------------- #
def aggregate_metrics(traces: List[RagTrace]) -> Dict[str, Any]:
    traces = list(traces)
    if not traces:
        return {}
    lats = [t.total_latency_ms() for t in traces]
    costs = [t.total_cost_usd() for t in traces]
    grounds = [t.diagnostics.get("groundedness") for t in traces if t.diagnostics.get("groundedness") is not None]
    fps = sorted({t.config_fingerprint for t in traces if t.config_fingerprint})

    def p95(xs: List[float]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]

    return {
        "n_traces": len(traces),
        "config_fingerprints": fps,
        "latency_ms": {"p50": round(median(lats), 1), "p95": round(p95(lats), 1), "max": round(max(lats), 1)},
        "cost_usd_total": round(sum(costs), 6),
        "groundedness_mean": round(mean(grounds), 3) if grounds else None,
    }


def generate_fleet_insights(traces: List[RagTrace]) -> List[Insight]:
    traces = list(traces)
    out: List[Insight] = []
    agg = aggregate_metrics(traces)
    if not agg:
        return out

    if len(agg["config_fingerprints"]) > 1:
        out.append(Insight(
            "info", "Mixed configurations",
            f"these {agg['n_traces']} traces span {len(agg['config_fingerprints'])} distinct "
            "config fingerprints",
            "split by fingerprint before comparing; this is what v2 `raglens diff` will automate",
        ))

    lat = agg["latency_ms"]
    if lat["p50"] > 0 and lat["p95"] >= 2 * lat["p50"]:
        out.append(Insight(
            "warn", "High latency variance",
            f"p95 ({lat['p95']:.0f} ms) is ≥2× p50 ({lat['p50']:.0f} ms)",
            "investigate tail latency (cold caches, slow retrievals, variable answer length)",
        ))

    gm = agg["groundedness_mean"]
    if gm is not None and gm < 0.45:
        out.append(Insight(
            "warn", "Fleet-wide weak grounding",
            f"mean groundedness across traces is {gm:.0%}",
            "retrieval may be under-serving the generator across many queries",
        ))

    return sorted(out, key=lambda i: SEVERITY_ORDER.get(i.severity, 9))

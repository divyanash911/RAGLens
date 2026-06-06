"""RAG datasheet (FR1.7) — auto-generated documentation from runtime behaviour.

A "datasheet" is a human-readable description of a RAG system derived *from its
own traces* rather than hand-written docs that rot. It now leads with
auto-generated **insights** (see :mod:`raglens.insights`) and a richer
**observability** section (per-stage latency, context budget, retrieval-score
stats, chunk redundancy, groundedness) — the explainability / observability /
documentation triad in one artifact.

Renders Markdown by default; ``render_html`` wraps the Markdown in minimal HTML.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable, List, Optional

from . import insights as _insights
from .ir import RagTrace


def _fmt_ms(ms: Optional[float]) -> str:
    return f"{ms:.0f} ms" if ms else "—"


def _fmt_usd(v: Optional[float]) -> str:
    return f"${v:.4f}" if v else "—"


def _fmt_num(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v:.0%}" if v is not None else "—"


def _safe_mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


def render_markdown(traces: Iterable[RagTrace], title: str = "RAG Datasheet") -> str:
    """Render a Markdown datasheet from one or more traces."""
    traces = list(traces)
    if not traces:
        return f"# {title}\n\n_No traces provided._\n"

    L: List[str] = [f"# {title}", "", f"_Generated from {len(traces)} trace(s) by RAGLens._", ""]

    _render_insights(L, traces)
    _render_provenance(L, traces)
    _render_pipeline_shape(L, traces)
    _render_observability(L, traces)
    _render_explainability(L, traces)
    _render_per_query(L, traces)

    return "\n".join(L).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Sections.
# --------------------------------------------------------------------------- #
def _render_insights(L: List[str], traces: List[RagTrace]) -> None:
    L.append("## 🔎 Generated Insights")
    L.append("")
    fleet = _insights.generate_fleet_insights(traces) if len(traces) > 1 else []
    per_trace = []
    for t in traces:
        per_trace.extend(_insights.generate_insights(t))
    all_insights = fleet + per_trace
    if not all_insights:
        L.append("_No notable findings. (Run attribution to unlock grounding & "
                 "retrieval-efficiency insights.)_")
        L.append("")
        return
    # De-duplicate identical titles+details (common across similar traces).
    seen = set()
    for ins in all_insights:
        key = (ins.severity, ins.title, ins.detail)
        if key in seen:
            continue
        seen.add(key)
        L.append(f"- {ins.line()}")
    L.append("")


def _render_provenance(L: List[str], traces: List[RagTrace]) -> None:
    fingerprints = sorted({t.config_fingerprint for t in traces if t.config_fingerprint})
    L.append("## Configuration & Provenance")
    L.append("")
    if fingerprints:
        L.append(f"- **Config fingerprint(s):** {', '.join(f'`{f}`' for f in fingerprints)}")
        if len(fingerprints) > 1:
            L.append("  - ⚠️ Multiple fingerprints — these traces span different configurations.")
    else:
        L.append("- **Config fingerprint(s):** _none recorded_")
    rep_cfg = next((t.config for t in traces if t.config), {})
    if rep_cfg:
        L.append("- **Representative config:**")
        for k in sorted(rep_cfg):
            L.append(f"  - `{k}`: {rep_cfg[k]}")
    L.append(f"- **IR schema version:** {', '.join(sorted({t.schema_version for t in traces}))}")
    L.append("")


def _model_of(t: RagTrace) -> Optional[str]:
    g = t.stage("generation")
    return (g.data.get("model") if g else None) or t.config.get("generation_model")


def _render_pipeline_shape(L: List[str], traces: List[RagTrace]) -> None:
    stage_kinds = Counter(s.kind for t in traces for s in t.stages)
    L.append("## Pipeline Shape")
    L.append("")
    if stage_kinds:
        order = ["query_transform", "retrieval", "rerank", "context_assembly", "generation"]
        present = [k for k in order if k in stage_kinds] + [k for k in stage_kinds if k not in order]
        L.append("Observed stages: " + " → ".join(f"`{k}`" for k in present))
    models = sorted({m for m in (_model_of(t) for t in traces) if m})
    L.append("")
    L.append(f"- **Generation model(s):** {', '.join(models) if models else '_unknown_'}")
    L.append("")


def _render_observability(L: List[str], traces: List[RagTrace]) -> None:
    L.append("## Observability")
    L.append("")

    # Fleet aggregate (latency percentiles, cost) when more than one trace.
    if len(traces) > 1:
        agg = _insights.aggregate_metrics(traces)
        lat = agg["latency_ms"]
        L.append("**Fleet summary**")
        L.append("")
        L.append("| Metric | Value |")
        L.append("|--------|-------|")
        L.append(f"| Traces | {agg['n_traces']} |")
        L.append(f"| Latency p50 / p95 / max | {lat['p50']:.0f} / {lat['p95']:.0f} / {lat['max']:.0f} ms |")
        L.append(f"| Total cost | {_fmt_usd(agg['cost_usd_total'])} |")
        if agg.get("groundedness_mean") is not None:
            L.append(f"| Mean groundedness | {_fmt_pct(agg['groundedness_mean'])} |")
        L.append("")

    # Per-stage latency breakdown (averaged across traces by stage kind).
    kind_lat = {}
    for t in traces:
        for row in _insights.stage_latency_breakdown(t):
            kind_lat.setdefault(row["stage"], []).append(row["latency_ms"])
    if kind_lat:
        total = sum(mean(v) for v in kind_lat.values())
        L.append("**Per-stage latency** (mean across traces)")
        L.append("")
        L.append("| Stage | Mean latency | Share |")
        L.append("|-------|-------------|-------|")
        for kind, vals in sorted(kind_lat.items(), key=lambda kv: mean(kv[1]), reverse=True):
            m = mean(vals)
            share = (100.0 * m / total) if total else 0.0
            L.append(f"| `{kind}` | {m:.0f} ms | {share:.0f}% |")
        L.append("")

    # Context & retrieval profile.
    ctx_tokens = [_insights.chunk_length_stats(t).get("context_tokens_est") for t in traces]
    n_chunks = [len(t.final_chunks()) for t in traces]
    ks = [(t.stage("retrieval").data.get("k") if t.stage("retrieval") else None) for t in traces]
    L.append("**Retrieval & context**")
    L.append("")
    L.append(f"- **Avg candidates retrieved (k):** {_fmt_num(_safe_mean([k for k in ks if k is not None]))}")
    L.append(f"- **Avg chunks reaching generation:** {_fmt_num(_safe_mean(n_chunks))}")
    L.append(f"- **Avg context size (est. tokens):** {_fmt_num(_safe_mean(ctx_tokens))}")
    # Retrieval score stats (only if the retriever exposed scores).
    score_means = [_insights.retrieval_score_stats(t).get("mean") for t in traces]
    if any(s is not None for s in score_means):
        L.append(f"- **Avg retrieval score:** {_fmt_num(_safe_mean(score_means))}")
    # Redundancy.
    red = [_insights.redundancy(t).get("max_overlap") for t in traces]
    if any(r is not None for r in red):
        L.append(f"- **Avg max chunk overlap (redundancy):** {_fmt_pct(_safe_mean(red))}")
    L.append("")

    # Cost / tokens table.
    latencies = [t.total_latency_ms() for t in traces]
    costs = [t.total_cost_usd() for t in traces]
    pt = [(t.stage("generation").data.get("prompt_tokens") if t.stage("generation") else None) for t in traces]
    ct = [(t.stage("generation").data.get("completion_tokens") if t.stage("generation") else None) for t in traces]
    L.append("**Cost & tokens**")
    L.append("")
    L.append("| Metric | Mean | Total |")
    L.append("|--------|------|-------|")
    L.append(f"| End-to-end latency | {_fmt_ms(_safe_mean(latencies))} | {_fmt_ms(sum(latencies))} |")
    L.append(f"| Cost (USD) | {_fmt_usd(_safe_mean(costs))} | {_fmt_usd(sum(costs))} |")
    L.append(f"| Prompt tokens | {_fmt_num(_safe_mean(pt))} | — |")
    L.append(f"| Completion tokens | {_fmt_num(_safe_mean(ct))} | — |")
    L.append("")


def _render_explainability(L: List[str], traces: List[RagTrace]) -> None:
    attributed = [t for t in traces if t.attributions]
    L.append("## Explainability (Counterfactual Attribution)")
    L.append("")
    if not attributed:
        L.append("_No attributions computed. Run `cap.attribute()` / `raglens explain`._")
        L.append("")
        return
    L.append(
        f"{len(attributed)} of {len(traces)} trace(s) have attribution. Scores are causal "
        "(answer change when a chunk is removed; 1.0 = answer fully changed)."
    )
    L.append("")
    all_scores = [a.score for t in attributed for a in t.attributions]
    if all_scores:
        inert = sum(1 for s in all_scores if s < 0.05)
        L.append(f"- **Mean chunk importance:** {_fmt_num(mean(all_scores))}")
        L.append(
            f"- **Inert chunks (importance < 0.05):** {inert} / {len(all_scores)} "
            f"({_pct(inert, len(all_scores)):.0f}%) — candidates for retrieval pruning."
        )
    grounds = [t.diagnostics.get("groundedness") for t in attributed if t.diagnostics.get("groundedness") is not None]
    if grounds:
        L.append(
            f"- **Mean groundedness:** {_fmt_pct(mean(grounds))} "
            "(answer change when *all* context is removed; low = parametric/hallucination risk)."
        )
    L.append("")


def _render_per_query(L: List[str], traces: List[RagTrace]) -> None:
    L.append("## Per-Query Detail")
    L.append("")
    for t in traces:
        L.append(f"### Query `{t.query_id[:8]}`")
        L.append("")
        L.append(f"> {t.query or '_(no query text)_'}")
        L.append("")
        if t.answer:
            ans = t.answer if len(t.answer) <= 500 else t.answer[:500] + "…"
            L.append(f"**Answer:** {ans}")
            L.append("")
        g = t.diagnostics.get("groundedness")
        if g is not None:
            awc = t.diagnostics.get("answer_without_context")
            awc_str = f" (without any context the model said: _{str(awc)[:80]}_)" if awc else ""
            L.append(f"**Groundedness:** {_fmt_pct(g)}{awc_str}")
            L.append("")
        if t.attributions:
            L.append("| Chunk | Importance | Source | Preview |")
            L.append("|-------|-----------|--------|---------|")
            by_id = {c.id: c for c in t.final_chunks()}
            for a in sorted(t.attributions, key=lambda x: x.score, reverse=True):
                c = by_id.get(a.chunk_id)
                preview = (c.text[:60].replace("\n", " ") + "…") if c and len(c.text) > 60 else (c.text if c else "")
                source = ((c.source or c.metadata.get("source")) if c else None) or "—"
                L.append(f"| `{a.chunk_id}` | {a.score:.3f} | {source} | {preview} |")
            L.append("")
        # Per-query insights.
        q_insights = _insights.generate_insights(t)
        if q_insights:
            L.append("_Insights:_")
            for ins in q_insights:
                L.append(f"- {ins.line()}")
            L.append("")
        L.append(f"_Latency: {_fmt_ms(t.total_latency_ms())} · Cost: {_fmt_usd(t.total_cost_usd())}_")
        L.append("")


def _pct(part: float, whole: float) -> float:
    return (100.0 * part / whole) if whole else 0.0


# --------------------------------------------------------------------------- #
# HTML + file output.
# --------------------------------------------------------------------------- #
def render_html(traces: Iterable[RagTrace], title: str = "RAG Datasheet") -> str:
    """Render the datasheet as a minimal standalone HTML document."""
    md = render_markdown(traces, title=title)
    escaped = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{font-family:ui-monospace,Menlo,Consolas,monospace;max-width:60rem;"
        "margin:2rem auto;padding:0 1rem;line-height:1.5}pre{white-space:pre-wrap}</style>"
        f"</head><body><pre>{escaped}</pre></body></html>\n"
    )


def write_datasheet(path: str, traces: Iterable[RagTrace], title: str = "RAG Datasheet") -> None:
    """Render to ``path``; format inferred from extension (.html → HTML)."""
    traces = list(traces)
    content = render_html(traces, title) if str(path).endswith((".html", ".htm")) else render_markdown(traces, title)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

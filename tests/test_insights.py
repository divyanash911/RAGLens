"""Tests for the analytics backend: metrics, groundedness, and generated insights."""

import raglens
from raglens.insights import (
    compute_metrics,
    generate_fleet_insights,
    generate_insights,
    redundancy,
)

DOCS = [
    {"text": "The capital of France is Paris.", "source": "geo"},
    {"text": "Bananas are yellow and grow in bunches.", "source": "noise"},
    {"text": "The capital of France is the city of Paris.", "source": "dup"},  # near-duplicate
]


def _grounded_gen(query, chunks):
    """Answer depends on retrieval: 'Paris' only if a Paris chunk is present."""
    for ch in chunks:
        if "Paris" in ch["text"]:
            return "Paris"
    return "I don't know"


def _ungrounded_gen(query, chunks):
    """Ignores chunks entirely → ungrounded answer."""
    return "The answer is forty-two regardless of context."


def _capture(gen):
    cap = raglens.run("capital of France?", lambda q: DOCS, gen, config={"embedding_model": "toy"})
    cap.attribute()  # groundedness=True by default
    return cap


def test_groundedness_recorded():
    cap = _capture(_grounded_gen)
    g = cap.trace.diagnostics.get("groundedness")
    assert g is not None and g > 0.5  # removing all context changes the answer a lot
    assert "answer_without_context" in cap.trace.diagnostics


def test_ungrounded_answer_flagged_critical():
    cap = _capture(_ungrounded_gen)
    assert cap.trace.diagnostics["groundedness"] < 0.15
    titles = [i.title for i in generate_insights(cap.trace)]
    assert any("ungrounded" in t.lower() for t in titles)
    sev = {i.title: i.severity for i in generate_insights(cap.trace)}
    assert sev[next(t for t in titles if "ungrounded" in t.lower())] == "critical"


def test_grounded_answer_marked_good():
    cap = _capture(_grounded_gen)
    insights = generate_insights(cap.trace)
    assert any(i.severity == "good" and "grounded" in i.title.lower() for i in insights)


def test_over_retrieval_insight():
    cap = _capture(_grounded_gen)
    # noise + duplicate chunks are inert → over-retrieval warning expected.
    titles = [i.title.lower() for i in generate_insights(cap.trace)]
    assert any("over-retrieval" in t for t in titles)


def test_redundancy_detected():
    cap = _capture(_grounded_gen)
    red = redundancy(cap.trace)
    assert red["max_overlap"] > 0.5  # the two Paris chunks overlap heavily
    titles = [i.title.lower() for i in generate_insights(cap.trace)]
    assert any("redundant" in t for t in titles)


def test_compute_metrics_shape():
    cap = _capture(_grounded_gen)
    m = compute_metrics(cap.trace)
    assert m["attribution"]["n"] == 3
    assert m["chunks"]["n_chunks"] == 3
    assert "by_stage" in m["latency"]
    assert m["groundedness"] is not None


def test_fleet_insights_mixed_configs():
    cap_a = raglens.run("q1", lambda q: DOCS, _grounded_gen, config={"embedding_model": "a"})
    cap_b = raglens.run("q2", lambda q: DOCS, _grounded_gen, config={"embedding_model": "b"})
    cap_a.attribute(); cap_b.attribute()
    fleet = generate_fleet_insights([cap_a.trace, cap_b.trace])
    assert any("config" in i.title.lower() for i in fleet)


def test_groundedness_can_be_disabled():
    cap = raglens.run("q", lambda q: DOCS, _grounded_gen, config={})
    cap.attribute(groundedness=False)
    assert "groundedness" not in cap.trace.diagnostics


def test_datasheet_includes_insights_and_observability():
    cap = _capture(_grounded_gen)
    md = raglens.render_markdown([cap.trace])
    assert "Generated Insights" in md
    assert "Observability" in md
    assert "Groundedness" in md or "groundedness" in md
    assert "Per-stage latency" in md

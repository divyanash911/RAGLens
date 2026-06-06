"""Tests for JSONL persistence round-trip and datasheet rendering."""

import raglens
from raglens.datasheet import render_html, render_markdown
from raglens.writer import iter_traces, read_traces


def _docs(query):
    return [{"text": "The capital of France is Paris.", "source": "geo"},
            {"text": "Bananas are yellow.", "source": "noise"}]


def _gen(query, chunks):
    return "Paris" if any("Paris" in (c["text"]) for c in chunks) else "unknown"


def _make_capture():
    cap = raglens.run("capital of France?", _docs, _gen, config={"embedding_model": "toy"})
    cap.attribute()
    return cap


def test_jsonl_roundtrip(tmp_path):
    cap = _make_capture()
    path = tmp_path / "traces.jsonl"
    cap.save(str(path), append=False)
    back = read_traces(str(path))
    assert len(back) == 1
    assert back[0].answer == "Paris"
    assert back[0].config_fingerprint == cap.trace.config_fingerprint
    assert len(back[0].attributions) == 2


def test_append_and_iter(tmp_path):
    path = tmp_path / "t.jsonl"
    _make_capture().save(str(path), append=True)
    _make_capture().save(str(path), append=True)
    assert len(list(iter_traces(str(path)))) == 2


def test_datasheet_markdown_contains_sections(tmp_path):
    cap = _make_capture()
    md = render_markdown([cap.trace], title="My Datasheet")
    assert "# My Datasheet" in md
    assert "Configuration & Provenance" in md
    assert "Counterfactual Attribution" in md
    assert cap.trace.config_fingerprint in md


def test_datasheet_html(tmp_path):
    cap = _make_capture()
    html = render_html([cap.trace])
    assert html.startswith("<!doctype html>")


def test_write_datasheet_infers_format(tmp_path):
    cap = _make_capture()
    md_path = tmp_path / "d.md"
    html_path = tmp_path / "d.html"
    raglens.write_datasheet(str(md_path), [cap.trace])
    raglens.write_datasheet(str(html_path), [cap.trace])
    assert md_path.read_text().startswith("# RAG Datasheet")
    assert html_path.read_text().startswith("<!doctype html>")

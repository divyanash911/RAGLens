"""End-to-end RAGLens example (FR1.9) — runnable with zero external services.

This is a *hand-rolled* RAG pipeline: no LangChain, no LlamaIndex, no vector DB,
no LLM API key. It uses a tiny in-memory corpus, a keyword retriever, and an
extractive "generator" so the whole thing runs offline and deterministically —
which lets the counterfactual attribution produce stable, inspectable scores.

Run it:

    python examples/simple_pipeline.py
    # or
    raglens demo

It prints attribution, writes traces.jsonl, and renders datasheet.md.
"""

from __future__ import annotations

import os

import raglens
from raglens import trace

# --------------------------------------------------------------------------- #
# A tiny in-memory corpus. Each doc is a (source, text) pair.
# --------------------------------------------------------------------------- #
CORPUS = [
    ("rag_intro.md", "Retrieval-Augmented Generation (RAG) combines a retriever with a generator "
                     "so a language model can answer using external documents."),
    ("retriever.md", "The retriever fetches the most relevant chunks for a query, often using "
                     "vector similarity over embeddings."),
    ("generator.md", "The generator is a large language model that conditions its answer on the "
                     "retrieved chunks passed as context."),
    ("eval.md", "Evaluating RAG involves measuring retrieval relevance, groundedness, and answer "
                "quality, sometimes with an LLM as a judge."),
    ("weather.md", "The weather today is sunny with a light breeze and a high of 24 degrees."),
]

PROMPT_TEMPLATE = "Answer the question using the context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

# Config that defines this pipeline's identity → drives the config_fingerprint.
CONFIG = {
    "corpus_version": "demo-v1",
    "embedding_model": "keyword-overlap",
    "chunking": {"strategy": "whole-doc", "max_tokens": 256},
    "retrieval": {"k": 3, "metric": "keyword_overlap"},
    "generation_model": "extractive-stub",
    "prompt_template": PROMPT_TEMPLATE,
}


@trace.retriever
def retrieve(query: str):
    """Keyword-overlap retriever: score docs by shared words, return top-k."""
    q_words = {w.lower().strip(".,?") for w in query.split()}
    scored = []
    for source, text in CORPUS:
        words = {w.lower().strip(".,?") for w in text.split()}
        overlap = len(q_words & words)
        scored.append({"text": text, "source": source, "score": float(overlap)})
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[: CONFIG["retrieval"]["k"]]


@trace.generator
def generate(query: str, chunks):
    """Extractive 'generator': returns the sentences from the context that share
    the most words with the query. Deterministic and offline, but it genuinely
    *depends on which chunks are present* — so ablating an influential chunk
    changes the answer, which is exactly what attribution measures."""
    q_words = {w.lower().strip(".,?") for w in query.split()}
    best = []
    for ch in chunks:
        text = ch["text"] if isinstance(ch, dict) else str(ch)
        score = len(q_words & {w.lower().strip(".,?") for w in text.split()})
        if score > 0:
            best.append((score, text))
    best.sort(key=lambda t: t[0], reverse=True)
    if not best:
        return "I don't have enough context to answer."
    # Use the single most relevant chunk's text as the answer.
    return best[0][1]


def main() -> None:
    question = "What does the retriever do in RAG?"

    # --- Capture (decorator frontend) ------------------------------------- #
    with raglens.capture(config=CONFIG) as cap:
        chunks = retrieve(question)
        answer = generate(question, chunks)

    print(f"Q: {question}")
    print(f"A: {answer}\n")

    # --- Explainability backend: counterfactual attribution --------------- #
    cap.attribute()
    print("Counterfactual chunk attribution (importance = answer change when removed):")
    for rc in raglens.top_chunks(cap, n=len(cap.trace.final_chunks())):
        src = rc.chunk.source if rc.chunk else "?"
        print(f"  {rc.score:5.3f}  [{src}]  {rc.chunk.text[:60] if rc.chunk else ''}…")

    # --- Persist trace + documentation backend ---------------------------- #
    out_dir = os.path.dirname(os.path.abspath(__file__))
    traces_path = os.path.join(out_dir, "traces.jsonl")
    datasheet_path = os.path.join(out_dir, "datasheet.md")

    cap.save(traces_path, append=False)
    raglens.write_datasheet(datasheet_path, [cap.trace], title="Demo RAG Datasheet")

    print(f"\nconfig_fingerprint: {cap.trace.config_fingerprint}")
    print(f"Wrote trace      → {traces_path}")
    print(f"Wrote datasheet  → {datasheet_path}")


if __name__ == "__main__":
    main()

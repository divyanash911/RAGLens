# Case study: RAGLens on a real LangChain RAG pipeline

**Question this answers:** is RAGLens actually usable on a real, third-party RAG
pipeline — not the one we designed it around?

We took the **canonical LangChain RAG pipeline** from the official tutorial
([docs.langchain.com/oss/python/langchain/rag](https://docs.langchain.com/oss/python/langchain/rag))
and ran it locally with real components — no API keys, no mocks:

| Component | What we used |
|-----------|--------------|
| Splitter | `RecursiveCharacterTextSplitter(chunk_size=260, chunk_overlap=40)` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| Retriever | `vector_store.as_retriever(k=3)` |
| LLM | `google/flan-t5-base` (local, via 🤗 transformers) |
| Composition | LCEL: `{"context": retriever \| format_docs, "question": …} \| prompt \| llm \| parser` |

Run it yourself:

```bash
PYTHONPATH=.. python real_world_demo.py      # from this folder, or:
python examples/langchain_rag/real_world_demo.py   # with raglens installed
```

(First run downloads MiniLM ~90 MB and flan-t5-base ~1 GB.)

---

## What happened

**RAG demonstrably fixed a wrong answer**, and RAGLens explained *why*:

```
Q: What was the name of the command module flown during the Apollo 11 mission?
Bare flan-t5 (no context):  'Apollo 11'   ← wrong (confuses mission with module)
Full RAG pipeline:          'Columbia'    ← correct, driven by retrieval

Counterfactual chunk attribution (importance = answer change when chunk removed):
  1.000  [apollo11.txt]  …Aldrin joined him 19 minutes later. Michael Collins flew the command module Columbia…
  0.000  [apollo11.txt]  Apollo 11 was the American spaceflight that first landed humans…
  0.000  [saturn_v.txt]  …meters tall. It launched every Apollo lunar mission…
```

The attribution is **causal**: RAGLens re-ran the real flan-t5 with each retrieved
chunk removed. Only removing the chunk that literally contains *"the command module
Columbia"* changed the answer — so it scores 1.000 and the rest score 0.000. The
auto-generated [datasheet.md](datasheet.md) captures the config fingerprint, the
pipeline shape, latency, and a "67% of retrieved chunks were inert" pruning hint.

---

## Usability verdict: yes, with one caveat

**What worked out of the box**
- `Chunk.from_raw` already understood LangChain `Document` objects
  (`page_content` + `metadata`) — retrieved docs flowed into the IR unmodified.
- `raglens.wrap` attached to the pipeline's **own** retriever and a generate step
  built from the **same** prompt + llm. **Zero changes to pipeline logic.**
- Counterfactual attribution worked against a real local LLM and real embeddings.
- The datasheet and `config_fingerprint` were produced with no extra effort.

**The caveat (an honest limitation, already flagged in the proposal)**
- The **`@trace.retriever` / `@trace.generator` decorators do not fit idiomatic
  LCEL.** A declarative chain has no `retrieve(query)` / `generate(query, chunks)`
  functions to annotate — retrieval and generation are fused inside one
  `rag_chain.invoke()`. `wrap` bridges this cleanly, but it means the lowest-friction
  frontend (decorators) is really aimed at *hand-rolled* pipelines, while
  framework pipelines need `wrap` (or the planned v2 LangChain auto-instrumentation,
  proposal **FR2.5**).

**Two real polish bugs this exercise surfaced in RAGLens — and we fixed them here:**
1. The datasheet showed the generation model as *"unknown"* under `wrap` (no SDK to
   auto-capture it). Fixed: it now falls back to `config["generation_model"]`.
2. A `Document`'s `metadata["source"]` wasn't promoted to the chunk's `source`
   field, so the datasheet's Source column was blank. Fixed in `Chunk.from_raw`.

## Bottom line

RAGLens ran on an unmodified, real LangChain pipeline and produced faithful,
causal, per-chunk explanations plus auto-documentation — with no changes to the
pipeline's retrieval, embeddings, or LLM. The integration cost was ~10 lines of
`wrap` glue. The main usability gap is that framework-style (LCEL) pipelines can't
use the decorator frontend; that's exactly the gap v2's auto-instrumentation
targets.

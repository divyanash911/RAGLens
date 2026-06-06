# RAGLens

**Drop-in explainability and auto-documentation for *your* RAG pipeline — no framework lock-in.**

RAGLens adds software-engineering principles — **explainability, observability, and
documentation** — to *any* Retrieval-Augmented Generation pipeline, including
hand-rolled ones, without requiring you to adopt LangChain, LlamaIndex, or a hosted
service. It treats your RAG pipeline as a software artifact deserving SE rigor, not
an ML model to be scored.

This repository implements **v1 — "Capture, Attribute, Document"** of
[`proposal.md`](proposal.md). v1 is intentionally **stdlib-only** (no required
third-party dependencies) and **local-first**.

> Looking for the full roadmap (v2 observability dashboard, drift/diff, exports;
> v3 OTel ingestion, runtime mode, published IR spec)? See [`proposal.md`](proposal.md).

---

## Why RAGLens (what's different)

Existing tools either score answer quality (RAGAS, ARES, DeepEval), attribute a
single answer (ContextCite), or provide tracing infrastructure (Phoenix, Langfuse).
RAGLens **unifies explainability + observability + documentation** on top of one
**pipeline-agnostic intermediate representation (`RagTrace`)**, and works on
pipelines that use no framework at all.

- **Faithful attribution, not self-reported.** Counterfactual ablation actually
  re-runs your generator with each chunk removed and measures the answer change —
  causal, not a hallucinated citation or an attention proxy.
- **Portable.** Two decorators (or one `wrap()` call) on functions you already have.
- **Auto-generated docs.** A "RAG datasheet" derived from real runtime behaviour.

---

## Install

```bash
pip install -e .            # core (stdlib only)
pip install -e ".[dev]"     # + pytest
pip install -e ".[openai]"  # optional SDK auto-instrumentation
```

Requires Python ≥ 3.9.

## Quick start (under 10 lines)

```python
import raglens
from raglens import trace

@trace.retriever
def retrieve(query):            # your existing retriever
    return my_vector_search(query)

@trace.generator
def generate(query, chunks):    # your existing generator
    return my_llm_answer(query, chunks)

with raglens.capture(config={"embedding_model": "bge-small", "retrieval": {"k": 5}}) as cap:
    chunks = retrieve("What is RAG?")
    answer = generate("What is RAG?", chunks)

cap.attribute()                              # counterfactual chunk attribution
cap.save("traces.jsonl")                     # versioned JSONL
raglens.write_datasheet("datasheet.md", [cap.trace])
```

### Works with how *you* write RAG code

The decorators bind arguments **by signature**, not by position, so they handle
the shapes real pipelines come in:

```python
class Pipe:
    @trace.retriever                              # `self` is skipped automatically
    def retrieve(self, query): ...

    @trace.generator(query="question", chunks="context")   # custom arg names
    def generate(self, question, context, temperature=0.0): ...

@trace.retriever                                  # async is supported
async def retrieve(query): ...
```

Counterfactual attribution re-invokes your generator with the chunk argument
swapped out while **preserving every other argument** (prompt, temperature,
`self`, …), so it works regardless of the generator's signature — not just
`(query, chunks)`.

If your pipeline doesn't decompose into a retriever and a generator at all (a
single `answer(query)` function, a declarative chain, etc.), use the manual
escape hatch:

```python
with raglens.capture(config=cfg) as cap:
    answer = my_monolithic_rag(question)
    trace.record_retrieval(question, chunks)
    trace.record_generation(question, chunks, answer,
                            regenerate=lambda new: my_generate(question, new))
```

### Prefer not to annotate your source? Use `wrap`

```python
pipe = raglens.wrap(retrieve=my_search, generate=my_answer, config=cfg)
cap = pipe.run("What is RAG?")
cap.attribute().save("traces.jsonl")
```

### Capture LLM token/cost automatically (optional)

```python
raglens.auto_instrument()   # patches openai / anthropic if installed
```

## Try the bundled example (no API key needed)

```bash
python examples/simple_pipeline.py
# or
raglens demo
```

It runs a fully offline, deterministic RAG pipeline, prints counterfactual
attribution, and writes `traces.jsonl` + `datasheet.md`.

## CLI

```bash
raglens run <script.py>          # run a pipeline script with raglens importable
raglens explain traces.jsonl     # show stored counterfactual attribution
raglens doc traces.jsonl -o datasheet.md   # render a RAG datasheet (.md or .html)
raglens demo                     # run the bundled example
raglens version
```

> **Note:** counterfactual attribution is *causal* — it re-runs your generator
> with chunks removed — so it is computed in-process (`cap.attribute()`) while
> your pipeline is live. `raglens explain` displays attributions already stored in
> a trace; it cannot re-generate from a static file alone.

## How it works (architecture)

A single internal representation populated by multiple *frontends* and consumed by
multiple *backends* (the "LLVM model"):

```
Frontends (capture)              IR              Backends (produce)
---------------------         --------        ----------------------
@trace decorators (Tier 1)                     Attribution (explainability)
raglens.wrap (Tier 1)         RagTrace         RAG datasheet (documentation)
SDK auto-instrument (Tier 2)  (JSONL)          JSONL persistence
```

| Module | Responsibility |
|--------|----------------|
| [raglens/ir.py](raglens/ir.py) | `RagTrace` IR — the stable, versioned contract |
| [raglens/trace.py](raglens/trace.py) | `@trace.retriever` / `@trace.generator` decorators |
| [raglens/wrap.py](raglens/wrap.py) | `wrap()` adapter for unannotated callables |
| [raglens/capture.py](raglens/capture.py) | active-capture context + trace building |
| [raglens/attribution.py](raglens/attribution.py) | counterfactual chunk attribution |
| [raglens/datasheet.py](raglens/datasheet.py) | RAG datasheet (Markdown/HTML) |
| [raglens/instrument.py](raglens/instrument.py) | OpenAI/Anthropic SDK auto-instrumentation |
| [raglens/fingerprint.py](raglens/fingerprint.py) | `config_fingerprint` |
| [raglens/cli.py](raglens/cli.py) | `raglens` CLI |

## Tests

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).

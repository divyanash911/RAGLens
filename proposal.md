# Proposal: RAGLens — A Portable Software-Engineering Layer for RAG Pipelines

## 1. Summary

RAGLens is an open-source tool that adds software-engineering principles —
**explainability, observability, and documentation** — to *any* Retrieval-Augmented
Generation (RAG) pipeline, including hand-rolled and polyglot ones, without
requiring the user to adopt a particular framework.

Unlike existing tools that either (a) score answer quality (RAGAS, ARES, DeepEval),
(b) provide attribution for a single answer (ContextCite, self-citation), or
(c) provide tracing infrastructure (Phoenix, Langfuse, LangSmith), RAGLens unifies
explainability + observability + documentation on top of a single **pipeline-agnostic
intermediate representation (IR)**, and treats the RAG pipeline as a software artifact
deserving SE rigor rather than an ML model to be scored.

### Positioning (one line)
> Drop-in explainability and auto-documentation for *your* RAG pipeline — no framework lock-in.

### Explicit non-goals
- Not another LLM-as-judge eval metric (we interoperate with RAGAS, not replace it).
- Not a tracing-infrastructure competitor to Langfuse/Phoenix (we can export to them).
- Not a hosted SaaS; RAGLens is a local-first OSS library + CLI.

---

## 2. Problem & Motivation

Existing RAG explainability/eval frameworks share recurring limitations:

1. **LLM-as-judge dependence** — non-deterministic, costly, biased, hard to reproduce.
2. **Unreliable or expensive attribution** — self-citation hallucinates; faithful
   ablation-based attribution (ContextCite) is O(chunks) forward passes per query.
3. **Framework lock-in** — instrumentation assumes LangChain/LlamaIndex; custom
   pipelines are unsupported.
4. **They explain the answer, not the pipeline** — no per-stage latency/cost,
   retrieval-distribution drift, config provenance, or failure-mode analysis.
5. **Offline-test-set bias** — eval is divorced from production traces.
6. **No documentation/lineage output** — no auto-generated "datasheet" of the system.
7. **No shared abstraction** — OpenInference/OTel GenAI conventions are tracing-oriented
   and only partially adopted.

RAGLens targets gaps 3–7 directly, and reduces 1–2 via faithful-but-cheap counterfactual
attribution.

---

## 3. Architecture Principle

A single internal representation populated by multiple "frontends" and consumed by
multiple "backends" (the LLVM model):

```
Frontends (how we capture)            IR              Backends (what we produce)
------------------------------    -----------    ------------------------------
Tier 0  black-box probing                          Explainability (attribution)
Tier 1  minimal adapter / decorators  RagTrace     Observability (metrics/traces)
Tier 2  auto-instrumentation (SDKs)   (versioned   Documentation (RAG datasheet)
Tier 3  OTel / OpenInference spans     JSONL)       Export (Langfuse/Phoenix/OTel)
```

The IR is the stable contract. Frontends and backends evolve independently.

### IR sketch (`RagTrace`)
- `query_id`, `timestamp`, `config_fingerprint` (hash of corpus version + embedding
  model + chunking params + prompt template — enables reproducibility & drift detection)
- `stages[]`: QueryTransform → Retrieval(candidates, scores, k) → Rerank? →
  ContextAssembly(final_chunks, tokens, truncated) → Generation(tokens, logprobs?,
  answer, model, latency, cost)
- `attributions[]?`: claim_span → supporting_chunk_ids → score

---

## 4. Versioned Requirements

The product is delivered in three versions. Each version is independently useful and
ships a coherent slice of value. Later versions build on the IR established in v1.

---

### Version 1 — "Capture, Attribute, Document" (MVP)

**Goal:** A developer with their own Python RAG pipeline can, in under ~10 lines and
~5 minutes, produce a trace, a faithful attribution, and an auto-generated datasheet.

#### Functional Requirements (v1)
- **FR1.1** Provide `@trace.retriever` and `@trace.generator` decorators that capture
  inputs/outputs of the user's existing retrieve and generate functions (Tier 1).
- **FR1.2** Provide a `raglens.wrap(...)` adapter for users who prefer not to annotate
  source, by pointing at existing callables.
- **FR1.3** Define and serialize the `RagTrace` IR to versioned JSONL.
- **FR1.4** Compute and persist a `config_fingerprint` per trace.
- **FR1.5** Auto-instrument at least the raw OpenAI and Anthropic SDK calls underneath
  user functions to capture token counts, latency, and cost (Tier 2, partial).
- **FR1.6** **Explainability backend:** counterfactual chunk attribution — ablate/drop
  each retrieved chunk, re-generate, measure answer delta; emit per-chunk attribution
  scores. Faithful (causal), not self-reported.
- **FR1.7** **Documentation backend:** render IR + config into a Markdown/HTML "RAG
  datasheet" (stages, models, chunking, retrieval distribution, cost/latency profile).
- **FR1.8** CLI: `raglens run <script/dataset>`, `raglens explain <trace>`,
  `raglens doc <traces>`.
- **FR1.9** Ship a runnable end-to-end example pipeline producing a trace + datasheet.

#### Non-Functional Requirements (v1)
- **NFR1.1 Ease of use:** time-to-first-value ≤ 5 min; ≤ 10 lines of integration.
- **NFR1.2 Portability:** works on a hand-rolled pipeline with no LangChain/LlamaIndex
  dependency required.
- **NFR1.3 Local-first:** no network calls except to the user's own LLM/vector backends.
- **NFR1.4 Low overhead:** trace capture adds ≤ ~5% latency in passthrough (non-ablation) mode.
- **NFR1.5 Reproducibility:** identical config + inputs yield identical traces (modulo
  LLM nondeterminism, which is recorded).
- **NFR1.6 Documentation:** README quickstart, API docs, one worked example.
- **NFR1.7 Licensing:** permissive OSS license (Apache-2.0 or MIT).
- **NFR1.8 Test coverage:** core IR + decorators + attribution covered by unit tests.

---

### Version 2 — "Observe & Compare"

**Goal:** Move from single-trace inspection to fleet/longitudinal observability and
regression comparison across pipeline configurations.

#### Functional Requirements (v2)
- **FR2.1** Aggregate many traces into datasets; compute per-stage metrics
  (latency, cost, token usage, retrieval-distribution stats, chunk redundancy/overlap).
- **FR2.2** Local observability dashboard (e.g. served via CLI) over collected traces.
- **FR2.3** **Drift detection:** compare retrieval distributions, cost, and attribution
  patterns across `config_fingerprint`s and over time.
- **FR2.4** `raglens diff <runA> <runB>`: regression report between two pipeline versions.
- **FR2.5** Expand auto-instrumentation to common vector DBs (Chroma, Pinecone, pgvector)
  and LangChain/LlamaIndex (Tier 2 broadened).
- **FR2.6** Export traces to Langfuse / Phoenix / OTel (interoperability, not competition).
- **FR2.7** Optional integration hook to run external eval metrics (e.g. RAGAS) and attach
  their scores to the IR.
- **FR2.8** Attribution efficiency: batched/grouped chunk ablation to reduce forward passes.

#### Non-Functional Requirements (v2)
- **NFR2.1 Scalability:** handle ≥ 100k traces without unacceptable degradation; streaming/
  chunked processing rather than full in-memory loads.
- **NFR2.2 Performance:** grouped attribution measurably cheaper than naive per-chunk ablation.
- **NFR2.3 Extensibility:** documented plugin interface for new frontends/backends.
- **NFR2.4 Stability:** IR schema versioned with migration support; backward compatibility
  for v1 traces.
- **NFR2.5 Interoperability:** exports conform to OpenInference/OTel GenAI conventions.
- **NFR2.6 Observability overhead:** dashboard usable on a developer laptop (no cluster).

---

### Version 3 — "Standardize & Productionize"

**Goal:** Establish RAGLens's IR as a shared schema, support production/runtime use, and
broaden language reach.

#### Functional Requirements (v3)
- **FR3.1** Tier 3 frontend: ingest OTel/OpenInference spans emitted by production
  pipelines (language-agnostic capture).
- **FR3.2** Runtime/online mode: sampled trace capture and attribution in production,
  not just offline batch.
- **FR3.3** Publish the `RagTrace` IR as a documented, versioned open specification.
- **FR3.4** Failure-mode analysis: automatic detection/classification of common RAG
  failure patterns (empty/low-score retrieval, context truncation, unsupported claims).
- **FR3.5** Data lineage: trace corpus → embedding model → index → answer provenance.
- **FR3.6** Language reach: at minimum a thin client/SDK for one additional language
  (e.g. JS/TS) feeding the same IR.
- **FR3.7** CI integration: `raglens` as a CI gate (fail build on attribution/cost/drift
  regressions beyond thresholds).

#### Non-Functional Requirements (v3)
- **NFR3.1 Production safety:** runtime capture has bounded, configurable overhead and
  sampling; degrades gracefully on backend failure.
- **NFR3.2 Privacy:** redaction/opt-out controls for sensitive query/context content.
- **NFR3.3 Standard adoption:** IR spec stable, documented, and externally consumable.
- **NFR3.4 Reliability:** runtime path has no single point of failure that can break the
  host pipeline (observation must never break production).
- **NFR3.5 Polyglot consistency:** traces from different language clients are schema-identical.
- **NFR3.6 Maintainability:** governance, contribution guide, and release process for OSS
  community.

---

## 5. Roadmap at a Glance

| Capability                    | v1 | v2 | v3 |
|-------------------------------|----|----|----|
| Trace capture (decorators)    | ✅ | ✅ | ✅ |
| Counterfactual attribution    | ✅ | ✅ (batched) | ✅ (runtime) |
| Auto-generated datasheet      | ✅ | ✅ | ✅ |
| Auto-instrumentation (SDKs)   | partial | broad | broad |
| Observability dashboard       | —  | ✅ | ✅ |
| Drift / diff / regression     | —  | ✅ | ✅ (CI gate) |
| Export (Langfuse/Phoenix/OTel)| —  | ✅ | ✅ |
| OTel span ingestion           | —  | —  | ✅ |
| Runtime/online mode           | —  | —  | ✅ |
| Published IR spec             | —  | —  | ✅ |
| Multi-language clients        | —  | —  | ✅ |

---

## 6. Success Criteria

- **v1:** A new user instruments their own pipeline and gets a datasheet + faithful
  attribution in ≤ 5 minutes; runnable example in the repo.
- **v2:** Detect a retrieval/cost regression between two pipeline configs automatically.
- **v3:** A third party adopts the IR spec or contributes a frontend/backend; runtime
  capture runs in a production pipeline without measurable user-facing impact.

---

## 7. Open Questions

- Default LLM backend for attribution re-generation (user-provided vs. pluggable).
- Whether to bundle a minimal storage layer (SQLite/DuckDB) in v2 or stay file-based.
- Scope of failure-mode taxonomy in v3 (heuristic vs. learned).

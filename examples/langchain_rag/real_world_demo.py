"""Running RAGLens on a *real* LangChain RAG pipeline.

This is not a toy. It builds the canonical LangChain RAG pipeline from the
official tutorial (https://docs.langchain.com/oss/python/langchain/rag):

    documents → RecursiveCharacterTextSplitter → embeddings → FAISS vector store
    → retriever → prompt → LLM → StrOutputParser   (composed with LCEL)

using real components that run fully locally:
    - embeddings:  sentence-transformers `all-MiniLM-L6-v2`
    - vector store: FAISS
    - LLM:         google/flan-t5-base via transformers (no API key)

Then it shows what it takes to put RAGLens on top of it, and what RAGLens reveals.

KEY USABILITY FINDING (see also the README in this folder):
Idiomatic LangChain composes retrieval and generation *declaratively* with LCEL
(`{"context": retriever | format_docs, ...} | prompt | llm | parser`). There is
no `retrieve(query)` / `generate(query, chunks)` function to put a decorator on —
so the `@trace.retriever` / `@trace.generator` frontend does NOT fit this style.
The `raglens.wrap` frontend does: we point it at the pipeline's *own* retriever
and a generate step built from the *same* prompt+llm, changing none of the
pipeline's logic. RAGLens's `Chunk.from_raw` already understands LangChain
`Document` objects, so retrieved docs flow into the IR unmodified.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline as hf_pipeline

import raglens

# --------------------------------------------------------------------------- #
# A small, real corpus (verifiable facts). Two of the five docs are distractors
# so retrieval has to discriminate and attribution has something to find.
# --------------------------------------------------------------------------- #
RAW_DOCS = [
    Document(
        page_content=(
            "Apollo 11 was the American spaceflight that first landed humans on the Moon. "
            "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo "
            "Lunar Module Eagle on July 20, 1969. Neil Armstrong became the first person to "
            "step onto the lunar surface; Aldrin joined him 19 minutes later. Michael Collins "
            "flew the command module Columbia alone in lunar orbit while they were on the surface."
        ),
        metadata={"source": "apollo11.txt"},
    ),
    Document(
        page_content=(
            "The Saturn V was an American super heavy-lift launch vehicle developed by NASA "
            "to support the Apollo program. It remains the tallest, heaviest, and most powerful "
            "rocket ever brought to operational status, standing 110.6 meters tall. It launched "
            "every Apollo lunar mission, including Apollo 11."
        ),
        metadata={"source": "saturn_v.txt"},
    ),
    Document(
        page_content=(
            "The Moon is Earth's only natural satellite. Its average distance from Earth is "
            "about 384,400 kilometers. The Moon is in synchronous rotation with Earth, so the "
            "same side always faces us. Its surface is covered in craters and basaltic plains "
            "called maria."
        ),
        metadata={"source": "moon.txt"},
    ),
    Document(
        page_content=(
            "Photosynthesis is the process by which green plants, algae, and some bacteria "
            "convert light energy into chemical energy stored in glucose. It takes place in "
            "the chloroplasts and releases oxygen as a byproduct."
        ),
        metadata={"source": "photosynthesis.txt"},
    ),
    Document(
        page_content=(
            "Mount Everest is Earth's highest mountain above sea level, located in the "
            "Mahalangur Himal sub-range of the Himalayas. Its peak is 8,849 meters above sea "
            "level. The first confirmed summit was by Edmund Hillary and Tenzing Norgay in 1953."
        ),
        metadata={"source": "everest.txt"},
    ),
]

QUESTION = "What was the name of the command module flown during the Apollo 11 mission?"

# Pipeline config — the identity of this pipeline, fed to config_fingerprint.
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 260, 40, 3
EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-base"
PROMPT_STR = (
    "Use the following context to answer the question.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)
CONFIG = {
    "embedding_model": EMBED_MODEL,
    "chunking": {"splitter": "RecursiveCharacterTextSplitter", "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
    "retrieval": {"vector_store": "FAISS", "k": TOP_K, "metric": "cosine"},
    "generation_model": GEN_MODEL,
    "prompt_template": PROMPT_STR,
}


def build_pipeline():
    """Construct the real LangChain RAG pipeline (LCEL), exactly tutorial-style."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)
    splits = splitter.split_documents(RAW_DOCS)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = FAISS.from_documents(splits, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    prompt = PromptTemplate.from_template(PROMPT_STR)
    llm = HuggingFacePipeline(
        pipeline=hf_pipeline("text2text-generation", model=GEN_MODEL, max_new_tokens=40, do_sample=False)
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # The canonical LCEL RAG chain — retrieval and generation fused, declaratively.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever, prompt, llm, format_docs, len(splits)


def main():
    rag_chain, retriever, prompt, llm, format_docs, n_chunks = build_pipeline()

    # --------------------------------------------------------------------- #
    # PART 1 — the real pipeline, completely unmodified.
    # --------------------------------------------------------------------- #
    print("=" * 70)
    print("PART 1: original LangChain LCEL pipeline (unmodified)")
    print("=" * 70)
    print(f"Corpus: {len(RAW_DOCS)} docs → {n_chunks} chunks (chunk_size={CHUNK_SIZE})")
    answer = rag_chain.invoke(QUESTION)
    print(f"Q: {QUESTION}")
    print(f"A: {answer!r}")

    # Show the model has weak parametric knowledge WITHOUT retrieval — so any
    # correct answer above is genuinely driven by retrieved context.
    no_context = llm.invoke(PROMPT_STR.format(context="(no context provided)", question=QUESTION))
    print(f"(same LLM, no retrieved context: {str(no_context).strip()!r})")

    # --------------------------------------------------------------------- #
    # PART 2 — add RAGLens WITHOUT changing pipeline logic, via `wrap`.
    # Decorators don't fit LCEL (no functions to annotate); wrap points at the
    # pipeline's own components.
    # --------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("PART 2: RAGLens via raglens.wrap (reusing the SAME retriever/prompt/llm)")
    print("=" * 70)

    gen_chain = prompt | llm | StrOutputParser()  # the pipeline's own generation half

    def retrieve(query):
        return retriever.invoke(query)  # returns List[Document]

    def generate(query, docs):
        return gen_chain.invoke({"context": format_docs(docs), "question": query})

    pipe = raglens.wrap(retrieve=retrieve, generate=generate, config=CONFIG)
    cap = pipe.run(QUESTION)

    print(f"config_fingerprint: {cap.trace.config_fingerprint}")
    print(f"Retrieved {len(cap.trace.final_chunks())} chunks; answer: {cap.trace.answer!r}\n")

    # Counterfactual attribution over the REAL retrieved chunks.
    cap.attribute()
    print("Counterfactual chunk attribution (importance = answer change when chunk removed):")
    for rc in raglens.top_chunks(cap, n=len(cap.trace.final_chunks())):
        src = rc.chunk.metadata.get("source", "?") if rc.chunk else "?"
        preview = rc.chunk.text[:64].replace("\n", " ") if rc.chunk else ""
        print(f"  {rc.score:5.3f}  [{src:>18}]  {preview}…")

    # Persist trace + auto-generate the datasheet.
    here = os.path.dirname(os.path.abspath(__file__))
    traces_path = os.path.join(here, "traces.jsonl")
    datasheet_path = os.path.join(here, "datasheet.md")
    cap.save(traces_path, append=False)
    raglens.write_datasheet(datasheet_path, [cap.trace], title="LangChain RAG — RAGLens Datasheet")
    print(f"\nWrote trace      → {traces_path}")
    print(f"Wrote datasheet  → {datasheet_path}")


if __name__ == "__main__":
    main()

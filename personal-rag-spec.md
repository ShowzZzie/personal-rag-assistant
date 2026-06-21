# Project: Personal RAG Knowledge Assistant

> **Portfolio project #2** — builds on SDR Agent foundations while introducing genuinely new skills: embeddings, vector databases, chunking, retrieval quality evaluation.

---

## The Problem This Solves

You consume information — research papers, DEXA scans, bloodwork PDFs, training plans, notes. You forget most of it. You can't query across it. You can't ask "given everything I've read about hypertrophy, what's the optimal training frequency for my situation?"

This project fixes that. Upload documents to topic-organized collections. Ask questions. Get answers grounded in your actual documents, with citations to the exact source chunks.

**Not a generic chatbot. A private, sourced, queryable knowledge base.**

---

## What You're Building

A CLI + API that:
1. Ingests documents (PDF, plain text, markdown) into topic collections
2. Chunks and embeds them into a vector database
3. Answers questions by retrieving relevant chunks and synthesizing an answer via Claude
4. Cites exactly which document and chunk each claim comes from
5. Stores conversation history per collection

**Callable via CLI and REST API. No frontend in V1.**

---

## New Skills This Forces

Skills you don't have from the SDR Agent:

| Skill | Why it matters |
|-------|---------------|
| Embeddings | Core AI engineering primitive — every RAG, search, and recommendation system uses them |
| Vector databases | How semantic search actually works |
| Chunking strategies | The thing that makes RAG fail or succeed — most tutorials skip the hard part |
| Retrieval evaluation | How to measure whether your system is finding the right content |
| Conversation memory | Managing multi-turn context without blowing token budgets |

---

## Learning Rules for This Project

**These are non-negotiable, based on what went wrong in the SDR Agent:**

1. **Docs first, always.** Before asking for help on any new concept, spend 30 minutes with the official docs. Come back with a specific question, not "I don't understand X."

2. **Break things deliberately.** When you learn chunking, intentionally use bad chunk sizes and observe what happens. When you learn retrieval, intentionally query with a bad prompt and observe what breaks.

3. **Write the code yourself.** Cursor and Claude Code can scaffold — but you type the logic. If you didn't write it, you don't own it.

4. **Understand before moving on.** If you implement something you can't explain in one sentence, stop. You skipped a step.

5. **Error messages are the docs.** Read the full stack trace before asking for help. The answer is usually in the error.

---

## Architecture

```
Documents (PDF/TXT/MD)
        ↓
   [Chunker]          ← splits into overlapping chunks
        ↓
   [Embedder]         ← converts chunks to vectors (OpenAI or local)
        ↓
 [Vector Store]       ← stores vectors + metadata (ChromaDB)
        ↓
   [Retriever]        ← finds top-k relevant chunks for a query
        ↓
   [Synthesizer]      ← Claude answers the question using retrieved chunks
        ↓
  Sourced Answer      ← answer + citations + chunk references
```

**Key insight:** the LLM never reads your full documents. It reads 3-5 relevant chunks retrieved for your specific question. This is what makes RAG cheap and scalable.

---

## Stack

- **Python 3.14+, uv**
- **Embeddings:** `text-embedding-3-small` via OpenAI API (cheap, good) OR `sentence-transformers` locally (free, slightly worse)
- **Vector database:** ChromaDB (local, no infra, persistent to disk)
- **LLM:** Claude Sonnet 4.6 for synthesis
- **Document parsing:** `pypdf` for PDFs, plain text for everything else
- **FastAPI** for API surface
- **Typer** for CLI
- **SQLite + SQLModel** for conversation history and document metadata
- **Pydantic v2** for all schemas
- **Langfuse** for observability (you already know this)
- **MyPy strict + Ruff** — enforced from day one

---

## Data Models

Define these in `schemas.py` **before writing any ingestion or retrieval code.**

```
DocumentChunk
├── chunk_id: str
├── document_id: str
├── collection: str
├── text: str
├── chunk_index: int
├── metadata: ChunkMetadata

ChunkMetadata
├── source_file: str
├── page_number: int | None
├── char_start: int
├── char_end: int

Document
├── document_id: str
├── collection: str
├── filename: str
├── ingested_at: datetime
├── chunk_count: int
├── embedding_model: str

RetrievedChunk
├── chunk: DocumentChunk
├── score: float          ← cosine similarity score
├── rank: int

Answer
├── question: str
├── answer: str
├── sources: list[RetrievedChunk]
├── model: str
├── input_tokens: int
├── output_tokens: int
├── collection: str
├── retrieved_at: datetime
```

**Rule:** every answer cites its sources. No unsourced claims. Same rule as SDR Agent.

---

## Project Structure

```
personal-rag/
├── src/rag/
│   ├── __init__.py
│   ├── config.py          # env loading, model names, paths
│   ├── schemas.py         # all Pydantic models
│   ├── chunker.py         # document chunking logic
│   ├── embedder.py        # embedding wrapper
│   ├── store.py           # ChromaDB wrapper
│   ├── retriever.py       # similarity search
│   ├── synthesizer.py     # Claude synthesis call
│   ├── pipeline.py        # ingest pipeline: chunk → embed → store
│   ├── query.py           # query pipeline: embed → retrieve → synthesize
│   ├── api.py             # FastAPI
│   └── cli.py             # Typer CLI
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_embedder.py
│   │   ├── test_retriever.py
│   │   └── test_synthesizer.py
│   └── evals/
│       ├── test_retrieval_evals.py
│       └── test_answer_evals.py
├── data/
│   ├── documents/         # raw uploaded files (gitignored)
│   └── chroma/            # vector store (gitignored)
├── pyproject.toml
├── .env.example
└── README.md
```

---

## V1 Feature Set

### Ingest
```bash
# Add a document to a collection
rag ingest --file paper.pdf --collection fitness

# Add multiple files
rag ingest --dir ./papers/ --collection fitness
```

- Parse PDF to text
- Chunk with configurable size and overlap
- Embed each chunk
- Store in ChromaDB under the collection namespace
- Record document metadata in SQLite

### Query
```bash
# Ask a question against a collection
rag ask --collection fitness "What's the optimal training frequency for natural lifters?"

# Get answer with source citations
rag ask --collection fitness --show-sources "How much protein do I need during a cut?"
```

- Embed the question
- Retrieve top-k chunks from ChromaDB
- Pass chunks + question to Claude
- Return answer with citations

### API
```
POST /ingest              # upload and ingest a document
GET  /collections         # list collections and document counts
POST /query               # ask a question, get sourced answer
GET  /documents/{id}      # document metadata
```

---

## The Chunking Problem (Read This Carefully)

Chunking is where most RAG systems fail. You need to understand why before implementing.

**The problem:** if you split a document randomly, you break semantic units. A chunk might start mid-sentence or split a key claim across two chunks — neither chunk is useful for retrieval.

**Three strategies to understand and experiment with:**

1. **Fixed-size chunking** — split every N characters with M overlap. Simple, dumb, surprisingly useful.
2. **Sentence-aware chunking** — split on sentence boundaries. Better semantic units.
3. **Recursive chunking** — try splitting on paragraphs, fall back to sentences, fall back to characters. LangChain popularized this; you'll implement it yourself.

**Your job:** implement all three, write tests for each, understand the tradeoffs. Don't just pick one and move on.

**Overlap matters:** chunks should overlap by 10-20% so retrieval doesn't miss a claim that spans a boundary.

---

## The Retrieval Problem

Retrieval is where RAG goes wrong at a more subtle level.

**Cosine similarity** finds chunks that are semantically similar to your question. But "similar to the question" ≠ "answers the question."

**Problems to understand:**
- A question like "what is protein?" retrieves chunks that define protein, not chunks that answer what *you* specifically want to know
- Longer documents get more chunks → higher chance of retrieval → biased toward verbose sources
- Questions phrased differently from how the document phrases things → poor retrieval

**For V1:** implement basic cosine similarity retrieval and document its failure modes. Understanding where it breaks is more valuable than adding complexity to fix it.

**Retrieval eval you must build:** given a question + the correct chunk that answers it (you define this manually for 10-20 cases), does your retriever find the right chunk in its top-3 results? Score = hits / total. This is Recall@3.

---

## Evals

Two layers for this project:

### Layer 1 — Retrieval evals (deterministic)

Build 15-20 "golden" question-chunk pairs. For each:
- A question you'd realistically ask
- The specific chunk in your documents that correctly answers it

Run retrieval, check if the correct chunk appears in top-3 results. Score = Recall@3.

```python
GOLDEN_PAIRS = [
    {
        "question": "What is the optimal rep range for hypertrophy?",
        "document": "menno_henselmans_hypertrophy.pdf",
        "expected_chunk_contains": "6-30 reps"
    },
    ...
]
```

**This is your primary quality signal.** If Recall@3 is below 0.7, your chunking or embedding is broken.

### Layer 2 — Answer evals (LLM-as-judge)

For 10 questions with known correct answers, use Claude Opus 4.7 as judge:
- Is the answer factually correct based on the source chunks?
- Are all claims cited?
- Does it hallucinate anything not in the retrieved chunks?

Same rubric structure as SDR Agent. You already know how to build this.

---

## Build Sequence

Do these in order. Don't start the next until the previous is tested and working.

**Phase 1 — Foundations (do this first, before any LLM calls)**
1. `schemas.py` — all Pydantic models
2. `chunker.py` — three chunking strategies, fully tested
3. `embedder.py` — wrapper around OpenAI embeddings, typed, tested with mocks
4. `store.py` — ChromaDB wrapper: add chunks, query by vector, delete collection

**Phase 2 — Pipeline**
5. `pipeline.py` — ingest: parse PDF → chunk → embed → store → record metadata
6. `retriever.py` — query: embed question → retrieve top-k → return RetrievedChunk list
7. Test end-to-end: ingest one document, query it, verify you get relevant chunks back

**Phase 3 — Synthesis**
8. `synthesizer.py` — pass retrieved chunks + question to Claude, get sourced Answer
9. Test: ingest a paper, ask a question, verify the answer cites the right chunks

**Phase 4 — Interfaces**
10. `cli.py` — `rag ingest` and `rag ask` commands
11. `api.py` — FastAPI endpoints

**Phase 5 — Evals**
12. Build retrieval eval harness (Layer 1)
13. Build answer eval harness (Layer 2)
14. Document scores in README

---

## What You're NOT Building in V1

- ❌ Conversation history / multi-turn chat (V2)
- ❌ Re-ranking (BM25 + semantic hybrid search) — understand it exists, don't build it yet
- ❌ Fine-tuning the embedding model
- ❌ Web UI
- ❌ Multi-user / auth
- ❌ Automatic re-ingestion when documents change
- ❌ LangChain or LlamaIndex — you implement the primitives yourself

---

## Docs to Read Before Writing Code

Read these in order. Don't skim.

1. **ChromaDB quickstart** — https://docs.trychroma.com/getting-started
2. **OpenAI embeddings guide** — https://platform.openai.com/docs/guides/embeddings
3. **What are embeddings?** — https://vicki.substack.com/p/neural-networks-are-just-matrix-multiplication (conceptual, 20 min)
4. **Chunking strategies** — https://www.pinecone.io/learn/chunking-strategies/ (read before implementing chunker.py)
5. **RAG evaluation** — https://docs.ragas.io/en/latest/concepts/metrics/ (understand Recall@K before building evals)

**Rule:** for each doc, write a 2-3 sentence summary in a `notes.md` file in the repo. This forces actual reading, not skimming.

---

## Success Criteria

Ship when ALL of these are true:

- ✅ `pytest tests/` passes
- ✅ `mypy --strict src/` passes
- ✅ `ruff check src/` passes
- ✅ Can ingest a PDF and query it end-to-end from CLI
- ✅ Every answer includes citations with source document and chunk
- ✅ Retrieval eval Recall@3 ≥ 0.70 on your golden pairs
- ✅ Answer eval scores documented in README
- ✅ Cost per query logged and documented
- ✅ README explains: setup, usage, how chunking works, eval scores, known limitations

---

## V2 Ideas (don't build yet)

- Conversation history — multi-turn Q&A within a collection
- Hybrid retrieval — combine semantic search (embeddings) with keyword search (BM25)
- Re-ranking — use a cross-encoder to rerank retrieved chunks before synthesis
- Document update detection — re-ingest when source files change
- MCP server — expose as a tool callable from Claude Desktop
- Web UI — simple file upload + chat interface

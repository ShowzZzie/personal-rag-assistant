# personal-rag

A local RAG system over a folder of PDFs. Ingest documents, chunk them, embed the chunks, store them in a vector DB, retrieve the top-k for a question, and synthesize a cited answer from an LLM.

Pipeline: **ingest → chunk → embed → store → retrieve → synthesize**.

## Stack

- Python 3.14
- ChromaDB (persistent, local) — chunk vectors
- SQLite via SQLModel — document metadata
- OpenAI `text-embedding-3-small` — embeddings
- Anthropic Claude Haiku 4.5 — answer synthesis
- pypdf — PDF text extraction
- FastAPI — HTTP API
- Typer — CLI

## Setup

```bash
uv sync
python -m spacy download en_core_web_sm
```

Copy `.env.example` to `.env` and fill in:

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

OpenAI key is used for embeddings, Anthropic key for synthesis (and for the eval judge, which calls Claude Opus 5).

All runtime settings are centralized in `src/rag/config.py` (a pydantic-settings `Settings` object, loaded from `.env`). The two API keys are required; everything else has a default and can be overridden by adding the matching env var:

| env var | default | controls |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | embedding model |
| `SYNTHESIS_MODEL` | `claude-haiku-4-5` | answer synthesis model |
| `SPACY_MODEL` | `en_core_web_sm` | sentence-aware chunker's spaCy model |
| `SYNTHESIS_MAX_TOKENS` | `1024` | max output tokens for synthesis |
| `CHUNK_SIZE` | `1500` | default chunk size (chars) |
| `CHUNK_OVERLAP` | `150` | default chunk overlap (chars) |
| `REFERENCE_CITATION_THRESHOLD` | `0.10` | citation density above which a chunk is dropped |
| `TOP_K` | `5` | default number of chunks retrieved per query |
| `CHROMA_PATH` | `data/chroma` | ChromaDB persistence directory |
| `SQLITE_DB_PATH` | `data/database.db` | SQLite metadata DB path |

## Usage

### CLI

```bash
# ingest a single file
rag ingest --file path/to/doc.pdf --collection sleep

# ingest every PDF in a directory
rag ingest --dir path/to/pdfs/ --collection sleep

# ask a question
rag ask "how much sleep should I be getting?" --collection sleep --show-sources
```

### API

```bash
uvicorn rag.api:app --reload
```

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Upload a PDF (multipart), ingest into a collection |
| `GET` | `/collections` | List collections and document counts |
| `POST` | `/query` | `{question, collection, top_k}` → answer with citations |
| `GET` | `/documents/{id}` | Metadata for one ingested document |

## Architecture: two databases

Chunk vectors live in **ChromaDB** (`data/chroma/`); document metadata (filename, collection, chunk count, ingest timestamp, embedding model) lives in **SQLite** (`data/database.db`) via SQLModel.

They're split because they serve different access patterns. Retrieval needs approximate nearest-neighbor search over embeddings — that's what Chroma's HNSW index is for. Bookkeeping ("what's in this collection", "when was this file ingested", "look up document by id") is relational lookup, not similarity search, and doesn't belong in a vector index. Chroma stores the chunk metadata it needs to reconstruct results (`document_id`, `chunk_index`, `source_file`, `char_start`/`char_end`) alongside each vector, but the SQLite `Document` table is the source of truth for document-level records.

Chroma's collection is created with `metadata={"hnsw:space": "cosine"}` so distance is cosine distance; retrieval converts it back to a similarity score with `score = 1 - distance`.

## Chunking

Three strategies are implemented in `src/rag/chunker.py`:

- **Fixed-size** (`chunker_fixed_size`) — hard character-count windows with overlap. No awareness of sentence or paragraph boundaries.
- **Sentence-aware** (`chunker_sentence_aware`) — uses spaCy (`en_core_web_sm`) to split into sentences, then groups sentences up to the size limit. Preserves sentence boundaries at the cost of variable chunk sizes.
- **Recursive** (`chunker_recursive`) — **default**, used by `ingest()`. Splits on a separator list (`["\n\n", "\n", ". "]`) in order: try paragraph breaks first, then line breaks, then sentence breaks, recursing into any block still over `size` before falling back to fixed-size hard splitting.

### Why `join_small_chunks` exists

PDF text extraction breaks lines at every visual line wrap, not just at paragraph ends — pypdf hands back a `\n` roughly every time a line wrapped in the original layout. Recursive chunking splits on `\n` first, so this produces a flood of tiny fragments (~55 characters each) instead of paragraph-sized chunks. `join_small_chunks` (in `pipeline.py`) is a post-processing pass that reassembles these fragments back up toward the target chunk size, carrying an overlap seed forward between merged chunks. Without it, ingestion for real PDFs produces chunks far too small to give the embedding model useful context.

## Evals

### Layer 1: retrieval (Recall@k)

14 golden pairs (drafted with LLM assistance, then hand-verified against stored chunks), evaluated with **adjacency-aware Recall**: a hit counts if the retrieved chunk *or either of its neighbors* (same `document_id`) contains the target phrase. This matters because chunks overlap — the exact phrase often sits one chunk over from the one actually retrieved.

Chunk-size sweep, overlap held at 10% of size:

| chunk size / overlap | @3 | @5 | @10 |
|---|---|---|---|
| 500/50 | 0.43 | 0.50 | 0.64 |
| 1000/100 | 0.57 | 0.71 | 0.93 |
| 1500/150 | **0.79** | **0.86** | 0.93 |
| 2000/200 | 0.64 | 0.71 | 0.93 |

**Shipped config: 1500/150.**

The curve has a clear shape. At 500 chars, chunks don't carry enough context for the embedding to represent the topic well, and with ~460 chunks in the corpus, too many near-duplicates compete for the top-3 slots. At 2000 chars, chunks start mixing multiple topics, so the resulting vector sits between them and matches any single query only weakly — recall drops back down from the 1500 peak. Recall@10 is flat at 0.93 from 1000 chars up: the right chunk stays *findable* at that size regardless; what changes with chunk size is how well it *ranks* into the top 3–5.

**Metric caveat:** strict substring matching (hit only if the retrieved chunk itself contains the phrase) gives 0.36 at 500/50, versus 0.43 for adjacency-aware. The gap exists because of chunk overlap — adjacency-aware credits the case where the answer legitimately lands in a neighboring chunk instead of the one holding the literal phrase. It's a real correction, not a way to inflate the number: it only moved the score by one pair at 500/50, so most of the shortfall at that size is a genuine retrieval problem, not a measurement artifact.

Run it: `pytest tests/evals/test_retrieval_evals.py` (hits the real OpenAI embedding API — not mocked, since the point is measuring actual retrieval quality).

### Layer 2: answer quality (LLM-as-judge)

10 questions × 3 rubric criteria (`factual`, `cited`, `grounded`), judged by Claude Opus 5 via a forced tool call. Scoring is strict: only `YES` counts as a pass, `PARTIALLY` and `NO` both score zero.

**Result: 21/30 = 0.70.**

Most `PARTIALLY` scores came from the judge flagging that "the context doesn't contain information about X" in cases where the retrieved chunks actually did contain relevant material — i.e. the synthesizer is over-conservative and hedges rather than using context that's there. This is a synthesis-prompt problem, not a retrieval problem.

Run it: `pytest tests/evals/test_answer_evals.py` (hits both the OpenAI and Anthropic APIs live — no mocking, real query → real judge).

## Known limitations

- **Ingest is slow.** Embedding calls are made one chunk at a time, sequentially. OpenAI's batch embeddings endpoint would fix this but isn't implemented.
- **Reference-list filtering has no measured benefit.** `is_reference_chunk` in `pipeline.py` regex-matches `year;volume(` citation patterns and drops chunks where citation density exceeds 0.10 per 100 characters, removing about 7% of chunks (mostly bibliography pages). It did not measurably move Recall in either direction — it's shipped because it removes obvious noise, not because it was proven to help retrieval.
- **3 of 14 golden pairs still miss** even at the 1500/150 config. Two are cases where retrieval found a different, topically valid chunk that the golden pair simply didn't credit (the golden-pair phrase is stricter than "correct answer"). One is a genuine miss: an abstract question phrased in everyday terms doesn't share vocabulary with the concrete clinical language in the source chunk.
- **LLM-as-judge scores are noisy between runs.** The same query/answer pair can get a different score on a re-run of the eval; 0.70 should be read as a point estimate, not a fixed number.

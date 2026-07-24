# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dev server
uvicorn app.main:app --reload

# Run all tests
pytest

# Run a single test
pytest tests/test_routes.py::test_health
```

## Architecture

This is a **RAG (Retrieval-Augmented Generation) Document Copilot** built with FastAPI. The pipeline has three stages that are currently stubs awaiting implementation:

1. **Ingest** ([app/ingest.py](app/ingest.py)): PDF → text chunks → embeddings → vector store upsert. Chunk size and overlap are controlled by `CHUNK_SIZE`/`CHUNK_OVERLAP` env vars (defaults: 500/50). PDFs are expected in `data/`.

2. **Retrieve** ([app/retrieve.py](app/retrieve.py)): Embeds the question, performs nearest-neighbour search against the vector store, returns top-k chunks.

3. **Generate** ([app/generate.py](app/generate.py)): Builds a prompt from retrieved chunks + question, calls the Claude API, returns markdown with inline chunk citations.

**Request/response flow**: `POST /ingest` → `ingest_document(filename)` | `POST /query` → `retrieve_chunks()` → `generate_answer()` → `QueryResponse` with `answer`, `citations`, and `question` fields.

**Config** ([app/config.py](app/config.py)): `pydantic-settings` loads from `.env`. `ANTHROPIC_API_KEY` is required with no default — tests must set it before importing app modules (see [tests/conftest.py](tests/conftest.py)).

## Environment

Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY`. The `EMBEDDING_MODEL` field is blank by default; set it when the retrieve/generate stubs are implemented.

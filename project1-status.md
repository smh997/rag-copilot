# Project 1 Status

**Pipeline COMPLETE + grounding verified.** `/query` returns grounded, cited answers; off-context questions are correctly refused.

## Progress

- **ingest.py done** — pypdf + tiktoken chunking, 23 chunks from attention.pdf, pages tracked, 16 tests pass.
- **ingest + storage done** — 23 chunks from attention.pdf embedded via Cohere (embed-english-v3.0, search_document), stored in Chroma "papers" collection, persisted to ./chroma_db.
- **retrieve.py done** — question embedded via Cohere search_query, top-k nearest-neighbour lookup against Chroma.
- **generate.py done** — hand-built numbered source prompt (no Cohere `documents=` param), grounding preamble, answer via `command-a-03-2025`, deduplicated citations. `/query` wired end-to-end. 29 tests pass.

Full chain: ingest → Cohere embed (search_document) → Chroma → retrieve (search_query) → generate (command-a-03-2025).

### Gotchas logged

- **input_type asymmetry** — must embed documents with `search_document` and queries with `search_query`; mismatched input_type silently degrades retrieval quality rather than erroring.
- **Python 3.12 pin** — Python 3.14 has no chromadb wheels yet.
- **command-r retired → command-a-03-2025** — Cohere deprecated the model mid-project; keeping `generation_model` in config instead of hardcoding meant the swap was a one-line change.

### Known polish (not yet done)

- Suppress citations when the answer is "I don't have enough information."
- Add reranking step before generation.
- Write README.

## Next

Project 1 frontend/deploy, OR start Module 7 (Tool Use) for Project 2.

## File Structure

```
rag-copilot/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI app: /health, /ingest, /query
│   ├── config.py      # pydantic-settings (COHERE_API_KEY, chunk params, embedding_model, generation_model)
│   ├── models.py      # Pydantic request/response models
│   ├── ingest.py      # PDF → chunks → embeddings
│   ├── retrieve.py    # question → top-k chunks (Cohere search_query + Chroma)
│   ├── generate.py    # chunks + question → grounded, cited answer (Cohere chat)
│   └── store.py       # shared Cohere/Chroma clients, embed + upsert
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_routes.py
│   ├── test_ingest.py
│   ├── test_store.py
│   ├── test_retrieve.py
│   └── test_generate.py
├── data/
│   └── .gitkeep
├── requirements.txt
├── .env.example
└── .gitignore
```

# Project 1 Status

**Backend scaffolded, stubs only — RAG logic comes Week 3.**

## Progress

- **ingest.py done** — pypdf + tiktoken chunking, 23 chunks from attention.pdf, pages tracked, 16 tests pass.
- **ingest + storage done** — 23 chunks from attention.pdf embedded via Cohere (embed-english-v3.0, search_document), stored in Chroma "papers" collection, persisted to ./chroma_db. 20 tests pass. Note: pinned to Python 3.12 (3.14 lacks chromadb wheels). Next: wire retrieve.py — embed query with search_query, return top-k.

## File Structure

```
rag-copilot/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI app: /health, /ingest, /query
│   ├── config.py      # pydantic-settings (COHERE_API_KEY, chunk params, embedding_model)
│   ├── models.py      # Pydantic request/response models
│   ├── ingest.py      # stub: PDF → chunks → embeddings
│   ├── retrieve.py    # stub: question → top-k chunks
│   └── generate.py    # stub: context + question → cited answer
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_routes.py # 4 passing route tests
├── data/
│   └── .gitkeep
├── requirements.txt
├── .env.example
└── .gitignore
```

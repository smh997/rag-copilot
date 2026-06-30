# RAG Document Copilot

A FastAPI service that answers questions about PDF documents using retrieval-augmented generation — grounded answers with page citations, and an honest refusal when the answer isn't in the sources.

**What this demonstrates:** RAG pipeline design, dense embeddings, vector similarity search, and grounded generation with citations.

---

## Architecture

```
PDF file
  └─► pypdf extraction (page-by-page text)
        └─► tiktoken chunking (500 tokens / 50 overlap, cl100k_base encoding)
              └─► Cohere embed-english-v3.0  [input_type=search_document]
                    └─► Chroma PersistentClient  (./chroma_db)
                          └─► query: embed question  [input_type=search_query]
                                └─► Chroma top-k nearest-neighbour search
                                      └─► command-a-03-2025 generation
                                            └─► answer + page citations
                                                (or no-info refusal with empty citations)
```

Chunk metadata (`source`, `page`) travels through the pipeline and surfaces as citations in every answer.

---

## Tech Stack

| Layer | Library |
|---|---|
| API | FastAPI |
| PDF extraction | pypdf |
| Tokenisation | tiktoken (`cl100k_base`) |
| Embeddings | Cohere `embed-english-v3.0` |
| Vector store | ChromaDB (persistent, local) |
| Generation | Cohere `command-a-03-2025` |
| Testing | pytest |

---

## Quickstart

> **Python 3.12 required.** ChromaDB does not publish wheels for 3.13/3.14 yet; use 3.12 to avoid a source build.

```bash
git clone <repo-url>
cd rag-copilot

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Open .env and set COHERE_API_KEY=<your key>
# Free key at dashboard.cohere.com

uvicorn app.main:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API explorer.

---

## Usage

### 1. Ingest a PDF

Place your PDF in `data/` then POST its filename:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"filename": "attention.pdf"}'
```

```json
{
  "filename": "attention.pdf",
  "chunks_indexed": 23,
  "status": "ok"
}
```

### 2. Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is multi-head attention?", "top_k": 5}'
```

**Grounded answer — question the sources can answer:**

```json
{
  "question": "What is multi-head attention?",
  "answer": "Multi-head attention is a mechanism that projects queries, keys, and values into multiple lower-dimensional subspaces, performs scaled dot-product attention in parallel on these projections, and then concatenates the results before a final projection. This allows the model to attend to information from different representation subspaces at different positions. (Source 1, p.4)",
  "citations": [
    {"source": "attention.pdf", "page": 4},
    {"source": "attention.pdf", "page": 3},
    {"source": "attention.pdf", "page": 2}
  ]
}
```

**No-info refusal — question the sources cannot answer:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?", "top_k": 5}'
```

```json
{
  "question": "What is the capital of France?",
  "answer": "The sources do not contain the answer to this question.",
  "citations": []
}
```

The model is instructed to reply with that exact sentence when the retrieved chunks don't contain the answer; the service detects it and returns an empty citations list.

---

## Design Decisions

### Why ChromaDB

Zero setup — a `PersistentClient` writes to `./chroma_db` with no external process. The workflow (collections, upsert, metadata filtering, similarity search) maps directly to production alternatives when needed:

- **pgvector** — vectors live in Postgres alongside relational data; good when the stack is already Postgres-based.
- **Pinecone** — fully managed ANN indexing; good when scale matters and you don't want to run infra.

### Why Cohere

The embedding model and generation model come from the same provider, which avoids integration friction. Cohere's `embed-english-v3.0` produces 1024-dimensional vectors; `command-a-03-2025` is the generation model. Cohere also exposes a `rerank` endpoint that can improve retrieval quality as a drop-in future upgrade.

### `input_type` asymmetry

Cohere's embedding API distinguishes between indexing and querying:

```
chunks  → input_type="search_document"
question → input_type="search_query"
```

Using `search_document` for a query (or vice versa) causes silently degraded retrieval — vectors that should be close are not. Both calls use the same model (`embed-english-v3.0`); only the `input_type` differs.

---

## Running Tests

```bash
pytest
```

Tests are fully mocked — no Cohere calls, no Chroma writes. Coverage includes ingest chunking, embedding/store, retrieval, generation (including the no-info refusal path), and all API routes.

---

## Future Work

- **Cohere rerank** — add a rerank step between retrieval and generation to improve chunk selection quality.
- **Web frontend** — a minimal chat UI to make the pipeline interactive without `curl`.
- **Deployment** — containerise with Docker; add a managed vector store (Pinecone or a hosted Chroma instance).
- **Multi-document corpus** — namespace collections by user or topic so multiple PDFs can be queried independently.

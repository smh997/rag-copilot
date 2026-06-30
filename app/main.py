from fastapi import FastAPI, HTTPException

from app.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app import ingest, retrieve, generate, store

app = FastAPI(title="RAG Document Copilot")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_route(body: IngestRequest) -> IngestResponse:
    try:
        chunks = ingest.ingest_document(body.filename)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {body.filename}")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {body.filename}")
    try:
        count = store.store_chunks(chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Store error: {exc}")
    return IngestResponse(filename=body.filename, chunks_indexed=count, status="ok")


@app.post("/query", response_model=QueryResponse)
def query_route(body: QueryRequest) -> QueryResponse:
    chunks = retrieve.retrieve_chunks(body.question, body.top_k)
    try:
        result = generate.generate_answer(body.question, chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}")
    return QueryResponse(answer=result["answer"], citations=result["citations"], question=body.question)

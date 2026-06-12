from fastapi import FastAPI

from app.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app import ingest, retrieve, generate

app = FastAPI(title="RAG Document Copilot")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_route(body: IngestRequest) -> IngestResponse:
    try:
        chunks_indexed = ingest.ingest_document(body.filename)
    except NotImplementedError:
        chunks_indexed = 0
    return IngestResponse(filename=body.filename, chunks_indexed=chunks_indexed, status="stub")


@app.post("/query", response_model=QueryResponse)
def query_route(body: QueryRequest) -> QueryResponse:
    try:
        chunks = retrieve.retrieve_chunks(body.question, body.top_k)
        answer = generate.generate_answer(body.question, chunks)
    except NotImplementedError:
        answer = "stub"
    return QueryResponse(answer=answer, citations=[], question=body.question)

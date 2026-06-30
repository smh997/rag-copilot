from pydantic import BaseModel


class IngestRequest(BaseModel):
    filename: str


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    status: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    question: str

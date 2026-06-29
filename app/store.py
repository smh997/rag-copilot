import chromadb
import cohere

from app.config import settings

CHROMA_PATH = "./chroma_db"

_co: cohere.Client | None = None
_client: chromadb.ClientAPI | None = None


def _get_co() -> cohere.Client:
    global _co
    if _co is None:
        _co = cohere.Client(api_key=settings.cohere_api_key)
    return _co


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    texts = [c["text"] for c in chunks]
    response = _get_co().embed(texts=texts, model=settings.embedding_model, input_type="search_document")
    return response.embeddings


def store_chunks(chunks: list[dict]) -> int:
    embeddings = embed_chunks(chunks)
    collection = _get_client().get_or_create_collection("papers")
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
    ids = [f"{c['source']}-{i}" for i, c in enumerate(chunks)]
    collection.upsert(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return len(chunks)

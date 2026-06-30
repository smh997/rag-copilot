from app.config import settings
from app.store import _get_client, _get_co


def retrieve_chunks(question: str, top_k: int = 5) -> list[dict]:
    query_vector = _get_co().embed(
        texts=[question],
        model=settings.embedding_model,
        input_type="search_query",
    ).embeddings[0]

    try:
        collection = _get_client().get_or_create_collection("papers")
        results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    except Exception:
        return []

    return [
        {"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

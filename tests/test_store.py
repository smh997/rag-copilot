from unittest.mock import MagicMock, patch

import app.store as store
from app.config import settings

SAMPLE_CHUNKS = [
    {"text": "chunk one", "source": "doc.pdf", "page": 1},
    {"text": "chunk two", "source": "doc.pdf", "page": 2},
]
SAMPLE_EMBEDDINGS = [[0.1] * 1024, [0.2] * 1024]


def _mock_co(embeddings=None):
    co = MagicMock()
    co.embed.return_value.embeddings = embeddings or SAMPLE_EMBEDDINGS
    return co


def _mock_client():
    client = MagicMock()
    collection = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client, collection


def test_embed_chunks_single_batched_call():
    mock_co = _mock_co()
    with patch.object(store, "_co", mock_co):
        result = store.embed_chunks(SAMPLE_CHUNKS)

    mock_co.embed.assert_called_once_with(
        texts=["chunk one", "chunk two"],
        model=settings.embedding_model,
        input_type="search_document",
    )
    assert result == SAMPLE_EMBEDDINGS


def test_store_chunks_upsert_alignment():
    mock_co = _mock_co()
    mock_client, mock_collection = _mock_client()
    with patch.object(store, "_co", mock_co), patch.object(store, "_client", mock_client):
        store.store_chunks(SAMPLE_CHUNKS)

    kw = mock_collection.upsert.call_args.kwargs
    n = len(SAMPLE_CHUNKS)
    assert len(kw["documents"]) == n
    assert len(kw["embeddings"]) == n
    assert len(kw["metadatas"]) == n
    assert len(kw["ids"]) == n
    assert kw["documents"][0] == "chunk one"
    assert kw["embeddings"][0] == SAMPLE_EMBEDDINGS[0]
    assert kw["metadatas"][0] == {"source": "doc.pdf", "page": 1}
    assert kw["ids"][0] == "doc.pdf-0"
    assert kw["ids"][1] == "doc.pdf-1"


def test_store_chunks_returns_chunk_count():
    mock_co = _mock_co()
    mock_client, _ = _mock_client()
    with patch.object(store, "_co", mock_co), patch.object(store, "_client", mock_client):
        result = store.store_chunks(SAMPLE_CHUNKS)

    assert result == len(SAMPLE_CHUNKS)

from unittest.mock import MagicMock, patch

import app.store as store
from app.config import settings
from app.retrieve import retrieve_chunks

QUERY = "what is attention?"
QUERY_VECTOR = [0.5] * 1024

CHROMA_RESULTS = {
    "documents": [["chunk one text", "chunk two text"]],
    "distances": [[0.1, 0.2]],
    "metadatas": [[{"source": "doc.pdf", "page": 1}, {"source": "doc.pdf", "page": 2}]],
}


def _mock_co():
    co = MagicMock()
    co.embed.return_value.embeddings = [QUERY_VECTOR]
    return co


def _mock_client(results=CHROMA_RESULTS):
    collection = MagicMock()
    collection.query.return_value = results
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client, collection


def test_embed_uses_search_query_input_type():
    mock_co = _mock_co()
    mock_client, _ = _mock_client()
    with patch.object(store, "_co", mock_co), patch.object(store, "_client", mock_client):
        retrieve_chunks(QUERY, top_k=2)

    mock_co.embed.assert_called_once_with(
        texts=[QUERY],
        model=settings.embedding_model,
        input_type="search_query",
    )


def test_query_uses_query_embeddings_not_query_texts():
    mock_co = _mock_co()
    mock_client, mock_collection = _mock_client()
    with patch.object(store, "_co", mock_co), patch.object(store, "_client", mock_client):
        retrieve_chunks(QUERY, top_k=2)

    kw = mock_collection.query.call_args.kwargs
    assert "query_embeddings" in kw
    assert "query_texts" not in kw
    assert kw["query_embeddings"] == [QUERY_VECTOR]


def test_query_passes_n_results_equal_to_top_k():
    mock_co = _mock_co()
    mock_client, mock_collection = _mock_client()
    with patch.object(store, "_co", mock_co), patch.object(store, "_client", mock_client):
        retrieve_chunks(QUERY, top_k=3)

    assert mock_collection.query.call_args.kwargs["n_results"] == 3


def test_returns_dicts_with_correct_keys_and_values():
    mock_co = _mock_co()
    mock_client, _ = _mock_client()
    with patch.object(store, "_co", mock_co), patch.object(store, "_client", mock_client):
        result = retrieve_chunks(QUERY, top_k=2)

    assert len(result) == 2
    assert result[0] == {"text": "chunk one text", "source": "doc.pdf", "page": 1, "distance": 0.1}
    assert result[1] == {"text": "chunk two text", "source": "doc.pdf", "page": 2, "distance": 0.2}


def test_empty_collection_returns_empty_list():
    mock_co = _mock_co()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value.query.side_effect = Exception(
        "Collection has 0 items"
    )
    with patch.object(store, "_co", mock_co), patch.object(store, "_client", mock_client):
        result = retrieve_chunks(QUERY, top_k=5)

    assert result == []


def test_embed_failure_returns_empty_list():
    mock_co = MagicMock()
    mock_co.embed.side_effect = Exception("Cohere unavailable")
    mock_client, _ = _mock_client()
    with patch.object(store, "_co", mock_co), patch.object(store, "_client", mock_client):
        result = retrieve_chunks(QUERY, top_k=5)
    assert result == []

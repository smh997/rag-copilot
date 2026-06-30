from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_missing_file_returns_404():
    response = client.post("/ingest", json={"filename": "nonexistent.pdf"})
    assert response.status_code == 404


def test_ingest_route_chunk_count():
    mock_chunks = [
        {"text": "chunk one", "source": "doc.pdf", "page": 1},
        {"text": "chunk two", "source": "doc.pdf", "page": 1},
    ]
    with patch("app.ingest.ingest_document", return_value=mock_chunks), \
         patch("app.store.store_chunks", return_value=2):
        response = client.post("/ingest", json={"filename": "doc.pdf"})
    assert response.status_code == 200
    body = response.json()
    assert body["chunks_indexed"] == 2
    assert body["status"] == "ok"
    assert body["filename"] == "doc.pdf"


def test_query_stub():
    with patch("app.retrieve.retrieve_chunks", return_value=[]):
        response = client.post("/query", json={"question": "What is RLHF?", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "stub"
    assert body["citations"] == []
    assert body["question"] == "What is RLHF?"


def test_query_validation():
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_ingest_path_traversal_returns_400():
    response = client.post("/ingest", json={"filename": "../../etc/passwd"})
    assert response.status_code == 400


def test_ingest_store_failure_returns_500():
    mock_chunks = [{"text": "chunk one", "source": "doc.pdf", "page": 1}]
    with patch("app.ingest.ingest_document", return_value=mock_chunks), \
         patch("app.store.store_chunks", side_effect=Exception("Chroma unavailable")):
        response = client.post("/ingest", json={"filename": "doc.pdf"})
    assert response.status_code == 500

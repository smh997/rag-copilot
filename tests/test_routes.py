from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_stub():
    response = client.post("/ingest", json={"filename": "paper.pdf"})
    assert response.status_code == 200
    body = response.json()
    assert body["chunks_indexed"] == 0
    assert body["status"] == "stub"


def test_query_stub():
    response = client.post("/query", json={"question": "What is RLHF?", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "stub"
    assert body["citations"] == []
    assert body["question"] == "What is RLHF?"


def test_query_validation():
    response = client.post("/query", json={})
    assert response.status_code == 422

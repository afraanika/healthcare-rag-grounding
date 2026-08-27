from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api as api_module
from app.embedder import EmbeddingClient
from app.generator import LLMClient
from app.pipeline import RagPipeline
from app.vectorstore import VectorStore

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


class _FakeOllamaClient:
    def generate(self, model, prompt):
        return {"response": "Fixtureville was founded in 1900 [1]."}


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingClient()


@pytest.fixture
def client(tmp_path, monkeypatch, embedder):
    vectorstore = VectorStore(persist_dir=str(tmp_path))
    llm = LLMClient(model="llama3", client=_FakeOllamaClient())
    test_pipeline = RagPipeline(embedder=embedder, vectorstore=vectorstore, llm=llm)
    monkeypatch.setattr(api_module, "pipeline", test_pipeline)
    return TestClient(api_module.app)


def test_health_endpoint_returns_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert "status" in response.json()


def test_ingest_rejects_non_pdf_file(client):
    response = client.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 400


def test_ingest_then_query_returns_grounded_answer_with_citation(client):
    with FIXTURE.open("rb") as f:
        ingest_resp = client.post("/ingest", files={"file": ("sample.pdf", f, "application/pdf")})

    assert ingest_resp.status_code == 200
    ingest_body = ingest_resp.json()
    assert ingest_body["document_id"] == "sample.pdf"
    assert ingest_body["num_pages"] == 2
    assert ingest_body["num_chunks"] > 0

    query_resp = client.post("/query", json={"question": "When was Fixtureville founded?", "top_k": 2})

    assert query_resp.status_code == 200
    body = query_resp.json()
    assert "1900" in body["answer"]
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["source_file"] == "sample.pdf"


def test_documents_endpoint_lists_ingested_documents(client):
    with FIXTURE.open("rb") as f:
        client.post("/ingest", files={"file": ("sample.pdf", f, "application/pdf")})

    resp = client.get("/documents")

    assert resp.status_code == 200
    assert resp.json().get("sample.pdf", 0) > 0


def test_delete_document_removes_it(client):
    with FIXTURE.open("rb") as f:
        client.post("/ingest", files={"file": ("sample.pdf", f, "application/pdf")})

    del_resp = client.delete("/documents/sample.pdf")
    assert del_resp.status_code == 200

    resp = client.get("/documents")
    assert "sample.pdf" not in resp.json()

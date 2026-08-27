import shutil
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, UploadFile

from app.config import settings
from app.models import AnswerResponse, QueryRequest
from app.pipeline import RagPipeline

app = FastAPI(title="healthcare-rag-grounding")
pipeline = RagPipeline()


@app.get("/health")
def health() -> dict:
    chroma_ok = True
    try:
        pipeline.list_documents()
    except Exception:
        chroma_ok = False

    ollama_ok = True
    try:
        httpx.get(f"{settings.ollama_host}/api/version", timeout=2.0).raise_for_status()
    except Exception:
        ollama_ok = False

    return {
        "status": "ok" if chroma_ok and ollama_ok else "degraded",
        "chroma": chroma_ok,
        "ollama": ollama_ok,
    }


@app.post("/ingest")
async def ingest(file: UploadFile) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / file.filename
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        return pipeline.ingest_document(tmp_path)


@app.get("/documents")
def list_documents() -> dict:
    return pipeline.list_documents()


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict:
    pipeline.delete_document(document_id)
    return {"deleted": document_id}


@app.post("/query", response_model=AnswerResponse)
def query(request: QueryRequest) -> AnswerResponse:
    return pipeline.answer_query(request.question, top_k=request.top_k)

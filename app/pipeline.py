from pathlib import Path

from app.chunker import chunk_pages
from app.citations import attach_citations
from app.embedder import EmbeddingClient
from app.generator import LLMClient, build_prompt
from app.ingest import extract_pages
from app.models import AnswerResponse
from app.retriever import Retriever
from app.vectorstore import VectorStore


class RagPipeline:
    def __init__(
        self,
        embedder: EmbeddingClient | None = None,
        vectorstore: VectorStore | None = None,
        llm: LLMClient | None = None,
    ):
        self._embedder = embedder or EmbeddingClient()
        self._vectorstore = vectorstore or VectorStore()
        self._llm = llm or LLMClient()
        self._retriever = Retriever(embedder=self._embedder, vectorstore=self._vectorstore)

    def ingest_document(self, path: str | Path) -> dict:
        path = Path(path)
        pages = extract_pages(path)
        chunks = chunk_pages(pages, source_file=path.name)
        if chunks:
            vectors = self._embedder.embed([c.text for c in chunks])
            self._vectorstore.add_chunks(chunks, vectors)
        return {"document_id": path.name, "num_pages": len(pages), "num_chunks": len(chunks)}

    def answer_query(self, question: str, top_k: int = 5) -> AnswerResponse:
        retrieved = self._retriever.retrieve(question, top_k=top_k)
        prompt = build_prompt(question, retrieved)
        answer_text = self._llm.generate(prompt)
        return attach_citations(answer_text, retrieved)

    def list_documents(self) -> dict[str, int]:
        return self._vectorstore.list_documents()

    def delete_document(self, source_file: str) -> None:
        self._vectorstore.delete_document(source_file)

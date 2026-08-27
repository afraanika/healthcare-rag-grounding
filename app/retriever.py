from dataclasses import dataclass

from app.embedder import EmbeddingClient
from app.vectorstore import VectorStore


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    distance: float


class Retriever:
    def __init__(self, embedder: EmbeddingClient | None = None, vectorstore: VectorStore | None = None):
        self._embedder = embedder or EmbeddingClient()
        self._vectorstore = vectorstore or VectorStore()

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        [query_vector] = self._embedder.embed([query])
        matches = self._vectorstore.query(query_vector, top_k=top_k)
        return [
            RetrievedChunk(
                id=m["id"],
                text=m["text"],
                source_file=m["metadata"]["source_file"],
                page_number=m["metadata"]["page_number"],
                chunk_index=m["metadata"]["chunk_index"],
                distance=m["distance"],
            )
            for m in matches
        ]

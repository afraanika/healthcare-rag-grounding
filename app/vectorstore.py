import chromadb

from app.config import settings
from app.models import Chunk

COLLECTION_NAME = "clinical_docs"


class VectorStore:
    def __init__(self, persist_dir: str | None = None, collection_name: str = COLLECTION_NAME):
        self._client = chromadb.PersistentClient(path=persist_dir or settings.chroma_dir)
        self._collection = self._client.get_or_create_collection(collection_name)

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source_file": c.source_file,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                }
                for c in chunks
            ],
        )

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        result = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
        if not result["ids"][0]:
            return []
        return [
            {"id": id_, "text": doc, "metadata": meta, "distance": dist}
            for id_, doc, meta, dist in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def list_documents(self) -> dict[str, int]:
        all_data = self._collection.get()
        counts: dict[str, int] = {}
        for meta in all_data["metadatas"]:
            counts[meta["source_file"]] = counts.get(meta["source_file"], 0) + 1
        return counts

    def delete_document(self, source_file: str) -> None:
        self._collection.delete(where={"source_file": source_file})

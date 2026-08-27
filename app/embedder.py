from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingClient:
    def __init__(self, model_name: str | None = None):
        self._model = SentenceTransformer(model_name or settings.embed_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()

from ollama import Client

from app.config import settings
from app.retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a clinical information assistant. Answer the user's question using ONLY "
    "the numbered context passages below. For every claim in your answer, cite the "
    "passage(s) it came from using square brackets, e.g. [1] or [1][2], placed right "
    "after the relevant sentence. If the context does not contain the answer, say so "
    "plainly instead of guessing."
)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = "\n\n".join(
        f"[{i}] (source: {c.source_file}, page {c.page_number})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    )
    return f"{SYSTEM_PROMPT}\n\nContext passages:\n{context_blocks}\n\nQuestion: {query}\nAnswer:"


class LLMClient:
    def __init__(self, model: str | None = None, host: str | None = None, client=None):
        self._model = model or settings.llm_model
        self._client = client or Client(host=host or settings.ollama_host)

    def generate(self, prompt: str) -> str:
        response = self._client.generate(model=self._model, prompt=prompt)
        return response["response"]

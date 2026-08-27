import pytest

from app.generator import LLMClient, build_prompt
from app.retriever import RetrievedChunk


def _chunk(text, source_file, page_number, chunk_index=0):
    return RetrievedChunk(
        id="x",
        text=text,
        source_file=source_file,
        page_number=page_number,
        chunk_index=chunk_index,
        distance=0.1,
    )


def test_build_prompt_numbers_context_and_includes_sources():
    chunks = [
        _chunk("Metformin is first-line therapy for T2DM.", "who.pdf", 3),
        _chunk("Insulin is used in later stages.", "who.pdf", 4),
    ]

    prompt = build_prompt("What is first-line therapy?", chunks)

    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "who.pdf" in prompt
    assert "page 3" in prompt
    assert "page 4" in prompt
    assert "What is first-line therapy?" in prompt


def test_build_prompt_instructs_citation_and_context_only_answering():
    prompt = build_prompt("q", [])

    assert "cite" in prompt.lower()
    assert "only" in prompt.lower()


class _FakeOllamaClient:
    def __init__(self, response_text="Metformin is first-line therapy [1]."):
        self._response_text = response_text
        self.last_call = None

    def generate(self, model, prompt):
        self.last_call = {"model": model, "prompt": prompt}
        return {"response": self._response_text}


def test_llm_client_generate_returns_response_text_and_passes_through_args():
    fake = _FakeOllamaClient()
    client = LLMClient(model="llama3", client=fake)

    result = client.generate("some prompt")

    assert result == "Metformin is first-line therapy [1]."
    assert fake.last_call == {"model": "llama3", "prompt": "some prompt"}


@pytest.mark.integration
def test_llm_client_live_generation_is_grounded_in_context():
    chunks = [
        _chunk(
            "Fixtureville was founded in 1900 for testing purposes only.",
            "sample.pdf",
            1,
        )
    ]
    prompt = build_prompt("When was Fixtureville founded?", chunks)

    client = LLMClient()
    answer = client.generate(prompt)

    assert "1900" in answer

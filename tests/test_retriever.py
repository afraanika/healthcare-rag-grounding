from pathlib import Path

import pytest

from app.chunker import chunk_pages
from app.embedder import EmbeddingClient
from app.ingest import extract_pages
from app.retriever import Retriever
from app.vectorstore import VectorStore

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingClient()


def _indexed_retriever(tmp_path, embedder):
    pages = extract_pages(FIXTURE)
    chunks = chunk_pages(pages, source_file="sample.pdf", chunk_size=200, chunk_overlap=20)
    vectors = embedder.embed([c.text for c in chunks])
    store = VectorStore(persist_dir=str(tmp_path))
    store.add_chunks(chunks, vectors)
    return Retriever(embedder=embedder, vectorstore=store)


def test_retrieve_finds_correct_page_for_founding_question(tmp_path, embedder):
    retriever = _indexed_retriever(tmp_path, embedder)

    results = retriever.retrieve("When was Fixtureville founded?", top_k=2)

    assert len(results) > 0
    assert results[0].page_number == 1
    assert results[0].source_file == "sample.pdf"


def test_retrieve_finds_correct_page_for_population_question(tmp_path, embedder):
    retriever = _indexed_retriever(tmp_path, embedder)

    results = retriever.retrieve("What is the population of Fixtureville?", top_k=2)

    assert results[0].page_number == 2


def test_retrieve_respects_top_k(tmp_path, embedder):
    retriever = _indexed_retriever(tmp_path, embedder)

    results = retriever.retrieve("Fixtureville", top_k=1)

    assert len(results) == 1

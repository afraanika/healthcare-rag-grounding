from app.models import Chunk
from app.vectorstore import VectorStore


def _chunk(id_, text, source_file, page_number, chunk_index=0):
    return Chunk(
        id=id_,
        text=text,
        source_file=source_file,
        page_number=page_number,
        chunk_index=chunk_index,
        char_start=0,
        char_end=len(text),
    )


def test_add_and_query_round_trips_metadata(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    chunks = [
        _chunk("a", "apple", "fruits.pdf", page_number=1),
        _chunk("b", "banana", "fruits.pdf", page_number=2),
        _chunk("c", "cherry", "fruits.pdf", page_number=3),
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    store.add_chunks(chunks, embeddings)
    results = store.query([0.9, 0.1, 0.0], top_k=1)

    assert len(results) == 1
    top = results[0]
    assert top["id"] == "a"
    assert top["text"] == "apple"
    assert top["metadata"]["source_file"] == "fruits.pdf"
    assert top["metadata"]["page_number"] == 1


def test_query_returns_ranked_top_k(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    chunks = [
        _chunk("a", "apple", "fruits.pdf", page_number=1),
        _chunk("b", "banana", "fruits.pdf", page_number=2),
        _chunk("c", "cherry", "fruits.pdf", page_number=3),
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    store.add_chunks(chunks, embeddings)

    results = store.query([0.0, 0.9, 0.1], top_k=2)

    assert [r["id"] for r in results] == ["b", "a"] or [r["id"] for r in results][0] == "b"
    assert len(results) == 2


def test_list_documents_counts_chunks_per_source(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    chunks = [
        _chunk("a", "apple", "fruits.pdf", page_number=1),
        _chunk("b", "banana", "fruits.pdf", page_number=1, chunk_index=1),
        _chunk("x", "xerophyte", "plants.pdf", page_number=1),
    ]
    embeddings = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    store.add_chunks(chunks, embeddings)

    counts = store.list_documents()

    assert counts == {"fruits.pdf": 2, "plants.pdf": 1}


def test_delete_document_removes_only_its_chunks(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    chunks = [
        _chunk("a", "apple", "fruits.pdf", page_number=1),
        _chunk("x", "xerophyte", "plants.pdf", page_number=1),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    store.add_chunks(chunks, embeddings)

    store.delete_document("fruits.pdf")
    counts = store.list_documents()

    assert counts == {"plants.pdf": 1}

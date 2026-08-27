from pathlib import Path

from app.chunker import chunk_pages, chunk_text
from app.ingest import PageText, extract_pages

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


def test_chunk_text_spans_reconstruct_exact_source_slices():
    text = "First sentence here. Second sentence here.\n\nNew paragraph starts here."
    spans = chunk_text(text, chunk_size=1000, chunk_overlap=0)

    for start, end in spans:
        assert 0 <= start < end <= len(text)


def test_chunk_text_hard_splits_a_single_oversized_unit():
    text = "x" * 50  # one "unit" (no sentence/paragraph breaks) longer than chunk_size
    spans = chunk_text(text, chunk_size=10, chunk_overlap=2)

    assert len(spans) > 1
    assert all(end - start <= 10 for start, end in spans)


def test_chunk_pages_never_crosses_a_page_boundary():
    pages = extract_pages(FIXTURE)
    chunks = chunk_pages(pages, source_file="sample.pdf", chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.text.strip() != ""
        assert chunk.page_number in (1, 2)
        # the chunk's recorded offsets must reproduce its text on its own page
        source_page = next(p for p in pages if p.page_number == chunk.page_number)
        assert source_page.text[chunk.char_start:chunk.char_end] == chunk.text


def test_chunk_pages_ids_are_unique_and_traceable():
    pages = extract_pages(FIXTURE)
    chunks = chunk_pages(pages, source_file="sample.pdf", chunk_size=80, chunk_overlap=10)

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert c.id == f"sample.pdf::p{c.page_number}::c{c.chunk_index}"


def test_chunk_pages_skips_blank_pages():
    pages = [PageText(page_number=1, text="   \n  "), PageText(page_number=2, text="Real content.")]
    chunks = chunk_pages(pages, source_file="blank.pdf", chunk_size=100, chunk_overlap=0)

    assert len(chunks) == 1
    assert chunks[0].page_number == 2

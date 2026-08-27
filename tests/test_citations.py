from app.citations import attach_citations
from app.retriever import RetrievedChunk


def _chunk(id_, text, source_file, page_number):
    return RetrievedChunk(
        id=id_, text=text, source_file=source_file, page_number=page_number, chunk_index=0, distance=0.1
    )


def test_attach_citations_maps_markers_to_correct_chunks():
    chunks = [
        _chunk("c1", "Metformin is first-line therapy.", "who.pdf", 3),
        _chunk("c2", "Insulin is used later.", "who.pdf", 4),
    ]
    answer = "Metformin is first-line [1]. Insulin comes later [2]."

    result = attach_citations(answer, chunks)

    assert result.answer == answer
    assert len(result.citations) == 2
    assert result.citations[0].marker == 1
    assert result.citations[0].source_file == "who.pdf"
    assert result.citations[0].page_number == 3
    assert result.citations[0].chunk_id == "c1"
    assert result.citations[1].marker == 2
    assert result.citations[1].page_number == 4


def test_attach_citations_handles_missing_marker_for_some_chunks():
    chunks = [
        _chunk("c1", "text one", "a.pdf", 1),
        _chunk("c2", "text two", "a.pdf", 2),
    ]
    answer = "Only the first passage is cited [1]."

    result = attach_citations(answer, chunks)

    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"


def test_attach_citations_out_of_range_marker_falls_back_to_all_chunks():
    # An out-of-range marker means every parsed marker was invalid, so no
    # valid citations were extracted — this degrades the same way as the
    # no-markers-at-all case: cite every retrieved chunk rather than none.
    chunks = [_chunk("c1", "only one chunk", "a.pdf", 1)]
    answer = "This cites a nonexistent passage [5]."

    result = attach_citations(answer, chunks)

    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"


def test_attach_citations_valid_marker_alongside_out_of_range_one():
    # When at least one marker is valid, out-of-range markers are simply
    # dropped rather than triggering the all-chunks fallback.
    chunks = [
        _chunk("c1", "text one", "a.pdf", 1),
        _chunk("c2", "text two", "a.pdf", 2),
    ]
    answer = "Cites a real passage [1] and a bogus one [9]."

    result = attach_citations(answer, chunks)

    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"


def test_attach_citations_deduplicates_repeated_markers():
    chunks = [_chunk("c1", "text", "a.pdf", 1)]
    answer = "Cited twice [1] and again [1]."

    result = attach_citations(answer, chunks)

    assert len(result.citations) == 1


def test_attach_citations_falls_back_to_all_chunks_when_no_markers_present():
    chunks = [
        _chunk("c1", "text one", "a.pdf", 1),
        _chunk("c2", "text two", "a.pdf", 2),
    ]
    answer = "This answer has no citation markers at all."

    result = attach_citations(answer, chunks)

    assert len(result.citations) == 2
    assert [c.chunk_id for c in result.citations] == ["c1", "c2"]


def test_attach_citations_with_no_chunks_and_no_markers_returns_empty():
    result = attach_citations("No context was available.", [])

    assert result.citations == []

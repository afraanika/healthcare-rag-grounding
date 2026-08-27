import re

from app.models import AnswerResponse, Citation
from app.retriever import RetrievedChunk

_SNIPPET_LEN = 240
_MARKER_PATTERN = re.compile(r"\[(\d+)\]")


def _citation_for(marker: int, chunk: RetrievedChunk) -> Citation:
    return Citation(
        marker=marker,
        source_file=chunk.source_file,
        page_number=chunk.page_number,
        chunk_id=chunk.id,
        snippet=chunk.text[:_SNIPPET_LEN],
    )


def attach_citations(answer_text: str, chunks: list[RetrievedChunk]) -> AnswerResponse:
    """Parse [n] markers out of the LLM's answer and map each to the chunk
    metadata it refers to (1-indexed, matching retrieval order). Markers
    that are out of range are dropped rather than raising, since they
    reflect the LLM miscounting rather than a caller error. If the answer
    contains no markers at all, fall back to citing every retrieved chunk
    so the answer stays traceable to its source context."""
    seen_markers: set[int] = set()
    citations: list[Citation] = []
    for marker_str in _MARKER_PATTERN.findall(answer_text):
        marker = int(marker_str)
        if marker in seen_markers:
            continue
        idx = marker - 1
        if idx < 0 or idx >= len(chunks):
            continue
        seen_markers.add(marker)
        citations.append(_citation_for(marker, chunks[idx]))

    if not citations and chunks:
        citations = [_citation_for(i, c) for i, c in enumerate(chunks, start=1)]

    return AnswerResponse(answer=answer_text, citations=citations)

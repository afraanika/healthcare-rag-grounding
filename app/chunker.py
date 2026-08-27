import re

from app.ingest import PageText
from app.models import Chunk

# Split on sentence-ending punctuation followed by whitespace, or a
# paragraph break. Because these are zero-width lookarounds / literal
# separators, slicing `text[last:m.start()]` always yields exact,
# non-overlapping units with correct offsets into the original text.
_SPLIT_POINT = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


def _split_units(text: str) -> list[tuple[str, int, int]]:
    units: list[tuple[str, int, int]] = []
    last = 0
    for m in _SPLIT_POINT.finditer(text):
        end = m.start()
        if end > last:
            units.append((text[last:end], last, end))
        last = m.end()
    if last < len(text):
        units.append((text[last:], last, len(text)))
    return units


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """Split `text` into (start, end) char-offset spans, packing sentence/
    paragraph units greedily up to chunk_size chars with chunk_overlap chars
    retained between consecutive chunks. A single unit longer than
    chunk_size is hard-split by character count."""
    units = _split_units(text)
    if not units:
        return []

    spans: list[tuple[int, int]] = []
    i = 0
    n = len(units)
    while i < n:
        start = units[i][1]
        end = units[i][2]
        j = i
        while j + 1 < n and units[j + 1][2] - start <= chunk_size:
            j += 1
            end = units[j][2]

        if end - start > chunk_size:
            step = max(chunk_size - chunk_overlap, 1)
            pos = start
            while pos < end:
                piece_end = min(pos + chunk_size, end)
                spans.append((pos, piece_end))
                pos += step
            i = j + 1
            continue

        spans.append((start, end))

        if j + 1 >= n:
            break

        if chunk_overlap > 0:
            target = end - chunk_overlap
            k = j
            while k > i and units[k][1] > target:
                k -= 1
            i = max(k, i + 1)
        else:
            i = j + 1

    return spans


def chunk_pages(
    pages: list[PageText],
    source_file: str,
    chunk_size: int = 650,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Chunk each page independently so no chunk ever spans a page
    boundary, keeping every chunk traceable to exactly one page."""
    chunks: list[Chunk] = []
    chunk_index = 0
    for page in pages:
        if not page.text.strip():
            continue
        for start, end in chunk_text(page.text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    id=f"{source_file}::p{page.page_number}::c{chunk_index}",
                    text=page.text[start:end],
                    source_file=source_file,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    char_start=start,
                    char_end=end,
                )
            )
            chunk_index += 1
    return chunks

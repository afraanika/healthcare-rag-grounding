from dataclasses import dataclass
from pathlib import Path

import pypdf


@dataclass
class PageText:
    page_number: int
    text: str


def extract_pages(path: str | Path) -> list[PageText]:
    reader = pypdf.PdfReader(str(path))
    return [
        PageText(page_number=i, text=page.extract_text() or "")
        for i, page in enumerate(reader.pages, start=1)
    ]

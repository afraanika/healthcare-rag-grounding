from pathlib import Path

from app.ingest import extract_pages

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


def test_extract_pages_returns_two_pages_in_order():
    pages = extract_pages(FIXTURE)

    assert len(pages) == 2
    assert [p.page_number for p in pages] == [1, 2]


def test_extract_pages_contains_expected_text():
    pages = extract_pages(FIXTURE)

    assert "Fixtureville" in pages[0].text
    assert "founded in" in pages[0].text
    assert "population of Fixtureville" in pages[1].text
    assert "Ada Lovelace" in pages[1].text

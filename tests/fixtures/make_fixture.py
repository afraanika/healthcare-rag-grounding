"""Regenerates tests/fixtures/sample.pdf. Run manually with:
    .venv/bin/python tests/fixtures/make_fixture.py
Not executed by the test suite itself; the generated PDF is committed
so tests don't need fpdf2 installed to run.
"""

from pathlib import Path

from fpdf import FPDF

PAGE_1 = (
    "The capital of Testland is Fixtureville. Fixtureville was founded in "
    "1900 for testing purposes only.\n\n"
    "This paragraph is the second paragraph on page one, added to test "
    "chunk splitting across paragraphs within a single page."
)

PAGE_2 = (
    "Page two discusses the population of Fixtureville, which is exactly "
    "42 according to official records. The mayor of Fixtureville is Ada "
    "Lovelace, appointed in 2020."
)


def main() -> None:
    pdf = FPDF()
    for page_text in (PAGE_1, PAGE_2):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, page_text)

    out_path = Path(__file__).parent / "sample.pdf"
    pdf.output(str(out_path))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

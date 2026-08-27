"""Batch-ingest one or more PDFs into the vector store, without going
through the API.

Usage:
    .venv/bin/python scripts/ingest_cli.py data/raw/*.pdf
"""

import sys

from app.pipeline import RagPipeline


def main(paths: list[str]) -> None:
    if not paths:
        print("Usage: python scripts/ingest_cli.py <pdf> [<pdf> ...]")
        raise SystemExit(1)

    pipeline = RagPipeline()
    for path in paths:
        result = pipeline.ingest_document(path)
        print(f"Ingested {result['document_id']}: {result['num_pages']} pages, {result['num_chunks']} chunks")


if __name__ == "__main__":
    main(sys.argv[1:])

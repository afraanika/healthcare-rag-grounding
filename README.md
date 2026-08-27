# healthcare-rag-grounding

A Retrieval-Augmented Generation (RAG) pipeline that answers questions about
healthcare documents (clinical guidelines, fact sheets, medical literature)
using a **fully local, open-source stack** — no hosted LLM APIs. Every answer
comes with inline citations back to the exact source PDF page it was drawn
from.

**Status:** working prototype. The full ingest → retrieve → generate → cite
pipeline is implemented, tested, and verified end-to-end against real CDC/NCHS
PDFs with a real local LLM. It is not yet production-hardened — see
[Limitations](#grounding--citations) below.

## Project structure

```
app/
├── config.py        # Settings (.env-driven)
├── models.py        # Pydantic schemas: Chunk, Citation, AnswerResponse, QueryRequest
├── ingest.py        # PDF -> per-page text (pypdf)
├── chunker.py       # Page-boundary-aware chunking
├── embedder.py      # sentence-transformers wrapper
├── vectorstore.py   # Chroma wrapper (add/query/list/delete)
├── retriever.py     # Embeds a query and searches the vector store
├── generator.py     # Prompt building + Ollama client
├── citations.py     # Maps [n] markers in the answer back to source chunks
├── pipeline.py       # Glue: RagPipeline used by both the API and the CLI
└── api.py            # FastAPI app: /health, /ingest, /query, /documents

scripts/ingest_cli.py   # Batch ingestion without the API
data/raw/               # Sample PDFs + SOURCES.md (provenance/license)
tests/                  # pytest suite (unit + integration-marked live-LLM tests)
```

## Stack

| Concern | Choice |
|---|---|
| API | FastAPI + Uvicorn |
| PDF parsing | `pypdf` |
| Chunking | Custom page-boundary-aware, sentence/paragraph-preserving splitter |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector store | Chroma (local, persistent) |
| LLM | Ollama (local), default model `llama3` |

## Setup

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally, with a model pulled:
  ```bash
  ollama pull llama3
  ```
  (To use a different model, set `LLM_MODEL` in `.env` and pull it instead.)

### 2. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Configure

```bash
cp .env.example .env
# edit .env if you want to change the model, chunk size, etc.
```

## Running

Start the API:

```bash
uvicorn app.api:app --reload
```

Then open http://localhost:8000/docs for the interactive Swagger UI.

### API endpoints

| Method & path | Purpose |
|---|---|
| `GET /health` | Checks that Chroma and the Ollama daemon are both reachable |
| `POST /ingest` | Upload a PDF (multipart); chunks, embeds, and indexes it |
| `GET /documents` | Lists indexed documents with their chunk counts |
| `DELETE /documents/{document_id}` | Removes a document's chunks from the index |
| `POST /query` | `{"question": str, "top_k": int}` → an answer with citations |

### Ingest documents

Either through the API:

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/raw/cdc_national_diabetes_fact_sheet_2003.pdf;type=application/pdf"
```

Or in bulk via the CLI (no server needed):

```bash
.venv/bin/python scripts/ingest_cli.py data/raw/*.pdf
```

### Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What percentage of US adults had hypertension between 2017 and 2018?", "top_k": 4}'
```

Example response:

```json
{
  "answer": "According to the data, 45.4% of US adults had hypertension between 2017 and 2018 [1].",
  "citations": [
    {
      "marker": 1,
      "source_file": "nchs_hypertension_prevalence_databrief_364.pdf",
      "page_number": 1,
      "chunk_id": "nchs_hypertension_prevalence_databrief_364.pdf::p1::c2",
      "snippet": "In survey period 2017-2018, the prevalence of age-adjusted hypertension was 45.4% among adults..."
    }
  ]
}
```

## Sample data

`data/raw/` ships with three small, public-domain CDC/NCHS PDFs (hypertension
and diabetes fact sheets) for prototyping — see `data/raw/SOURCES.md` for
provenance and licensing. They are for testing only; re-verify licensing
before any production use.

## Grounding & citations

Every chunk indexed into the vector store carries its source filename, page
number, and character offsets as metadata. Chunking never merges text across
a page boundary, so every chunk is traceable to exactly one page. At query
time, the LLM is prompted to cite the numbered context passage(s) supporting
each part of its answer using `[n]` markers; these are parsed and mapped back
to the originating chunk's file/page/snippet in the API response.

**Current limitation:** this citation mechanism confirms *which chunk the LLM
says it used* — it does not independently verify that the answer text is
actually, factually supported by that chunk (i.e. no hallucination-detection
pass yet). Treat citations as "here's where to check," and manually verify
against the source PDF for anything safety-critical. A verification/confidence
layer that checks generated claims against cited text is a natural next step.

## Testing

```bash
pytest                        # unit tests only (fast, no external services)
pytest -m integration         # includes tests that call the live Ollama daemon
```

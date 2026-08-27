from pydantic import BaseModel


class Chunk(BaseModel):
    id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int


class Citation(BaseModel):
    marker: int
    source_file: str
    page_number: int
    chunk_id: str
    snippet: str


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

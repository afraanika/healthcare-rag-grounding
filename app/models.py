from pydantic import BaseModel


class Chunk(BaseModel):
    id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int

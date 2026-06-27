from sqlmodel import SQLModel, Field
from datetime import datetime

class ChunkMetadata(SQLModel):
    source_file: str
    page_number: int | None
    char_start: int
    char_end: int

class DocumentChunk(SQLModel):
    chunk_id: str
    document_id: str
    collection: str
    text: str
    chunk_index: int
    metadata: ChunkMetadata

class RetrievedChunk(SQLModel):
    chunk: DocumentChunk
    score: float
    rank: int

class Document(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    document_id: str = Field(unique=True)
    collection: str
    filename: str
    ingested_at: datetime = Field(default_factory=datetime.now)
    chunk_count: int
    embedding_model: str

class Answer(SQLModel):
    question: str
    answer: str
    sources: list[RetrievedChunk]
    model: str
    input_tokens: int
    output_tokens: int
    collection: str
    retrieved_at: datetime = Field(default_factory=datetime.now)
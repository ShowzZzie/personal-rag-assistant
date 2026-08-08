# PHASE 2: pipeline.py — ingest: parse PDF → chunk → embed → store → record metadata

import pathlib
import uuid
from typing import TYPE_CHECKING

from pypdf import PdfReader
from sqlmodel import Session, SQLModel, create_engine

from rag.chunker import chunker_recursive
from rag.schemas import Document
from rag.store import add_chunks

if TYPE_CHECKING:
    from sqlalchemy import Engine

DEFAULT_CHUNK_SIZE = 500 # placeholder pre-config
DEFAULT_CHUNK_OVERLAP = 50 # placeholder pre-config
EMBEDDING_MODEL_NAME = "text-embedding-3-small" # placeholder pre-config

sqlite_db_filename = "database.db"
sqlite_db_uri = f"sqlite:///data/{sqlite_db_filename}"
sqlite_engine = create_engine(sqlite_db_uri)

SQLModel.metadata.create_all(sqlite_engine)

def ingest(
    file: str,
    collection: str,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    engine: Engine = sqlite_engine
    ) -> Document:
    reader = PdfReader(file)
    page_list: list[str] = []

    for page in reader.pages:
        page_list.append(page.extract_text())

    doc_onestring = "\n\n".join(page_list)
    file_name = pathlib.Path(file).name
    doc_id = str(uuid.uuid4())

    chunking_result = chunker_recursive(
        document=doc_onestring,
        filename=file_name,
        document_id=doc_id,
        collection=collection,
        size=size,
        overlap=overlap,
        recursive_order=["\n\n", "\n", "."]
    )

    add_chunks(chunking_result)
    
    # metadata sqlite
    with Session(engine) as session:
        document = Document(
            document_id=doc_id,
            collection=collection,
            filename=file_name,
            chunk_count=len(chunking_result),
            embedding_model=EMBEDDING_MODEL_NAME
        )
        session.add(document)
        session.commit()
        session.refresh(document)
    
    return document
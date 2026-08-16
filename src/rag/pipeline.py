# PHASE 2: pipeline.py — ingest: parse PDF → chunk → embed → store → record metadata

import pathlib
import uuid
from typing import TYPE_CHECKING

from pypdf import PdfReader
from sqlmodel import Session, SQLModel, create_engine, select

from rag.chunker import chunker_recursive
from rag.schemas import ChunkMetadata, Document, DocumentChunk
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

def join_small_chunks(
    chunks: list[DocumentChunk],
    size: int,
    overlap: int
) -> list[DocumentChunk]:
    new_chunks: list[DocumentChunk] = []
    current_group: list[DocumentChunk] = []
    temp_chunk = ""
    chunk_index = 0

    document_id = chunks[0].document_id
    collection = chunks[0].collection
    source_file=chunks[0].chunk_metadata.source_file
    page_number=chunks[0].chunk_metadata.page_number

    def flush(group: list[DocumentChunk], text: str, index: int) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=uuid.uuid4(),
            document_id=document_id,
            collection=collection,
            text=text,
            chunk_index=index,
            chunk_metadata=ChunkMetadata(
                source_file=source_file,
                page_number=page_number,
                char_start=group[0].chunk_metadata.char_start,
                char_end=group[-1].chunk_metadata.char_end,
            ),
        )

    while len(chunks) > 0:
        if len(temp_chunk)+len(chunks[0].text) <= size:
            lookup_chunk = chunks.pop(0)
            current_group.append(lookup_chunk)
            temp_chunk = (temp_chunk + " " + lookup_chunk.text).strip()
        else:
            new_chunks.append(flush(current_group, temp_chunk, chunk_index))
            chunk_index+=1

            seed: list[DocumentChunk] = []
            seed_len = 0
            for frag in reversed(current_group):
                if seed_len + len(frag.text) > overlap and seed:
                    break
                seed.insert(0, frag)
                seed_len += len(frag.text)

            current_group = seed
            temp_chunk = " ".join(f.text for f in seed).strip()

    if temp_chunk:
        new_chunks.append(flush(current_group, temp_chunk, chunk_index))

    return new_chunks

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

    jsc_chunking_result = join_small_chunks(chunking_result, size, overlap)
    add_chunks(jsc_chunking_result)
    
    # metadata sqlite
    with Session(engine) as session:
        document = Document(
            document_id=doc_id,
            collection=collection,
            filename=file_name,
            chunk_count=len(jsc_chunking_result),
            embedding_model=EMBEDDING_MODEL_NAME
        )
        session.add(document)
        session.commit()
        session.refresh(document)
    
    return document


def query_by_collection(
    collection: str,
    engine: Engine = sqlite_engine
    ) -> list[Document]:
    
    with Session(engine) as session:
        stmnt = select(Document).where(Document.collection==collection)
        rslt = session.exec(stmnt).all()
        return list(rslt)


def query_by_id(
    doc_id: int,
    engine: Engine = sqlite_engine
    ) -> Document:

    with Session(engine) as session:
        stmnt = select(Document).where(Document.id==doc_id)
        rslt = session.exec(stmnt).first()
        if rslt is None:
            raise ValueError("File not found")
        return rslt


def get_collections_counts(engine: Engine = sqlite_engine) -> dict[str, int]:
    with Session(engine) as session:
        stmnt = select(Document.collection).distinct()
        rslt = list(session.exec(stmnt).all())
        result_dict = {}
        for col in rslt:
            stmnt_count = select(Document).where(Document.collection==col)
            rslt_count = len(session.exec(stmnt_count).all())
            result_dict[col] = rslt_count

        return result_dict
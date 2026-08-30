# PHASE 2: pipeline.py — ingest: parse PDF → chunk → embed → store → record metadata

import pathlib
import re
import uuid
from typing import TYPE_CHECKING

from pypdf import PdfReader
from sqlmodel import Session, SQLModel, create_engine, select

from rag.chunker import chunker_fixed_size, chunker_recursive
from rag.config import settings
from rag.schemas import ChunkMetadata, Document, DocumentChunk
from rag.store import add_chunks

if TYPE_CHECKING:
    from sqlalchemy import Engine

CITATION_PATTERN = re.compile(r"\d{4};\s*\d+\s*\(")

sqlite_engine = create_engine(settings.sqlite_db_uri)

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
            if len(chunks[0].text) > size:
                cnk = chunks.pop(0)
                cfs_result = chunker_fixed_size(
                    document=cnk.text,
                    filename=cnk.chunk_metadata.source_file,
                    document_id=cnk.document_id,
                    collection=cnk.collection,
                    size=size,
                    overlap=overlap
                )
                for chunk in cfs_result:
                    chunk.chunk_index=chunk_index
                    chunk_index+=1
                new_chunks.extend(cfs_result)
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

            if chunks and seed_len + len(chunks[0].text) > size:
                seed = []
                seed_len = 0

            current_group = seed
            temp_chunk = " ".join(f.text for f in seed).strip()

    if temp_chunk:
        new_chunks.append(flush(current_group, temp_chunk, chunk_index))

    return new_chunks

def is_reference_chunk(text: str, threshold: float = settings.reference_citation_threshold) -> bool:
    if not text:
        return False
    return len(CITATION_PATTERN.findall(text)) / (len(text) / 100) > threshold

def ingest(
    file: str,
    collection: str,
    size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
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
        recursive_order=["\n\n", "\n", ". "]
    )

    jsc_chunking_result = join_small_chunks(chunking_result, size, overlap)

    citation_removal_result = [c for c in jsc_chunking_result if not is_reference_chunk(c.text)]

    add_chunks(citation_removal_result)
    
    # metadata sqlite
    with Session(engine) as session:
        document = Document(
            document_id=doc_id,
            collection=collection,
            filename=file_name,
            chunk_count=len(citation_removal_result),
            embedding_model=settings.embedding_model
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
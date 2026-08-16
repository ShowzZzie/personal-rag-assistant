from sqlmodel import create_engine, SQLModel, Session, select
import rag.pipeline, rag.embedder, rag.store
from types import SimpleNamespace as sn
from rag.schemas import Document, DocumentChunk, ChunkMetadata
import chromadb
import uuid

TEST_DATABASE = "sqlite://"
test_engine = create_engine(TEST_DATABASE)
SQLModel.metadata.create_all(test_engine)

def _frag(text: str, index: int, start: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=uuid.uuid4(),
        document_id="jsc_doc",
        collection="jsc_collection",
        text=text,
        chunk_index=index,
        chunk_metadata=ChunkMetadata(
            source_file="jsc.pdf",
            page_number=None,
            char_start=start,
            char_end=start + len(text) - 1,
        ),
    )

def test_ingest(monkeypatch):
    # mock get_embedding like test_store.py
    def mock_embed(input: str, model: str):
        resp = sn(data=[sn(embedding=[-0.1, 0.23, -0.004, 1.2])])
        return resp
    monkeypatch.setattr(rag.embedder.client.embeddings, "create", mock_embed)
    monkeypatch.setattr(rag.store, "chroma_persistent_client", chromadb.Client())
    
    # call rag.pipeline.ingest(file, collection, engine=test_engine)
    ingest_result = rag.pipeline.ingest("tests/data/czarnogora_notatki.pdf", "test_collection", engine=test_engine)
    
    # assert doc.chunk_count > 0
    assert ingest_result.chunk_count > 0
    
    # assert Chroma collection has matching chunk count
    collection = rag.store.chroma_persistent_client.get_or_create_collection("test_collection")
    assert collection.count() == ingest_result.chunk_count

    
    # query test_engine's Session for the Document row, assert it exists
    with Session(test_engine) as session:
        stmnt = select(Document).where(Document.document_id == ingest_result.document_id)
        rslt = session.exec(stmnt).first()
        assert rslt is not None

def test_join_small_chunks_merges_and_overlaps():
    texts = [
        "alpha alpha alpha",
        "bravo bravo bravo",
        "charlie charlie",
        "delta delta delta",
        "echo echo echo",
        "foxtrot foxtrot",
    ]
    frags = []
    pos = 0
    for i, t in enumerate(texts):
        frags.append(_frag(t, i, pos))
        pos += len(t) + 1

    merged = rag.pipeline.join_small_chunks(frags, size=40, overlap=15)

    assert len(merged) > 1
    assert len(merged) < len(texts)
    assert all(len(c.text) <= 40 for c in merged)
    assert [c.chunk_index for c in merged] == list(range(len(merged)))
    assert all(c.document_id == "jsc_doc" for c in merged)
    assert all(c.collection == "jsc_collection" for c in merged)

    for a, b in zip(merged, merged[1:]):
        assert b.text.split()[0] in a.text
from sqlmodel import create_engine, SQLModel, Session, select
import rag.pipeline, rag.embedder, rag.store
from types import SimpleNamespace as sn
from rag.schemas import Document

TEST_DATABASE = "sqlite://"
test_engine = create_engine(TEST_DATABASE)
SQLModel.metadata.create_all(test_engine)

def test_ingest(monkeypatch):
    # mock get_embedding like test_store.py
    def mock_embed(input: str, model: str):
        resp = sn(data=[sn(embedding=[-0.1, 0.23, -0.004, 1.2])])
        return resp
    monkeypatch.setattr(rag.embedder.client.embeddings, "create", mock_embed)
    
    # call rag.pipeline.ingest(file, collection, engine=test_engine)
    ingest_result = rag.pipeline.ingest("tests/data/czarnogora_notatki.pdf", "test_collection", engine=test_engine)
    
    # assert doc.chunk_count > 0
    assert ingest_result.chunk_count > 0
    
    # assert Chroma collection has matching chunk count
    collection = rag.store.chroma_client.get_or_create_collection("test_collection")
    assert collection.count() == ingest_result.chunk_count

    
    # query test_engine's Session for the Document row, assert it exists
    with Session(test_engine) as session:
        stmnt = select(Document).where(Document.document_id == ingest_result.document_id)
        rslt = session.exec(stmnt).first()
        assert rslt is not None
import uuid
from rag.schemas import DocumentChunk, ChunkMetadata
import rag.embedder, rag.store
from types import SimpleNamespace as sn
import pytest

client = rag.store.chroma_client

def test_store(monkeypatch):

    def mock_embed(input: str, model: str):
        resp = sn(data=[sn(embedding=[-0.1, 0.23, -0.004, 1.2])])
        return resp

    document_chunks = [
        DocumentChunk(chunk_id=str(uuid.uuid4()), document_id="t123", collection="t_collection", text="Random text", chunk_index=0, chunk_metadata=ChunkMetadata(source_file="xyz", page_number=1, char_start=1, char_end=100)),
        DocumentChunk(chunk_id=str(uuid.uuid4()), document_id="t456", collection="t_collection", text="Random text 123 ra", chunk_index=1, chunk_metadata=ChunkMetadata(source_file="xyz", page_number=2, char_start=1000, char_end=3456))
    ]
    
    monkeypatch.setattr(rag.embedder.client.embeddings, "create", mock_embed)

    rag.store.add_chunks(document_chunks)
    collection = client.get_or_create_collection(name="t_collection")
    recs = collection.get()
    
    assert len(recs["ids"]) == 2
    assert len(recs["documents"]) == 2
    assert len(recs["metadatas"]) == 2
    assert recs["documents"] == ["Random text", "Random text 123 ra"]


def test_add_chunks_empty_list():
    with pytest.raises(IndexError) as err:
        rag.store.add_chunks([])
    assert "empty" in str(err)
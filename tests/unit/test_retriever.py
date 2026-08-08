from types import SimpleNamespace as sn
import rag.embedder, rag.retriever, rag.store
from rag.schemas import DocumentChunk
import uuid, pytest
import chromadb

def test_retrieve(monkeypatch):

    text_to_vec = {
        "text one": [1, 0, 0],
        "text two": [0, 1, 0],
        "text three": [0, 0, 1],
        "test 101": [1, 0, 1],
        "test 110": [1, 1, 0],
        "test 011": [0, 1, 1],
        "test 111": [1, 1, 1]
    }

    def mock_embed(input: str, model: str):
        return sn(data=[sn(embedding=text_to_vec[input])])
    monkeypatch.setattr(rag.embedder.client.embeddings, "create", mock_embed)
    monkeypatch.setattr(rag.store, "chroma_persistent_client", chromadb.Client())

    doc_chunks=[
        DocumentChunk(chunk_id=str(uuid.uuid4()), document_id="t123", collection="ret_collection", text="text one", chunk_index=0, chunk_metadata=rag.store.ChunkMetadata(source_file="xyz", page_number=1, char_start=1, char_end=100)),
        DocumentChunk(chunk_id=str(uuid.uuid4()), document_id="t123", collection="ret_collection", text="text two", chunk_index=1, chunk_metadata=rag.store.ChunkMetadata(source_file="xyz", page_number=1, char_start=101, char_end=200)),
        DocumentChunk(chunk_id=str(uuid.uuid4()), document_id="t123", collection="ret_collection", text="text three", chunk_index=2, chunk_metadata=rag.store.ChunkMetadata(source_file="xyz", page_number=1, char_start=201, char_end=300))
    ]

    rag.store.add_chunks(doc_chunks)

    res = rag.retriever.retrieve("test 110", "ret_collection", 2)
    assert len(res)==2
    assert all(chunk.chunk.text in ["text one", "text two"] for chunk in res)
    assert res[0].rank == 1
    assert res[1].rank == 2
    assert res[0].score == pytest.approx(0.707, abs=0.01)
    assert res[1].score == pytest.approx(0.707, abs=0.01)
import uuid

import chromadb
from chromadb.errors import NotFoundError
from rag.schemas import DocumentChunk, ChunkMetadata
import rag.embedder, rag.store
from types import SimpleNamespace as sn
import pytest

client = chromadb.Client()

def test_add_chunks(monkeypatch):

    def mock_embed(input: str, model: str):
        resp = sn(data=[sn(embedding=[-0.1, 0.23, -0.004, 1.2])])
        return resp

    document_chunks = [
        DocumentChunk(chunk_id=str(uuid.uuid4()), document_id="t123", collection="t_collection", text="Random text", chunk_index=0, chunk_metadata=ChunkMetadata(source_file="xyz", page_number=1, char_start=1, char_end=100)),
        DocumentChunk(chunk_id=str(uuid.uuid4()), document_id="t456", collection="t_collection", text="Random text 123 ra", chunk_index=1, chunk_metadata=ChunkMetadata(source_file="xyz", page_number=2, char_start=1000, char_end=3456))
    ]
    
    monkeypatch.setattr(rag.embedder.client.embeddings, "create", mock_embed)

    rag.store.add_chunks(document_chunks, client)
    collection = client.get_or_create_collection(name="t_collection")
    recs = collection.get()
    
    assert len(recs["ids"]) == 2
    assert len(recs["documents"]) == 2
    assert len(recs["metadatas"]) == 2
    assert recs["documents"] == ["Random text", "Random text 123 ra"]


def test_add_chunks_empty_list():
    with pytest.raises(IndexError) as err:
        rag.store.add_chunks([], client)
    assert "empty" in str(err)



def test_query_by_vector(monkeypatch):
    text_to_vec = {
        "text one": [1, 0, 0],
        "text two": [0, 1, 0],
        "text three": [0, 0, 1],
    }

    def mock_embed(input: str, model: str):
        return sn(data=[sn(embedding=text_to_vec[input])])

    doc_chunks=[
        DocumentChunk(chunk_id=str(uuid.uuid4()), document_id="t123", collection="vec_collection", text="text one", chunk_index=0, chunk_metadata=ChunkMetadata(source_file="xyz", page_number=1, char_start=1, char_end=100)),
        DocumentChunk(chunk_id=str(uuid.uuid4()), document_id="t123", collection="vec_collection", text="text two", chunk_index=1, chunk_metadata=ChunkMetadata(source_file="xyz", page_number=1, char_start=101, char_end=200)),
        DocumentChunk(chunk_id=str(uuid.uuid4()), document_id="t123", collection="vec_collection", text="text three", chunk_index=2, chunk_metadata=ChunkMetadata(source_file="xyz", page_number=1, char_start=201, char_end=300))
    ]

    monkeypatch.setattr(rag.embedder.client.embeddings, "create", mock_embed)
    rag.store.add_chunks(doc_chunks, client)
    result_1 = rag.store.query_by_vector([1,0,0], "vec_collection", 1, client)

    assert len(result_1) == 1
    assert result_1[0].score == pytest.approx(1.0)
    assert result_1[0].chunk.text == "text one"
    assert result_1[0].rank == 1

    result_2 = rag.store.query_by_vector([0,1,0], "vec_collection", 3, client)
    
    assert len(result_2) == 3
    assert result_2[0].score == pytest.approx(1.0)
    assert result_2[0].chunk.text == "text two"
    assert result_2[0].rank == 1
    assert all(r_chunk.score == pytest.approx(0.0) for r_chunk in result_2[1:])
    assert all(r_chunk.rank in [2, 3] for r_chunk in result_2[1:])
    assert all(r_chunk.chunk.text in ["text one", "text three"] for r_chunk in result_2[1:])


def test_successful_delete_collection(monkeypatch):
    def mock_embed(input: str, model: str):
        resp = sn(data=[sn(embedding=[-0.1, 0.23, -0.004, 1.2])])
        return resp

    document_chunks = [
        DocumentChunk(chunk_id=str(uuid.uuid4()), document_id="t123", collection="t_collection", text="Random text", chunk_index=0, chunk_metadata=ChunkMetadata(source_file="xyz", page_number=1, char_start=1, char_end=100)),
        DocumentChunk(chunk_id=str(uuid.uuid4()), document_id="t456", collection="t_collection", text="Random text 123 ra", chunk_index=1, chunk_metadata=ChunkMetadata(source_file="xyz", page_number=2, char_start=1000, char_end=3456))
    ]
    
    monkeypatch.setattr(rag.embedder.client.embeddings, "create", mock_embed)

    rag.store.add_chunks(document_chunks, client)
    collection = client.get_or_create_collection(name="t_collection")
    assert collection

    rag.store.delete_collection("t_collection", client)

    with pytest.raises(NotFoundError) as e:
        client.get_collection("t_collection")
    assert "does not exist" in str(e.value)

def test_failed_delete_collection():
    with pytest.raises(ValueError) as e:
        rag.store.delete_collection("t_collection", client)
    assert "DELETE FAILED" in str(e.value)
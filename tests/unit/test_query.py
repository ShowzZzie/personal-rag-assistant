import uuid
from rag.schemas import Answer, ChunkMetadata, DocumentChunk
import rag.embedder
import rag.store
import rag.synthesizer
import rag.query
from types import SimpleNamespace as sn

def test_query(monkeypatch):

    # mock embed
    def mock_embed(input: str, model: str):
        resp = sn(data=[sn(embedding=[-0.1, 0.23, -0.004, 1.2])])
        return resp
    monkeypatch.setattr(rag.embedder.client.embeddings, "create", mock_embed)

    # mock Anthropic client call
    def mock_anthropic_create(max_tokens: int, messages: list[dict[str,str]], model: str):
        return sn(content=[sn(type="text", text="query test text")], usage=sn(input_tokens=123, output_tokens=456))
    monkeypatch.setattr(rag.synthesizer.client.messages, "create", mock_anthropic_create)

    doc_chunks = [
        DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id="q_t_123",
            collection="query_test_collection",
            text="text one",
            chunk_index=0,
            chunk_metadata=ChunkMetadata(source_file="q_test.pdf", page_number=1, char_start=0, char_end=8)
        ),
        DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id="q_t_123",
            collection="query_test_collection",
            text="text two",
            chunk_index=1,
            chunk_metadata=ChunkMetadata(source_file="q_test.pdf", page_number=1, char_start=9, char_end=17)
        ),
    ]
    rag.store.add_chunks(doc_chunks)
    result = rag.query.query("test query", "query_test_collection", 2)

    assert isinstance(result, Answer)
    assert result.question == "test query"
    assert result.answer == "query test text"
    assert all(chunk.chunk.text in ["text one", "text two"] for chunk in result.sources)
    assert result.model == "claude-haiku-4-5"
    assert result.input_tokens == 123
    assert result.output_tokens == 456
    assert result.collection == "query_test_collection"
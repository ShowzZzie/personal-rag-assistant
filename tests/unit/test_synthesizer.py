from types import SimpleNamespace as sn
from rag.schemas import ChunkMetadata, DocumentChunk, RetrievedChunk
import uuid
import rag.synthesizer


def test_synthesize(monkeypatch):

    def mock_anthropic_create(max_tokens: int, messages: list[dict[str, str]], model: str):
        return sn(content=[sn(type="text", text="random test text")], usage=sn(input_tokens=123, output_tokens=456))

    monkeypatch.setattr(rag.synthesizer.client.messages, "create", mock_anthropic_create)

    source_chunks = [
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id="t_syn_123",
                collection="syn_collection",
                text="Montenegro has beautiful mountains.",
                chunk_index=0,
                chunk_metadata=ChunkMetadata(source_file="t_syn.pdf", page_number=1, char_start=0, char_end=36)
            ),
            score=0.9,
            rank=1
        )
    ]

    result = rag.synthesizer.synthesize(
        "question", source_chunks, "syn_collection"
    )
    assert result.input_tokens == 123
    assert result.output_tokens == 456
    assert result.answer == "random test text"
    assert result.question == "question"
    assert result.sources == source_chunks
    assert result.collection == "syn_collection"
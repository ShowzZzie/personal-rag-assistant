import rag.embedder
from types import SimpleNamespace as sn
import pytest

def test_get_embedding(monkeypatch):
    
    def mock_embed(input: str, model: str):
        resp = sn(data=[sn(embedding=[-0.1, 0.23, -0.004, 1.2])])
        return resp

    monkeypatch.setattr(rag.embedder.client.embeddings, "create", mock_embed)

    result = rag.embedder.get_embedding("yo")
    assert len(result)==4
    assert result[0]==-0.1
    assert result[1]==0.23
    assert result[2]==-0.004
    assert result[3]==1.2

def test_get_embedding_model_names(monkeypatch):
    models_called = {}

    def mock_embed(input: str, model: str):
        resp = sn(data=[sn(embedding=[-0.1, 0.23, -0.004, 1.2])])
        models_called["model"]=model
        return resp

    monkeypatch.setattr(rag.embedder.client.embeddings, "create", mock_embed)

    rag.embedder.get_embedding("yo")
    assert models_called["model"] == "text-embedding-3-small"
    rag.embedder.get_embedding("cya", "test-embdr-1-large")
    assert models_called["model"] =="test-embdr-1-large"


def test_get_embedding_empty_string(monkeypatch):

    def mock_embed(input: str, model: str):
        resp = sn(data=[sn(embedding=[-0.1, 0.23, -0.004, 1.2])])
        return resp

    monkeypatch.setattr(rag.embedder.client.embeddings, "create", mock_embed)
    
    with pytest.raises(ValueError) as err1:
        rag.embedder.get_embedding("")
    assert "empty" in str(err1.value)

    with pytest.raises(ValueError) as err2:
        rag.embedder.get_embedding("       ")
    assert "empty" in str(err2.value)
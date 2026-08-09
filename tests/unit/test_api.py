import rag.api
from fastapi.testclient import TestClient

from rag.schemas import Document

client = TestClient(rag.api.app)

def test_documents_404(monkeypatch):

    def mock_404(id: int):
        raise ValueError("File not found")
    monkeypatch.setattr(rag.api, "query_by_id", mock_404)

    response = client.get("/documents/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "File not found"


def test_ingest(monkeypatch):
    
    def mock_ingest(file: str, collection: str, size: int, overlap: int):
        return Document(document_id="testdocid", collection="testdoccollection", filename="testdoc", chunk_count=3, embedding_model="testmod")
    monkeypatch.setattr(rag.api, "ingest", mock_ingest)

    response = client.post("/ingest", params={"collection": "blah"}, files={"file": ("test.pdf", b"fake pdf bytes", "application/pdf")})

    assert response.status_code == 200
    assert response.json()["document_id"] == "testdocid"
    assert response.json()["chunk_count"] == 3
from rag.schemas import DocumentChunk
from rag.embedder import get_embedding
import chromadb

chroma_client = chromadb.Client()

def add_chunks(chunks: list[DocumentChunk]) -> None:
    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    
    if chunks is None or len(chunks) == 0:
        raise IndexError("Chunks can't be empty")
    else:
        collection = chroma_client.get_or_create_collection(name=chunks[0].collection)

    for chunk in chunks:
        ids.append(str(chunk.chunk_id))
        embeddings.append(get_embedding(chunk.text))
        documents.append(chunk.text)
        metadatas.append(chunk.chunk_metadata.model_dump())

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
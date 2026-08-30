import uuid
from typing import TYPE_CHECKING

import chromadb
from chromadb.errors import NotFoundError

from rag.config import settings
from rag.embedder import get_embedding
from rag.schemas import ChunkMetadata, DocumentChunk, RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chromadb.api import ClientAPI
    from chromadb.api.types import Metadata

chroma_persistent_client = chromadb.PersistentClient(path=settings.chroma_path)


def add_chunks(chunks: list[DocumentChunk], client: ClientAPI | None = None) -> None:
    ids: list[str] = []
    embeddings: list[Sequence[float]] = []
    documents: list[str] = []
    metadatas: list[Metadata] = []

    if client is None:
        client = chroma_persistent_client

    if chunks is None or len(chunks) == 0:
        raise IndexError("Chunks can't be empty")
    else:
        collection = client.get_or_create_collection(
            name=chunks[0].collection,
            metadata={"hnsw:space": "cosine"}
        )

    for chunk in chunks:
        ids.append(str(chunk.chunk_id))
        embeddings.append(get_embedding(chunk.text))
        documents.append(chunk.text)
        meta = chunk.chunk_metadata.model_dump()
        meta["document_id"] = chunk.document_id
        meta["chunk_index"] = chunk.chunk_index
        meta = {k: v for k, v in meta.items() if v is not None}
        metadatas.append(meta)

    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)


def query_by_vector(
    vector: list[float],
    collection_query: str,
    top_k: int,
    client: ClientAPI | None = None
    ) -> list[RetrievedChunk]:
    if client is None:
        client = chroma_persistent_client
    
    collection = client.get_or_create_collection(
        name=collection_query,
        metadata={"hnsw:space": "cosine"}
    )

    query_vectors: list[Sequence[float]] = [vector]

    result = collection.query(
        query_embeddings=query_vectors,
        n_results=top_k
        )
    
    results_to_return = []
    assert result["documents"]
    assert result["distances"]
    assert result["metadatas"]

    for i, id in enumerate(result["ids"][0]):
        results_to_return.append(RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id=uuid.UUID(id),
                document_id=result["metadatas"][0][i]["document_id"],
                collection=collection_query,
                text=result["documents"][0][i],
                chunk_index=result["metadatas"][0][i]["chunk_index"],
                chunk_metadata=ChunkMetadata(
                    source_file=result["metadatas"][0][i]["source_file"],
                    page_number=result["metadatas"][0][i].get("page_number"),
                    char_start=result["metadatas"][0][i]["char_start"],
                    char_end=result["metadatas"][0][i]["char_end"]
                )
            ),
            score=1-result["distances"][0][i],
            rank=i+1
        ))
    
    return results_to_return



def delete_collection(collection_name: str, client: ClientAPI | None = None) -> None:
    if client is None:
        client = chroma_persistent_client

    try:
        client.delete_collection(collection_name)
    except NotFoundError as err:
        raise ValueError(f"[store:delete_collection] DELETE FAILED: {err}") from err
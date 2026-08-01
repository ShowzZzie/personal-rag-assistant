from typing import TYPE_CHECKING

from rag.embedder import get_embedding
from rag.store import query_by_vector

if TYPE_CHECKING:
    from rag.schemas import RetrievedChunk


def retrieve(
    query: str,
    collection_name: str,
    top_k: int
    ) -> list[RetrievedChunk]:

    embedded_query = get_embedding(query)
    query_result = query_by_vector(
        embedded_query,
        collection_name,
        top_k
    )

    return query_result
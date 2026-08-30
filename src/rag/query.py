from typing import TYPE_CHECKING

import rag.retriever
import rag.synthesizer
from rag.config import settings

if TYPE_CHECKING:
    from rag.schemas import Answer, RetrievedChunk

def query(
    user_query: str,
    user_collection: str,
    user_top_k: int = settings.top_k,
) -> Answer:
    
    retrieve_result: list[RetrievedChunk] = rag.retriever.retrieve(
        query=user_query,
        collection_name=user_collection,
        top_k=user_top_k
    )

    synthesize_result: Answer = rag.synthesizer.synthesize(
        user_question=user_query,
        source_chunks=retrieve_result,
        collection=user_collection
    )

    return synthesize_result
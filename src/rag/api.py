# POST /ingest              # upload and ingest a document
# GET  /collections         # list collections and document counts
# POST /query               # ask a question, get sourced answer
# GET  /documents/{id}      # document metadata

from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from rag.pipeline import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    get_collections_counts,
    ingest,
    query_by_id,
)
from rag.query import DEFAULT_TOP_K, query
from rag.schemas import Answer, Document


class QueryBody(BaseModel):
    question: str
    collection: str
    top_k: int = DEFAULT_TOP_K # waiting for config.py

app = FastAPI()

@app.post("/ingest") # to use pipeline's ingest, we need: file, collection, size, overlap
async def post_ingest_api(
    file: UploadFile,
    collection: str,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> Document:

    content = await file.read()

    if file.filename is None:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename")
    
    save_path = Path("data/documents") / file.filename
    save_path.write_bytes(content)
    
    result_document = ingest(str(save_path), collection, size, overlap)

    return result_document

@app.get("/collections")
def get_collections_api() -> dict[str,int]:
    return get_collections_counts()

@app.post("/query")
# need to adjust rag.store to only use get_collection when using this.
# Right now, querying non-existent collections, creates it via get_or_create_collection
def post_query_api(request: QueryBody) -> Answer:
    result = query(request.question, request.collection, request.top_k)
    return result

@app.get("/documents/{id}", response_model=Document)
def get_documents_api(id: int) -> Document:
    try:
        return query_by_id(id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ingest import load_documents
from chunk_ember import chunk_documents
from vector_store import build_faiss_index, search_faiss_index
from llm_reasoning import run_llm_on_query

DOCS_PATH = os.getenv("DOCS_PATH", "docs")

app = FastAPI(title="Policy QA API")


class QueryIn(BaseModel):
    query: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/ingest")
def ingest_endpoint():
    if not os.path.isdir(DOCS_PATH):
        raise HTTPException(status_code=400, detail=f"Docs folder not found at path: {DOCS_PATH}")
    docs = load_documents(DOCS_PATH)
    chunked = chunk_documents(docs)
    build_faiss_index(chunked)
    return {"status": "ingested", "documents": len(docs), "chunks": sum(len(d.get("chunks", [])) for d in chunked)}


@app.post("/query")
def query_endpoint(payload: QueryIn):
    results = search_faiss_index(payload.query, k=5)
    llm_output = run_llm_on_query(payload.query, results)
    return {"query": payload.query, "retrieved": len(results), "llm": llm_output}

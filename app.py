"""
app.py — FastAPI wrapper. Builds the index once at startup (TF-IDF fit is
cheap, but no reason to redo it per request), serves /ask after that.

    uvicorn app:app --reload --port 8006
    curl -X POST http://localhost:8006/ask -H "Content-Type: application/json" \
        -d '{"query": "What was the gross margin?", "mode": "mock"}'
"""

from __future__ import annotations

import os
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag import ask, build_corpus_index

DOCS_DIR = os.environ.get("FILINGS_DOCS_DIR", "sample_filings")

_index = None

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app_: FastAPI):
    # Lifespan pattern (matches THESIS's own main.py; @app.on_event is deprecated).
    # TF-IDF fitting is cheap but there's no reason to redo it per request —
    # build once at startup, serve reads after.
    global _index
    _index = build_corpus_index(DOCS_DIR)
    print(f"[app] Indexed corpus from '{DOCS_DIR}'.")
    yield


app = FastAPI(title="Filings RAG Assistant API", version="1.0", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AskRequest(BaseModel):
    query: str
    mode: str = Field("mock", pattern="^(mock|live)$")
    top_k: int = Field(4, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok", "docs_dir": DOCS_DIR, "chunks_indexed": len(_index.chunks) if _index else 0}


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    try:
        result = ask(req.query, _index, mode=req.mode, top_k=req.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "query": result.query,
        "answer": result.answer.answer,
        "citations": result.answer.citations,
        "mode": result.answer.mode,
        "retrieved": [{"chunk_id": r.chunk.chunk_id, "score": r.score} for r in result.retrieved],
    }

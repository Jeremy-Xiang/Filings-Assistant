"""
rag.py — Ties ingestion, retrieval, and generation together. The only
function the CLI and the API both call.
"""

from __future__ import annotations

from dataclasses import dataclass

from .index import RetrievedChunk, TfidfIndex, build_index
from .ingest import load_corpus
from .llm_client import RAGAnswer, get_llm_client


@dataclass
class RAGResult:
    query: str
    answer: RAGAnswer
    retrieved: list[RetrievedChunk]


def build_corpus_index(docs_dir: str = "sample_filings") -> TfidfIndex:
    chunks = load_corpus(docs_dir)
    if not chunks:
        raise ValueError(f"No .txt documents found in '{docs_dir}'.")
    return build_index(chunks)


def ask(query: str, index: TfidfIndex, mode: str = "mock", top_k: int = 4) -> RAGResult:
    retrieved = index.search(query, top_k=top_k)
    llm_client = get_llm_client(mode)
    answer = llm_client.generate_answer(query, retrieved)
    return RAGResult(query=query, answer=answer, retrieved=retrieved)

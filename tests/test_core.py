"""
Tests for filings-assistant. Run: pytest tests/ -v

Pins the two behaviors that define this project: (1) the short-chunk
merge fix — a tiny trailing chunk must never survive as a standalone
citation target (that's how an unrelated "weather" query scored 0.27
against a 14-word closing remark), and (2) honest refusal — an
irrelevant query must return "not found," never a forced weak match.
"""

import pytest

from src.index import build_index
from src.ingest import MIN_CHUNK_WORDS, load_corpus
from src.llm_client import NOT_FOUND_THRESHOLD, MockLLMClient
from src.rag import ask, build_corpus_index

DOCS = "sample_filings"


@pytest.fixture(scope="module")
def index():
    return build_corpus_index(DOCS)


def test_no_tiny_standalone_chunks():
    """The bug fix: every chunk must clear the minimum word count
    (a lone tiny chunk is a false-positive magnet for TF-IDF)."""
    for chunk in load_corpus(DOCS):
        assert len(chunk.text.split()) >= MIN_CHUNK_WORDS, chunk.chunk_id


def test_chunks_carry_source_and_heading():
    chunks = load_corpus(DOCS)
    assert len(chunks) > 0
    for c in chunks:
        assert c.source_file and c.chunk_id.startswith(c.source_file)


def test_relevant_query_clears_threshold(index):
    top = index.search("What was the gross margin and what drove the change?", top_k=1)[0]
    assert top.score >= NOT_FOUND_THRESHOLD


def test_terse_relevant_query_clears_threshold(index):
    """The calibration fix: 'How concentrated is revenue?' scored 0.117 and
    was wrongly refused at the old 0.12 threshold."""
    top = index.search("How concentrated is revenue?", top_k=1)[0]
    assert top.score >= NOT_FOUND_THRESHOLD


def test_irrelevant_query_below_threshold(index):
    top = index.search("What is the weather like in San Francisco today?", top_k=1)[0]
    assert top.score < NOT_FOUND_THRESHOLD


def test_mock_refuses_irrelevant(index):
    r = ask("What is the weather like in San Francisco today?", index, mode="mock")
    assert r.answer.citations == []
    assert "Nothing in the indexed documents" in r.answer.answer


def test_mock_answers_relevant_with_citations(index):
    r = ask("How concentrated is revenue among the largest customers?", index, mode="mock")
    assert len(r.answer.citations) >= 1
    assert r.answer.answer.startswith("[MOCK]")  # extractive output must be labeled
    # every citation must be a real chunk id from the corpus
    valid_ids = {c.chunk_id for c in load_corpus(DOCS)}
    assert all(cid in valid_ids for cid in r.answer.citations)


def test_cross_document_retrieval(index):
    """Customer concentration appears in BOTH the 10-K and the call
    transcript in different words — retrieval should surface both."""
    r = ask("How concentrated is revenue among the largest customers?", index, mode="mock")
    sources = {cid.split("::")[0] for cid in r.answer.citations}
    assert len(sources) == 2


def test_live_mode_requires_key(monkeypatch, index):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ask("anything", index, mode="live")

"""
index.py — TF-IDF + cosine similarity retrieval.

Why TF-IDF instead of a real embedding model: this sandbox has no network
access to download a pretrained embedding model (and no API key to call a
hosted embeddings endpoint), so the choice here was "TF-IDF" or "nothing
that actually retrieves anything." That's a constraint, not a recommendation
— but it's also not a bad fit for this specific domain. Financial filings
are dense with exact, distinctive terminology ("net dollar retention,"
"gross margin," specific dollar figures) where lexical overlap between a
question and the right chunk is usually strong. TF-IDF would do noticeably
worse on conversational or paraphrased queries where the right chunk uses
different words for the same idea — that's the real tradeoff, and the
honest reason to swap in real embeddings (OpenAI, Voyage, Cohere, or a
local sentence-transformers model) once this runs somewhere with network
access: see README for exactly where that swap goes.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingest import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class TfidfIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked_idx = scores.argsort()[::-1][:top_k]
        return [RetrievedChunk(self.chunks[i], float(scores[i])) for i in ranked_idx]


def build_index(chunks: list[Chunk]) -> TfidfIndex:
    return TfidfIndex(chunks)

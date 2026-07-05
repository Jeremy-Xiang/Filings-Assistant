"""
llm_client.py — Turns retrieved chunks into a grounded, cited answer.

Two implementations, same pattern as the multi-agent-analyst project:
- AnthropicLLMClient: real Claude call. The system prompt is the actual
  anti-hallucination mechanism — it instructs the model to answer ONLY
  from the provided chunks, cite every claim, and explicitly say so when
  the answer isn't supported, rather than filling the gap from general
  knowledge about the topic.
- MockLLMClient: no network, no API key. Purely extractive — extracts the
  top-scoring chunk(s) above a threshold and presents them as the "answer,"
  with an explicit [MOCK] label. It cannot synthesize across chunks or
  paraphrase; it can only prove that retrieval and citation plumbing work
  end to end. Don't mistake fluency for synthesis in mock mode — there
  isn't any.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .index import RetrievedChunk

NOT_FOUND_THRESHOLD = 0.10  # calibrated against this project's sample corpus:
                              # relevant-but-terse queries ("How concentrated is
                              # revenue?") score ~0.117, clearly-irrelevant ones
                              # ("weather in San Francisco?") score ~0.067 after
                              # the short-chunk merge fix. 0.12 was rejecting the
                              # former; 0.10 separates the two cleanly. Recalibrate
                              # for a different or much larger corpus.

SYSTEM_PROMPT = """\
You are a financial filings assistant. You will be given a question and several \
excerpts retrieved from a company's filings and earnings call transcripts. \
Answer using ONLY information contained in the excerpts provided — never fill \
gaps from general knowledge about the company, the industry, or what a typical \
filing might say. \
For every factual claim in your answer, cite the excerpt it came from using its \
chunk_id in brackets, e.g. [solstice_10k_excerpt.txt::chunk4]. \
If the excerpts don't contain enough information to answer the question, say so \
explicitly instead of guessing or extrapolating."""


@dataclass
class RAGAnswer:
    answer: str
    citations: list[str]
    mode: str


class LLMClient(ABC):
    @abstractmethod
    def generate_answer(self, query: str, retrieved: list[RetrievedChunk]) -> RAGAnswer:
        ...


class AnthropicLLMClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before using AnthropicLLMClient, "
                "or use MockLLMClient for offline testing."
            )
        self.client = anthropic.Anthropic()
        self.model = model

    def generate_answer(self, query: str, retrieved: list[RetrievedChunk]) -> RAGAnswer:
        context = "\n\n".join(
            f"[{r.chunk.chunk_id}] (relevance score {r.score:.2f}, section: {r.chunk.heading or 'n/a'})\n{r.chunk.text}"
            for r in retrieved
        )
        user_message = f"EXCERPTS:\n\n{context}\n\nQUESTION: {query}"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer_text = "".join(b.text for b in response.content if b.type == "text")
        citations = sorted({r.chunk.chunk_id for r in retrieved if r.chunk.chunk_id in answer_text})
        return RAGAnswer(answer=answer_text, citations=citations, mode="live")


class MockLLMClient(LLMClient):
    """Purely extractive — no real synthesis. See module docstring."""

    def generate_answer(self, query: str, retrieved: list[RetrievedChunk]) -> RAGAnswer:
        relevant = [r for r in retrieved if r.score >= NOT_FOUND_THRESHOLD]

        if not relevant:
            return RAGAnswer(
                answer=f"[MOCK] Nothing in the indexed documents scored above the "
                f"relevance threshold ({NOT_FOUND_THRESHOLD}) for this question. "
                f"Best available match scored {retrieved[0].score:.3f} — too low to "
                f"treat as a real answer. Saying 'not found' here rather than forcing "
                f"a guess from a weak match.",
                citations=[],
                mode="mock",
            )

        lines = [f"[MOCK] Extractive answer — top matching excerpt(s), not a real synthesized answer:\n"]
        for r in relevant:
            lines.append(f"From [{r.chunk.chunk_id}] (score={r.score:.2f}):\n{r.chunk.text}\n")

        return RAGAnswer(
            answer="\n".join(lines),
            citations=[r.chunk.chunk_id for r in relevant],
            mode="mock",
        )


def get_llm_client(mode: str) -> LLMClient:
    if mode == "live":
        return AnthropicLLMClient()
    if mode == "mock":
        return MockLLMClient()
    raise ValueError(f"Unknown mode '{mode}'. Use 'live' or 'mock'.")

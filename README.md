# filings-assistant

Q&A over financial filings and earnings transcripts, with citations. Ask a question, get an answer that cites the specific chunk it came from — or an explicit refusal if nothing relevant is indexed. The refusal is the feature: a system that answers confidently from weak matches is worse than one that says "not found."

## How it works

Documents (10-Ks, earnings call transcripts) go into `.txt` files in `sample_filings/`. The chunker splits on paragraph breaks, merges consecutive paragraphs up to ~120 words, and attaches the nearest preceding section heading to each chunk for citation context. Chunks are indexed via TF-IDF.

At query time, cosine similarity retrieves the top-k chunks. In mock mode, those chunks are returned verbatim with their chunk IDs labeled. In live mode, they're passed to Claude with a system prompt that constrains the response to only information present in the provided excerpts — no filling gaps from general knowledge, every claim cited by chunk ID.

The relevance threshold (0.10) is where this project had its one interesting calibration problem.

## The short-chunk false positive

A 14-word chunk survived as a standalone citation target: `"...Thank you for joining us today."` Querying something completely unrelated — weather in San Francisco — scored **0.27** against it. The cause: with so few terms in the chunk, a single incidental shared word ("today") dominated the TF-IDF cosine score. A 0.27 score from a relevant query on a real question scores around 0.30, so this chunk was surfacing alongside genuinely useful results.

Fix: chunks under 25 words get merged backward into the previous chunk rather than standing alone (`MIN_CHUNK_WORDS` in `src/ingest.py`). After the fix, the weather query scores 0.067 — clearly below the 0.10 threshold. The real queries stay above it.

The threshold was also recalibrated during this process. "How concentrated is revenue among the largest customers?" — an answerable question — scored 0.117 and was wrongly refused at the original 0.12 cutoff. Irrelevant queries score ~0.067. Moving to 0.10 separates them cleanly; the test suite pins both cases.

Short chunks are specifically a TF-IDF problem. Dense embedding models handle sparse term overlap much better because semantic similarity doesn't depend on exact word match. If you swap in real embeddings later, the minimum chunk size threshold can probably be loosened.

## Why TF-IDF

No network access to download a pretrained embedding model. TF-IDF is a reasonable fit for financial filings specifically — they're dense with distinctive terminology ("net dollar retention," specific dollar figures) where lexical overlap between a question and the right chunk tends to be strong. Where it fails: paraphrased questions where the right chunk uses different words for the same concept. That's the real reason to upgrade to embeddings — swap `TfidfIndex` in `src/index.py` for an embedding-based index with the same `search(query, top_k) -> list[RetrievedChunk]` interface, and nothing else changes.

## Running it

```bash
pip install -r requirements.txt

python run_rag.py --query "What drove the gross margin improvement?" --mock
export ANTHROPIC_API_KEY=sk-...
python run_rag.py --query "..." --live --docs-dir path/to/your/filings
```

Or as an API:

```bash
uvicorn app:app --port 8006
curl -X POST http://localhost:8006/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How concentrated is revenue?", "mode": "mock"}'
```

Set `FILINGS_DOCS_DIR` to point at real filings. Any `.txt` files in that directory get chunked and indexed at startup.

Live mode without a key returns a clean `400`, not a crash.

## Running the tests

```bash
pytest tests/ -v
```

Nine tests. The two most relevant: `test_no_tiny_standalone_chunks` verifies every chunk clears `MIN_CHUNK_WORDS`, and `test_cross_document_retrieval` verifies that a question about customer concentration (which appears in both the 10-K and the earnings call, in different words) surfaces chunks from both documents.

## Structure

```
filings-assistant/
├── run_rag.py          # CLI
├── app.py              # FastAPI service
├── sample_filings/     # synthetic 10-K excerpt + earnings call (fictional company)
├── src/
│   ├── ingest.py       # chunking with heading-aware split and short-chunk merge
│   ├── index.py        # TF-IDF retrieval
│   ├── llm_client.py   # AnthropicLLMClient (real) + MockLLMClient (extractive)
│   └── rag.py          # orchestrates ingest → index → retrieve → generate
└── tests/test_core.py
```

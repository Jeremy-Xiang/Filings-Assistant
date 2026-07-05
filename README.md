# filings-assistant

A citation-grounded Q&A system over financial filings and earnings call
transcripts. Retrieval is TF-IDF + cosine similarity over chunked
documents; generation is a real Claude call constrained to answer only
from retrieved text, with every claim citable back to a specific chunk —
or, when nothing relevant is found, an explicit refusal instead of a
plausible-sounding guess.

## The core design goal: never answer from outside the documents

This only matters if it actually holds up when nothing relevant is
indexed. Tested directly:

```
QUERY: What is the weather like in San Francisco today?
[MOCK] Nothing in the indexed documents scored above the relevance
threshold (0.10) for this question. Best available match scored 0.067 —
too low to treat as a real answer. Saying 'not found' here rather than
forcing a guess from a weak match.
```

versus a real, well-supported query:

```
QUERY: How concentrated is revenue among the largest customers?
[citations: solstice_earnings_call_q4.txt::chunk4, solstice_10k_excerpt.txt::chunk3]
```

Both the 10-K and the earnings call discuss customer concentration in
different words ("our three largest customers accounted for approximately
41% of total revenue" vs. "Our top three customers represented
approximately 41% of revenue") — retrieval correctly pulls both, which is
the whole point of indexing more than one document type.

## Why TF-IDF instead of real embeddings — an honest tradeoff, not a recommendation

This sandbox has no network access to download a pretrained embedding
model and no API key to call a hosted embeddings endpoint. TF-IDF was the
option that could actually retrieve something versus nothing. That said,
it's a defensible choice for *this specific domain*, not just a fallback:
financial filings are dense with exact, distinctive terminology ("net
dollar retention," specific dollar figures, "gross margin") where lexical
overlap between a question and the right chunk tends to be strong. Where
it would do noticeably worse: a paraphrased or conversational question
where the right chunk uses different words for the same idea — TF-IDF has
no notion that "profit per dollar of sales" and "gross margin" are related.
That's the honest reason to swap in real embeddings (OpenAI, Voyage,
Cohere, or a local sentence-transformers model) once this runs somewhere
with network access — swap `TfidfIndex` in `src/index.py` for an embedding-
based index with the same `search(query, top_k) -> list[RetrievedChunk]`
interface, and nothing else in the pipeline needs to change.

## A real bug found while building this

The first version of the chunker left a 14-word trailing chunk standing on
its own (`"...Thank you for joining us today."`). Querying something
totally unrelated — "What is the weather like in San Francisco today?" —
scored **0.27** against that chunk, higher than several genuinely relevant
results score against real questions. The cause: TF-IDF cosine similarity
on a very short chunk has few terms diluting the vector, so a single
incidental shared word ("today") dominated the score. Fixed by merging
trailing chunks under ~25 words into the previous chunk instead of leaving
them as standalone, fragile citations (`MIN_CHUNK_WORDS` in
`src/ingest.py`). After the fix, the same weather query scores 0.067 — well
below the "found" threshold, where it belongs.

The general lesson: short chunks are a real liability for TF-IDF
retrieval specifically (much less of an issue for real embeddings, which
don't degrade the same way with sparse term overlap) — worth checking for
directly if you extend this rather than assuming "it retrieved something
with a nonzero score" means "it retrieved something relevant."

## Mock mode vs. live mode

| | `--mock` | `--live` |
|---|---|---|
| Cost | Free | Real Anthropic API usage |
| What it does | Extracts the top-scoring chunk(s) verbatim, prefixed `[MOCK]` | Real Claude call, constrained by system prompt to cite and to refuse when unsupported |
| Can it synthesize across chunks | No — purely extractive | Yes |

Mock mode exists to prove the retrieval and citation plumbing work without
spending money or needing a key — it is structurally incapable of
synthesizing an answer, which is the point, not a limitation to apologize
for.

## Running it

```bash
pip install -r requirements.txt

python run_rag.py --query "What was the gross margin and what drove the change?" --mock
export ANTHROPIC_API_KEY=sk-...
python run_rag.py --query "..." --live
```

Or as an API:
```bash
uvicorn app:app --reload --port 8006
curl -X POST http://localhost:8006/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How concentrated is revenue among the largest customers?", "mode": "mock"}'
```

## Using your own documents

```bash
python run_rag.py --query "..." --docs-dir path/to/your/filings --mock
```

Or set `FILINGS_DOCS_DIR` for the API. Any `.txt` files in that directory
get chunked and indexed at startup — drop in real 10-Ks, 10-Qs, or earnings
call transcripts for any of your 53 THESIS tickers and ask against them
directly.

## Wiring into THESIS

1. Copy `src/` in as a sibling package, mount `app.py`'s `/ask` route under
   THESIS's existing FastAPI app — same pattern as the rest of this
   project series.
2. Point `FILINGS_DOCS_DIR` at a directory of real filings for your 53
   tickers (SEC EDGAR's full-text search API is a reasonable source for
   10-Ks; you'll need a real fetcher, since none is built here on purpose —
   same reasoning as `multi-agent-analyst`'s headlines.py: don't build a
   redundant data layer when one might already make sense elsewhere in
   THESIS).
3. New React tab: a question box, the answer with inline citation badges,
   and a collapsible "sources" panel showing each retrieved chunk's score
   and excerpt — letting someone check the citation against the actual
   source text is the actual point of the auditability here.

## Project structure

```
filings-assistant/
├── run_rag.py              # CLI
├── app.py                  # FastAPI service
├── sample_filings/          # synthetic 10-K excerpt + earnings call (fake company)
└── src/
    ├── ingest.py              # chunking (paragraph-aware, heading-aware, merges tiny trailing chunks)
    ├── index.py                # TF-IDF retrieval
    ├── llm_client.py            # AnthropicLLMClient (real) + MockLLMClient (extractive)
    └── rag.py                    # orchestrates ingest -> index -> retrieve -> generate
```

## Running the tests

```bash
pytest tests/ -v
```

The suite pins the behaviors that actually caught bugs during development
(see the sections above), not ceremony coverage — every test encodes a
check where the wrong answer was at some point the actual behavior.

## Possible next steps

- Swap in real embeddings once this runs with network access (see "why
  TF-IDF" above) — the interface is already designed for that swap to be
  a one-file change.
- Add a re-ranking step: retrieve more candidates than `top_k` with TF-IDF,
  then have the LLM itself re-rank or discard irrelevant ones before
  generating — cheap precision improvement once you're paying for live
  calls anyway.
- Persist the index instead of rebuilding it at every startup, once the
  corpus is large enough (real 10-Ks across 53 tickers) that re-chunking
  and re-fitting TF-IDF on every restart is actually slow.

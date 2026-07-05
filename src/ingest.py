"""
ingest.py — Load .txt filings and split them into citable chunks.

Chunking strategy: split on paragraph breaks first (preserves natural
sentence/idea boundaries instead of cutting mid-sentence at a fixed
character count), then merge consecutive paragraphs up to a target word
count. Each chunk also carries the nearest preceding ALL-CAPS heading line
it fell under (e.g. "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS"), so a
citation can say *where* in the document an answer came from, not just
which file.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

TARGET_WORDS_PER_CHUNK = 120
_HEADING_PATTERN = re.compile(r"^[A-Z0-9 .,&'\-]{6,}$")  # a whole line, mostly caps — a heading guess


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    heading: str
    text: str


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _looks_like_heading(paragraph: str) -> bool:
    # Single short-ish line, mostly uppercase — distinguishes "ITEM 1A. RISK
    # FACTORS" from an actual all-caps acronym buried inside a normal sentence.
    lines = paragraph.splitlines()
    return len(lines) <= 2 and bool(_HEADING_PATTERN.match(paragraph.strip()))


MIN_CHUNK_WORDS = 25  # below this, a chunk has too few distinct terms for TF-IDF
                        # similarity to be meaningful — a single incidental shared
                        # word (e.g. "today") can dominate the score against an
                        # otherwise-unrelated query. See README for the concrete case.


def chunk_document(path: str) -> list[Chunk]:
    with open(path) as f:
        text = f.read()

    source_file = os.path.basename(path)
    paragraphs = _split_paragraphs(text)

    chunks: list[Chunk] = []
    current_heading = ""
    buffer: list[str] = []
    buffer_words = 0
    chunk_idx = 0

    def flush(force: bool = False):
        nonlocal buffer, buffer_words, chunk_idx
        if not buffer:
            return
        text_block = "\n\n".join(buffer)

        # A too-small trailing chunk gets merged into the previous chunk of
        # the SAME document rather than standing alone, unless it's the only
        # content this document has at all (force=False path with no prior
        # chunk yet) or the caller forces a flush anyway (section boundary).
        if buffer_words < MIN_CHUNK_WORDS and chunks and chunks[-1].source_file == source_file and not force:
            prev = chunks[-1]
            chunks[-1] = Chunk(
                chunk_id=prev.chunk_id,
                source_file=prev.source_file,
                heading=prev.heading,
                text=prev.text + "\n\n" + text_block,
            )
        else:
            chunk_idx += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{source_file}::chunk{chunk_idx}",
                    source_file=source_file,
                    heading=current_heading,
                    text=text_block,
                )
            )
        buffer = []
        buffer_words = 0

    for para in paragraphs:
        if _looks_like_heading(para):
            flush(force=True)  # a real section boundary should still start a fresh chunk
            current_heading = para.strip()
            continue

        buffer.append(para)
        buffer_words += len(para.split())
        if buffer_words >= TARGET_WORDS_PER_CHUNK:
            flush(force=True)

    flush()  # trailing remainder — this is the one allowed to merge backward
    return chunks


def load_corpus(docs_dir: str) -> list[Chunk]:
    chunks = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        chunks.extend(chunk_document(path))
    return chunks

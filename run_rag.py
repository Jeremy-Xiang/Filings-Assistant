"""
run_rag.py — CLI entry point.

    python run_rag.py --query "What was the gross margin?" --mock
    python run_rag.py --query "..." --live    # requires ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse

from src.rag import ask, build_corpus_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--docs-dir", default="sample_filings")
    parser.add_argument("--top-k", type=int, default=4)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--mock", action="store_true")
    mode_group.add_argument("--live", action="store_true")
    args = parser.parse_args()

    mode = "live" if args.live else "mock"
    index = build_corpus_index(args.docs_dir)
    result = ask(args.query, index, mode=mode, top_k=args.top_k)

    print(f"Query: {result.query}\n")
    print(f"Answer ({mode} mode):\n{result.answer.answer}\n")
    print(f"Citations: {result.answer.citations}\n")
    print("Retrieved chunks (all, including any below the answer threshold):")
    for r in result.retrieved:
        print(f"  score={r.score:.3f}  {r.chunk.chunk_id}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI: python scripts/ask.py "What is the notice period?" — quick smoke test without the API/UI."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chain import get_rag_chain


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "your question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = get_rag_chain().answer(question)

    print(f"\nQ: {question}")
    print(f"A: {result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for src in result["sources"]:
            print(f"  - {src['source']} (relevance={src['score']})")


if __name__ == "__main__":
    main()

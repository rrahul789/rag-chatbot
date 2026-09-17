#!/usr/bin/env python3
"""CLI: python scripts/ingest.py [--path data/sample_docs] [--reset]"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingest import ingest_path, reset_collection

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--path", default=settings.documents_dir, help="File or directory to ingest.")
    parser.add_argument("--reset", action="store_true", help="Clear the collection before ingesting.")
    args = parser.parse_args()

    if args.reset:
        reset_collection()

    count = ingest_path(args.path)
    print(f"Ingested {count} chunk(s) from {args.path} into collection '{settings.chroma_collection_name}'.")


if __name__ == "__main__":
    main()

"""Load raw files from disk into LangChain Document objects.

Supports the formats a take-home is realistically tested with: PDF, plain
text, markdown and docx. Deliberately not using `unstructured` here — it
pulls in a heavy dependency tree (poppler/libmagic/onnx) for what is, for
this project's scope, a solved problem with the lighter pypdf/docx2txt
loaders LangChain already ships wrappers for.
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def _load_single_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif suffix == ".docx":
        docs = Docx2txtLoader(str(path)).load()
    elif suffix in (".txt", ".md"):
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    for doc in docs:
        doc.metadata["source"] = path.name

    return docs


def load_documents(path: str | Path) -> list[Document]:
    """Load one file, or every supported file in a directory (recursive)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file or directory: {path}")

    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    documents: list[Document] = []
    for file_path in files:
        try:
            documents.extend(_load_single_file(file_path))
        except Exception:  # noqa: BLE001 - one bad file shouldn't kill a full ingest run
            logger.exception("Failed to load %s, skipping it", file_path)

    logger.info("Loaded %d document(s)/page(s) from %s", len(documents), path)
    return documents

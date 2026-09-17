"""Chunk documents, embed them, and persist to the Chroma vector store."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.embeddings import get_embeddings
from app.loaders import load_documents

logger = logging.getLogger(__name__)


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata.setdefault("chunk_id", i)
    logger.info("Split %d document(s) into %d chunk(s)", len(documents), len(chunks))
    return chunks


def _chunk_id(chunk: Document) -> str:
    """Deterministic id from source + chunk index + content.

    Chroma upserts by id, so re-ingesting an unchanged file is a no-op instead
    of piling up duplicate chunks (which happened during manual testing:
    re-running ingest on the same sample docs let identical chunks crowd out
    everything else in top-k retrieval).
    """
    source = chunk.metadata.get("source", "unknown")
    chunk_index = chunk.metadata.get("chunk_id", 0)
    digest = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{source}:{chunk_index}:{digest}"


def get_vectorstore() -> Chroma:
    """Return a handle to the (possibly empty) persisted vector store."""
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
        # Chroma's default relevance-score conversion assumes L2 distance scaled
        # to roughly match OpenAI's embedding space, which produces nonsensical
        # (sometimes very negative) "relevance" scores for other embedding
        # models. Cosine distance + langchain's _cosine_relevance_score_fn
        # (1 - distance) is well-behaved regardless of embedding provider.
        collection_metadata={"hnsw:space": "cosine"},
    )


def ingest_path(path: str | Path) -> int:
    """Load, split, embed and upsert everything under `path`. Returns chunk count."""
    documents = load_documents(path)
    if not documents:
        logger.warning("No documents found at %s, nothing to ingest", path)
        return 0

    chunks = split_documents(documents)
    store = get_vectorstore()
    store.add_documents(chunks, ids=[_chunk_id(c) for c in chunks])
    logger.info("Ingested %d chunk(s) from %s into '%s'", len(chunks), path, settings.chroma_collection_name)
    return len(chunks)


def reset_collection() -> None:
    """Drop the collection entirely (used before a clean re-ingest)."""
    store = get_vectorstore()
    ids = store.get()["ids"]
    if ids:
        store.delete(ids=ids)
    logger.info("Cleared collection '%s' (%d chunk(s) removed)", settings.chroma_collection_name, len(ids))

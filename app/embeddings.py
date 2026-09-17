"""Embedding model factory.

Kept separate from chain.py so tests and scripts can swap in a fake
embedding function without importing the whole RAG chain.
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from app.config import settings


def get_embeddings() -> Embeddings:
    # Local HuggingFace model — no API key required, runs on CPU. Trades some
    # embedding quality for that; see README §4/§8 for the trade-off.
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embedding_model)

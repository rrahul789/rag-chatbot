"""Central configuration, read from environment variables / .env.

Kept as one small pydantic-settings object instead of scattering os.getenv()
calls through the codebase, so every tunable (chunk size, model names,
thresholds) lives in one place and is easy to override per-deployment.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Models: Ollama for the LLM, local HuggingFace for embeddings -----
    llm_model: str = "qwen3:1.7b"
    llm_temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector store ----------------------------------------------------
    chroma_persist_dir: str = str(BASE_DIR / "storage" / "chroma")
    chroma_collection_name: str = "knowledge_base"

    # --- Chunking / retrieval -------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 4
    retrieval_score_threshold: float = 0.25  # min relevance to trust a chunk (0-1, higher = stricter)

    # --- Data locations ---------------------------------------------------
    documents_dir: str = str(BASE_DIR / "data" / "sample_docs")
    uploads_dir: str = str(BASE_DIR / "data" / "uploads")

    # --- API ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Logging / observability ------------------------------------------
    log_level: str = "INFO"
    langchain_tracing_v2: bool = False  # set True + LANGCHAIN_API_KEY to use LangSmith


settings = Settings()

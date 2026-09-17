"""FastAPI service exposing the RAG chatbot: /chat, /ingest, /health.

A REST API in front of the chain (rather than having the Streamlit app
call the chain directly) exists so any client — a real frontend, curl,
another service — can use the same chatbot, and so ingestion/query
concerns can eventually scale independently of each other.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.chain import get_rag_chain
from app.guardrails import InputRejected
from app.ingest import ingest_path
from app.loaders import SUPPORTED_EXTENSIONS
from app.schemas import ChatRequest, ChatResponse, IngestResponse

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a local take-home; lock this down for real deployments
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        chain = get_rag_chain()
        history = [m.model_dump() for m in request.chat_history]
        result = chain.answer(request.question, chat_history=history)
        return ChatResponse(**result)
    except InputRejected as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error while answering question")
        raise HTTPException(status_code=500, detail="Internal error while generating an answer.") from exc


@app.post("/ingest", response_model=IngestResponse)
def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
    upload_dir = Path(settings.uploads_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {upload.filename}")
        dest = upload_dir / upload.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved_paths.append(dest)

    total_chunks = 0
    for path in saved_paths:
        total_chunks += ingest_path(path)

    return IngestResponse(chunks_ingested=total_chunks, files_processed=[p.name for p in saved_paths])


@app.post("/ingest/sample-docs", response_model=IngestResponse)
def ingest_sample_docs() -> IngestResponse:
    """Convenience endpoint to (re)ingest the bundled demo documents."""
    chunks = ingest_path(settings.documents_dir)
    files = [p.name for p in Path(settings.documents_dir).glob("*")]
    return IngestResponse(chunks_ingested=chunks, files_processed=files)

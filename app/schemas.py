from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    chat_history: list[ChatMessage] = Field(default_factory=list)


class Source(BaseModel):
    source: str
    page: int | None = None
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    standalone_question: str


class IngestResponse(BaseModel):
    chunks_ingested: int
    files_processed: list[str]

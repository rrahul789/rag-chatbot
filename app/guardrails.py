"""Lightweight, explicit guardrails.

Deliberately simple rather than reaching for a guardrails framework
(NeMo Guardrails / Guardrails AI) — for the scope of this project a few
explicit checks are easier to reason about, easier to unit test, and
easier for a reviewer to audit than a rules DSL would be. See the
README ("what I'd do differently") for where a real framework earns its
keep at larger scale.
"""
from __future__ import annotations

MIN_QUESTION_LENGTH = 3
MAX_QUESTION_LENGTH = 2000

NO_CONTEXT_ANSWER = (
    "I don't have information about that in the documents I've been given. "
    "Try rephrasing, or ask something covered by the ingested knowledge base."
)


class InputRejected(Exception):
    """Raised when a user message fails basic input validation."""


def validate_question(question: str) -> str:
    question = (question or "").strip()
    if len(question) < MIN_QUESTION_LENGTH:
        raise InputRejected("Question is too short.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise InputRejected(f"Question is too long (max {MAX_QUESTION_LENGTH} characters).")
    return question


def passes_relevance_threshold(scored_docs: list[tuple], threshold: float) -> bool:
    """scored_docs: list of (Document, relevance_score) where higher score = more relevant (0-1).

    Chroma's `similarity_search_with_relevance_score` already normalizes distance
    into this 0-1 "relevance" space, so callers don't need to know the underlying
    distance metric to use this.
    """
    if not scored_docs:
        return False
    best_score = max(score for _, score in scored_docs)
    return best_score >= threshold

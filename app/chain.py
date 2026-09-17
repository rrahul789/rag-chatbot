"""RAG chain: condense question -> retrieve -> relevance guardrail -> generate.

Split into named steps (rather than one long LCEL `|` pipeline) so each
stage is independently unit-testable and so the no-context short-circuit
is explicit: an irrelevant question never reaches the LLM at all, which
keeps behavior predictable and saves a call.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.guardrails import NO_CONTEXT_ANSWER, passes_relevance_threshold, validate_question
from app.ingest import get_vectorstore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the context below, which was \
retrieved from the user's own document collection.

Rules:
- If the answer is not contained in the context, say you don't know rather than guessing or using outside knowledge.
- Never invent facts, sources, or document names that are not in the context.
- Keep answers concise; use bullet points for lists of more than two items.
- The UI shows sources separately, so do not repeat raw filenames inside your answer text.

Context:
{context}
"""

CONDENSE_PROMPT = """Given the conversation history and a follow-up question, rewrite the follow-up question as a \
standalone question that includes any necessary context from the history. If it is already standalone, return it \
unchanged. Only output the rewritten question, nothing else.

Chat history:
{chat_history}

Follow-up question: {question}
Standalone question:"""


def get_llm():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        base_url=settings.ollama_base_url,
    )


def _format_chat_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in history)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata.get('source', 'unknown')}] {d.page_content}" for d in docs)


def _docs_to_sources(scored_docs: list[tuple[Document, float]]) -> list[dict]:
    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "score": round(float(score), 3),
            "snippet": doc.page_content[:300],
        }
        for doc, score in scored_docs
    ]


class RagChain:
    def __init__(self, llm=None, vectorstore=None):
        self.llm = llm or get_llm()
        self.vectorstore = vectorstore or get_vectorstore()
        self._condense_chain = ChatPromptTemplate.from_template(CONDENSE_PROMPT) | self.llm | StrOutputParser()
        self._answer_chain = (
            ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
            | self.llm
            | StrOutputParser()
        )

    def _standalone_question(self, question: str, chat_history: list[dict]) -> str:
        if not chat_history:
            return question
        return self._condense_chain.invoke(
            {"question": question, "chat_history": _format_chat_history(chat_history)}
        ).strip()

    def _retrieve(self, question: str) -> list[tuple[Document, float]]:
        return self.vectorstore.similarity_search_with_relevance_scores(question, k=settings.retrieval_top_k)

    def answer(self, question: str, chat_history: list[dict] | None = None) -> dict:
        question = validate_question(question)
        chat_history = chat_history or []

        standalone_question = self._standalone_question(question, chat_history)
        scored_docs = self._retrieve(standalone_question)

        if not passes_relevance_threshold(scored_docs, settings.retrieval_score_threshold):
            logger.info("No sufficiently relevant context for: %r", question)
            return {"answer": NO_CONTEXT_ANSWER, "sources": [], "standalone_question": standalone_question}

        docs = [doc for doc, _ in scored_docs]
        answer_text = self._answer_chain.invoke({"context": _format_docs(docs), "question": standalone_question})

        return {
            "answer": answer_text.strip(),
            "sources": _docs_to_sources(scored_docs),
            "standalone_question": standalone_question,
        }


_chain: RagChain | None = None


def get_rag_chain() -> RagChain:
    global _chain
    if _chain is None:
        _chain = RagChain()
    return _chain

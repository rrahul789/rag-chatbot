# rag-chatbot

Document Q&A chatbot — point it at PDFs/text/markdown/docx and ask questions in a
chat UI. Answers are grounded in retrieved passages only; it says "I don't know"
when nothing relevant is found.

**Stack:** LangChain (LCEL) · Chroma · FastAPI · Streamlit · Ollama (local LLM) ·
HuggingFace (local embeddings) — everything runs locally, no API key needed.

---

## 1. Quick setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ollama pull qwen3:1.7b        # or any model you have pulled — set LLM_MODEL in .env
cp .env.example .env          # defaults already point at Ollama + local embeddings

python scripts/ingest.py                       # ingest bundled sample docs
python scripts/ask.py "How many days of PTO?"  # CLI smoke test

uvicorn api.main:app --reload &     # API on :8000
streamlit run ui/streamlit_app.py   # UI on :8501
```

Open `localhost:8501`, click **"Load bundled sample docs"** (if you didn't already
run `scripts/ingest.py`), and ask away. Upload your own PDFs/txt/md/docx from the
sidebar to use a different collection.

Answers take ~10–30s depending on the local model and your machine — noticeably
slower and less sharp than a hosted model would be (see [§8](#8-whatd-i-do-differently)),
but free and fully offline.

---

## 2. Architecture

The RAG loop — ingest once, retrieve-and-guardrail-and-generate on every question.
Every box below is the actual technique/library/parameter used, not a placeholder:

```
┌────────────────────────────────────────────────────────┐
│                       documents                        │
│                   (pdf/txt/md/docx)                    │
│       PyPDFLoader / Docx2txtLoader / TextLoader        │
└────────────────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│               chunk + embed  (ingest.py)               │
│    RecursiveCharacterTextSplitter, 1000/150 overlap    │
│        MiniLM-L6-v2 embeddings, 384-dim (local)        │
└────────────────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│                 Chroma  (vector store)                 │
│           cosine distance, persisted on-disk           │
│ deterministic chunk IDs -> re-ingest upserts, no dupes │
└────────────────────────────────────────────────────────┘
                             │ top-k
                             ▼
┌────────────────────────────────────────────────────────┐         ┌──────────┐
│               retrieve top-k  (chain.py)               │         │ question │
│         top-k = 4, relevance = 1 - cosine_dist         │   <--   └──────────┘
└────────────────────────────────────────────────────────┘
                             │
                             ▼
              guardrail: best score >= 0.25 ?
                           ├─ no  ──▶  "I don't know"  (LLM never called, stop here)
                           │
                           └─ yes ──▶  continue below
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│                      LLM generate                      │
│          Ollama, qwen3:1.7b, temperature = 0           │
└────────────────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│                    answer + sources                    │
│              (source, relevance, snippet)              │
└────────────────────────────────────────────────────────┘
```

Before retrieval, a follow-up question is first condensed against chat history
into a standalone question (not shown above — see [§4](#4-ragllm-approach--decisions)),
so "what about after two years?" doesn't get embedded on its own. The guardrail is
the core design bet here: below the 0.25 threshold, the LLM is never called at all
— cheaper, and it doesn't depend on the model refusing on its own.

This chain is called the same way from three places: a CLI (`scripts/ask.py`), a
FastAPI backend (`api/main.py`: `/chat`, `/ingest`), and the Streamlit UI — which
talks to the API over HTTP rather than importing the chain directly, so the
backend stays independently scalable and swappable later. CORS is wide open (`*`)
on that API — fine locally, not for the internet (see [§3](#3-productionizing-on-a-hyperscaler)/[§6](#6-engineering-standards)).

## 3. Productionizing on a hyperscaler

Roughly in the order I'd tackle it:

- **Vector store** — Chroma-on-disk doesn't survive multiple replicas or nodes.
  Move to a managed store: Pinecone, hosted pgvector, or a hyperscaler-native option
  (Vertex AI Vector Search / Azure AI Search / OpenSearch). Biggest structural
  change — touches `app/ingest.py` and `app/chain.py` directly.
- **Compute** — containerize the API and UI and run them on Cloud Run/ECS Fargate
  (or EKS/GKE if a cluster already exists), behind a load balancer autoscaling on
  concurrency (LLM calls are latency-bound, not CPU-bound). UI becomes a proper
  frontend on S3+CloudFront / Cloud Storage+CDN / Cloudflare Pages.
- **Ingestion** — `/ingest` currently blocks on embedding calls. Split into:
  upload → object storage → queued worker (SQS/Lambda, Cloud Tasks, or Celery/RQ)
  → status endpoint the UI polls.
- **Secrets** — nothing needs an API key today (Ollama + local embeddings), but a
  production deployment would likely swap in a hosted model for latency/quality at
  scale, at which point that key goes into Secrets Manager/Secret Manager/Key
  Vault, injected at container start, never in `.env` committed anywhere.
- **Auth & abuse protection** — currently none; `/chat`/`/ingest` are open and CORS
  is `*`. Minimum bar: API keys or OAuth via the platform's gateway, per-key rate
  limiting, WAF in front (arbitrary user text is going straight to an LLM).
- **Observability** — `LANGCHAIN_TRACING_V2` is wired through as a one-line
  LangSmith on-ramp but not exercised end-to-end. Add structured log shipping,
  request tracing, and alerts on guardrail-refusal rate and p95 latency.
- **CI/CD** — GitHub Actions running tests/lint per PR, build+push on merge,
  platform-native rollout (Cloud Run revisions / ECS blue-green) for one-command
  rollback.
- **Cost** — content-hash IDs already make re-ingestion idempotent at the upsert
  level; next step is skipping the embedding *call* for unchanged content, and
  using a cheaper model tier for the condense step specifically.

---

## 4. RAG/LLM approach & decisions

| Component | Considered | Chose | Why |
|---|---|---|---|
| Orchestration | LangChain, LlamaIndex, hand-rolled | **LangChain/LCEL** | Required by the brief; LCEL made it natural to split condense/retrieve/generate into small testable steps instead of one opaque chain. |
| LLM | Hosted API (GPT-4o-mini/Claude Haiku), local via Ollama | **Ollama, local** | No API key, no cost, fully offline — the brief allowed any LLM, and a local model keeps this reviewer-runnable with zero setup friction. Trade-off: noticeably weaker instruction-following than a hosted model (see [§8](#8-whatd-i-do-differently)). |
| Embeddings | Hosted API, local `sentence-transformers` (MiniLM) | **MiniLM, local** | Same reasoning as the LLM — free, CPU-only, no external dependency. Weaker at nuance than a hosted embedding model, good enough for the sample docs here. |
| Vector DB | Chroma, FAISS, Pinecone, pgvector | **Chroma** | Embedded, zero infra for a reviewer to run. FAISS needs hand-rolled metadata storage; Pinecone/pgvector need external provisioning. Production trade-off in [§3](#3-productionizing-on-a-hyperscaler). |
| Context management | Stuff full history into prompt vs. condense-then-retrieve | **Condense-then-retrieve** | A raw follow-up ("what about after two years?") embeds poorly on its own; rewriting to a standalone question first fixes retrieval, at the cost of one extra LLM call/turn. |
| Guardrails | LLM self-refusal, a guardrails framework, explicit score threshold | **Explicit relevance threshold, prompt as backup** | Self-policing is unreliable, especially on small models. A hard threshold means the LLM is never even called for out-of-scope questions — cheaper and more reliable. |
| Quality/testing | Automated unit/eval tests vs. manual e2e runs | **Manual e2e runs only** | Kept the project deliberately simple — no test suite. Manual runs against both providers are how I found both bugs in [§5](#5-key-technical-decisions--why). No automated eval — see [§8](#8-whatd-i-do-differently). |
| Observability | Full LangSmith, custom metrics, nothing | **Structured logs + LangSmith on-ramp** | Added logging that'd actually help debugging now; left tracing as a documented toggle rather than an unverified integration. |

---

## 5. Key technical decisions & why

Two real bugs, found by actually running the pipeline rather than reading it back:

- **Chroma's default relevance score breaks for non-OpenAI-scale embeddings.**
  LangChain's distance→relevance conversion defaults to assuming OpenAI's
  embedding scale, even though nothing here uses OpenAI — with the local MiniLM
  embeddings, that default silently produced scores like `-538.1`, which would
  have made the guardrail threshold meaningless. Fix: `collection_metadata=
  {"hnsw:space": "cosine"}` in `app/ingest.py`, so LangChain uses
  `1 - cosine_distance` instead — sane regardless of embedding model.
- **Re-ingesting the same files duplicated chunks.** No explicit IDs meant every
  ingest run got fresh UUIDs, so re-running on unchanged files (redeploys, retries)
  piled up duplicates — caught when the same chunk showed up 4/4 times in a
  retrieval while taking screenshots. Fix: deterministic `sha256(source+chunk_index+content)`
  IDs passed to `add_documents(ids=...)`, so Chroma's upsert makes it idempotent.

Other decisions:
- `RagChain(llm=..., vectorstore=...)` — constructor injection instead of global
  singletons, so the chain logic can be exercised with a fake LLM/store later
  without an API key or a real index, even though no test suite exists yet.
- No-context guardrail short-circuits *before* the LLM is called — cheaper and
  doesn't depend on the model actually obeying "say you don't know."
- Model choice lives behind `get_llm()`/`get_embeddings()` factory functions and
  env vars (`LLM_MODEL`, `EMBEDDING_MODEL`), not hardcoded — swapping to a
  different Ollama or `sentence-transformers` model is a `.env` change, not a
  code change.

---

## 6. Engineering standards

**Followed:** type hints + `pydantic-settings` for config; dependency injection kept
in the design even without tests to exercise it yet; `.env.example` committed,
`.env` gitignored; structured logging, not `print`; docstrings only where the *why*
isn't obvious.

**Skipped, deliberately — kept this simple, and what I'd add first if it grew up:**
- No automated tests, no CI pipeline.
- No Docker/containerization — just a venv and `pip install`.
- No lint/format enforcement (`ruff`/`black` not wired up).
- No auth or rate limiting on the API — fine locally, not for the internet.
- No streaming responses (`/chat` is request/response, not SSE).
- No automated retrieval/answer-quality eval — manual testing only (see [§8](#8-whatd-i-do-differently)).
- No PII/content-moderation/prompt-injection guardrail on uploaded document content.

---

## 7. How I used AI tools

Used Claude Code as a pair-programmer for the mechanical parts: LangChain
loaders/chain/ingest scaffolding and FastAPI/Streamlit boilerplate — code whose
shape is well-established and not worth hand-typing.

Not outsourced: the decisions in [§4](#4-ragllm-approach--decisions)/[§5](#5-key-technical-decisions--why)
and what to cut in [§6](#6-engineering-standards) — those are judgment calls, which is the
point of a take-home. I didn't trust generated code blindly either: I ran the full
pipeline against a real vector store and a real (local) LLM specifically to catch
what only shows up when you execute it, which is how I found both bugs in [§5](#5-key-technical-decisions--why) —
neither was visible from reading the code.

The screenshots in [§9](#9-screenshots) are from a real running instance, including
one follow-up question that produced a wrong "I don't know" — kept in rather than
re-shot with a cleaner example, because it's a real, honest limitation (see [§8](#8-whatd-i-do-differently)).

---

## 8. What'd I do differently

- **A test suite.** Cut entirely to keep this simple — `RagChain` already takes
  `llm`/`vectorstore` as constructor args specifically so it *could* be tested
  with a fake model, deterministically, I just didn't write the tests. First
  thing I'd add back.
- **Hybrid search** — pure embedding similarity misses exact-match signals (model
  numbers, section numbers, exact figures); add a BM25 pass alongside vectors.
- **Reranking** — over-fetch top-15, cross-encoder rerank down to the best 4,
  instead of trusting raw cosine ordering.
- **Fix the condense step's silent failure on a small model.** With `qwen3:1.7b`,
  a follow-up ("what about after two years?") after a PTO question came back
  unrewritten instead of condensed, so retrieval found nothing (see [§9](#9-screenshots)
  screenshot). A larger model would likely follow the rewrite instruction more
  reliably, but the chain currently trusts whatever LLM is configured to follow it
  unconditionally. I'd add a deterministic fallback: if the rewrite comes back
  identical to the input and history exists, append the last turn to the
  retrieval query instead of trusting the rewrite blindly.
- **Section-aware chunking** — fixed-size splitting sometimes cuts mid-sentence;
  markdown-header-aware splitting would respect actual document structure.
- **An automated eval harness** — a golden (question, source, facts) set scored on
  retrieval precision@k and answer faithfulness (LLM-judge or RAGAS) in CI, instead
  of "I tried some questions and read the answers."
- **Streaming responses**, for real chat UX.
- **The production hardening in [§3](#3-productionizing-on-a-hyperscaler)** — none of which makes sense to
  half-build for a local assessment repo, but all real work before real users.

---

## 9. Screenshots

Screenshots can be found in the docs/screenshots folder.

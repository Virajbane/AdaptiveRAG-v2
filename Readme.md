# RAG 2.0 — Adaptive Retrieval-Augmented Generation System

An enterprise-style, multi-agent RAG system that doesn't just search documents — it **decides where to look first, checks its own retrieval, and refuses to guess when it isn't sure.** A planner agent analyzes each question, a retriever agent runs hybrid (vector + keyword) search, a grader agent filters out weak evidence before it ever reaches the LLM, and a critic agent double-checks the final answer for hallucination — all running on a local, self-hosted LLM via Ollama.

Built end-to-end: authentication, a Docling-powered document ingestion pipeline, hybrid search with cross-encoder reranking, multi-agent orchestration with deterministic safety nets, memory, security hardening, automated testing, and a CI/CD pipeline that tests, containerizes, and deploys on every push.

---

## Why this exists

Most RAG demos are a single retrieval step bolted onto an LLM call. This project instead models RAG as a small **pipeline of specialized agents**, each with one job, coordinated by an orchestrator — closer to how a production system would actually be structured.

It's also **local-first**: the default LLM and embedding model both run on [Ollama](https://ollama.com) (`qwen2.5:7b` + `nomic-embed-text`), so the core system works without any external API key.

The thing that sets this apart from a typical student RAG project isn't the pipeline shape — it's what happens when the pipeline *doesn't* work correctly. Small local LLMs misbehave in specific, repeatable ways (dropping the word "my" from a rewrite, hallucinating a plausible-sounding but wrong number, rubber-stamping their own hallucinations). Rather than trusting prompt instructions alone, every agent that can silently fail has a **deterministic, code-level backstop** layered on top of the LLM call — documented in the code as it was found, fixed, and re-broken by a harder edge case. See [Multi-agent system](#multi-agent-system) below for the actual mechanisms.

---

## Architecture

```
                         ┌─────────────────────┐
                         │   Next.js Frontend    │
                         │  (chat, docs, auth)   │
                         └──────────┬────────────┘
                                    │ HTTP / JWT
                                    ▼
                         ┌─────────────────────┐
                         │   FastAPI Backend     │
                         │ (auth, rate-limit,    │
                         │  security headers)    │
                         └──────────┬────────────┘
                                    ▼
              ┌──────────────────────────────────────────┐
              │            Agent Orchestrator (LangGraph)  │
              │                                             │
              │  Rewriter ──▶ Planner ∥ Retriever ──▶       │
              │  Grader ──▶ Tool Agent (web/calc) ──▶       │
              │  Answer ──▶ Critic (retry loop)             │
              │                                             │
              └──────┬───────────────┬───────────────┬──────┘
                     ▼               ▼               ▼
            ┌────────────┐  ┌──────────────┐  ┌────────────┐
            │   Qdrant    │  │   MongoDB     │  │   Redis     │
            │ (vectors,   │  │ (docs, chat,  │  │ (sessions,  │
            │  per-user   │  │  metadata,    │  │  short-term │
            │  namespace) │  │  long-term    │  │  memory,    │
            │             │  │  memory)      │  │  query      │
            │             │  │               │  │  cache)     │
            └─────────────┘  └──────────────┘  └────────────┘
                     ▲
                     │ embeddings + generation
              ┌──────┴──────┐
              │    Ollama     │
              │ (qwen2.5:7b,  │
              │  nomic-embed) │
              └───────────────┘
```

> **Note on this diagram vs. `docker-compose.yml`:** the compose file at the project root provisions the three stateful infra services — MongoDB, Redis, Qdrant. Ollama, the FastAPI backend, and the Next.js frontend run as separate host processes in local dev (see [Getting started](#getting-started)); the production deploy containerizes the backend separately with its own Dockerfile.

### Request flow

1. A user message hits `POST /agents/chat`.
2. **Rewriter Agent** resolves conversational context ("what about X?" → full standalone question) and fixes typos, using short-term memory — but only if the rewrite survives five separate correctness checks (see below). `state.question` is never mutated; the rewrite lives in a separate field so the original is always available as a fallback.
3. **Planner Agent** decides which sources are needed (`documents`, `web`, `tools`) — running in **parallel** with the **Retriever Agent**, which always performs hybrid search regardless of what the Planner decides, so the results are ready the instant they're needed.
4. **Grader Agent** drops chunks that are noise relative to the best match, and can reject the batch entirely (skipping generation) if even the top match is too weak to trust.
5. **Tool Agent** calls web search or a sandboxed calculator if the plan calls for it.
6. **Answer Agent** generates a response strictly from the sources the Planner actually asked for, with every source traced back to a real retrieved chunk or web result — never invented.
7. **Critic Agent** validates groundedness. If it fails, a surgical **answer-only retry** runs (not the whole pipeline). A grounding backstop also catches cases where the judge model itself gets it wrong in *either* direction — see below.
8. The result is cached in Redis, keyed for repeat queries.

---

## Multi-agent system

Each agent inherits from `BaseAgent`, which wraps `_execute()` in error handling so one agent's failure doesn't crash the graph — errors are recorded on `state.error` instead of raised. Below is what each agent actually does internally, not just its one-line job description.

### 1. Rewriter Agent
**File:** `app/agents/rewriter.py`

Runs first, unconditionally (no skip-heuristic — a naive spellcheck would misfire on domain terms like "Qdrant"). Combines context resolution and typo normalization in one fast-model call, then validates the result against **five independent deterministic checks** before trusting it:

| Check | Catches |
|---|---|
| `_dropped_personal_reference` | Rewrite silently drops "my"/"I"/"me" — breaks the Planner's routing signal for personal-content questions |
| `_dropped_quantifier` | Rewrite silently drops a numeral ("two techniques" → "the technique") |
| `_dropped_or_altered_acronym` | Rewrite expands or alters an acronym the prompt explicitly forbids touching (e.g. "FNOL" → a hallucinated expansion) |
| `_diverged_too_much` | Rewrite replaces the question's actual content with a different (but related) question — via `SequenceMatcher` string similarity |
| `_matches_prior_turn` | Rewrite regurgitates a *prior* user or assistant turn instead of the current question — via paraphrase-tolerant word-overlap comparison, checked *relative to* the current question so legitimate follow-up resolution isn't penalized |

Any single failure discards the rewrite; downstream agents fall back to the literal `state.question`. This list grew one bug at a time — each check exists because a specific, logged failure got past the prompt instructions alone.

### 2. Planner Agent
**File:** `app/agents/planner.py`

Classifies each question into `sources_needed` (`documents` / `web` / `tools`) via an LLM call, then applies **deterministic overrides** on top, because the LLM classifier has repeatedly misrouted in specific, recurring ways:

- **Personal-only override** — "my"/"I"/"me" with no comparison signal → forces `documents` only.
- **Needs-both override** — personal reference *and* a comparison/benchmark signal (e.g. "how does my resume compare to industry standards") → ensures `documents` isn't dropped.
- **Uploaded-doc-reference override** — "this paper" / "the document" with no pronoun at all → forces `documents`, since these were being misrouted to web search.
- **Metadata short-circuit** — title/author questions bypass retrieval entirely and answer directly from metadata extracted at ingestion time, since a title literally never contains the word "title" and can't be found by embedding similarity no matter how retrieval is tuned.

All routing overrides check the **original** question text, not the rewritten one — a paraphrase-for-recall rewrite is genuinely useful for retrieval, but was found to accidentally erase or invent the exact keywords these regex backstops key on.

### 3. Retriever Agent
**File:** `app/agents/retriever.py`

Always runs, in parallel with the Planner (the orchestrator decides whether to *use* its output afterward). Two-step process:

1. `document_resolver.py` fuzzy-matches the question against the user's uploaded filenames (token overlap + sequence similarity) to optionally scope search to a single document — e.g. "summarize the resume" only searches that file instead of pooling every document the user has ever uploaded.
2. `HybridSearchEngine.search()` runs the actual retrieval (see below), then batch-attaches real filenames to every result in a single Mongo query, so citations show a real document name instead of a generic "Source N".

### 4. Grader Agent
**File:** `app/agents/grader.py`

No LLM call — pure post-retrieval filtering, reusing scores already computed during retrieval. Two independent thresholds:

- **Relative floor (2%)** — drops chunks scoring below 2% of the *top* chunk's score *within this result set*, adapting per-query instead of using one fixed cutoff. Always keeps at least 3 chunks even if they technically fail this, so multi-hop questions aren't starved down to one chunk.
- **Absolute floor (0.05)** — catches the case where the *entire batch* is weak (e.g. every retrieved chunk came from a References section, not the actual answer) — something a relative-only check can't see, since a batch of five equally-bad chunks all pass a relative comparison against each other. Below this floor, generation is skipped entirely and a direct "not found" response is returned — faster and strictly safer than letting the LLM see noise and hope it declines gracefully.
- **Bidirectional source override** — if document confidence is high enough (`top_score ≥ 0.5`), `web` is actively *removed* from `sources_needed` to stop irrelevant web noise from diluting a strong in-corpus answer. A second, lower threshold (`0.15`) adds `documents` back in when the Planner missed real (but moderately-scored) document evidence — two different thresholds because "should this source be included at all" and "should this source stand alone" turned out to need different bars.

### 5. Answer Agent
**File:** `app/agents/answer.py`

Builds context strictly from what the Planner actually requested (`docs_wanted = "documents" in sources_needed`) — earlier versions always included document chunks regardless of routing, which meant a purely web-routed question got 5 irrelevant document chunks mixed into its context. After generation, a **numeric-claims grounding check** (`_numeric_claims_grounded`) verifies every number in the answer appears verbatim somewhere in the actual retrieved context; if not, the answer is discarded in favor of an honest "I couldn't verify a specific number" response rather than shipping an unverified figure. Final `confidence_final` is a blend of Planner confidence and the BGE rerank score (not the RRF fusion score, which is rank-based and clusters too tightly to carry real signal).

### 6. Critic Agent
**File:** `app/agents/critic.py`

An LLM judge validates the final answer for groundedness — but the judge itself turned out to be unreliable in *both* directions, so it's backstopped by a deterministic **grounding score** (do the answer's numbers/percentages/proper nouns actually appear in the retrieved context?):

- **Unexplained-rejection override** — if the judge rejects an answer with 0 confidence and *no stated reason*, and the deterministic grounding score is high (≥0.8), the rejection is overridden. An unexplained blanket "invalid" looks like judge noise, not a genuine catch — genuine catches name a specific problem.
- **Overconfident-acceptance override** — the more dangerous mirror case: the judge approves an answer at high confidence while the deterministic grounding score is near zero (a benchmark or figure that appears nowhere in the retrieved context). This is treated as a likely rubber-stamped hallucination and force-rejected regardless of what the judge said.

### 7. Tool Agent
Calls web search or a sandboxed calculator when the Planner's routing includes `tools`/`web`. Results are merged into Answer Agent's context and surfaced as real, separately-labeled sources.

### 8. Orchestrator
A LangGraph `StateGraph` wiring the above into a single `AgentState` object, with a conditional edge on Critic's `is_valid` that drives an answer-only retry loop (not a full pipeline restart) on failure.

---

## Document ingestion pipeline

**File:** `app/services/document/processor.py`

```
Upload
  │
  ▼
PDF? ─── yes ──▶ Docling (structure-aware parsing: headings, tables, layout)
  │no
  ▼
PyMuPDF / plain parser (DOCX, TXT, CSV)
  │
  ▼
Metadata extraction (title/authors) — from raw opening text, before chunking
  │
  ▼
Chunking — Docling's heading/table-scoped chunker for PDFs, recursive splitter otherwise
  │
  ▼
BM25 keyword indexing  +  Ollama embeddings (nomic-embed-text)
  │
  ▼
Vector storage in Qdrant (per-user namespace)
```

PDFs specifically moved from PyMuPDF + regex table detection to **Docling** after eval work surfaced two concrete failure classes: dense numeric tables losing their column labels, and chunks bleeding across section boundaries (diluting embeddings enough to drop a relevant chunk out of the top-20 vector search). DOCX/TXT/CSV stay on the lighter-weight parser — Docling's model weight (torch/torchvision layout + table models) is only worth paying where it measurably fixes a problem.

---

## Hybrid retrieval engine

**File:** `app/services/retrieval/hybrid_search.py`

- **Dense retrieval** — Qdrant vector search over `nomic-embed-text` embeddings.
- **Sparse retrieval** — BM25 keyword search, run in parallel with vector search.
- **Fusion** — Reciprocal Rank Fusion: `score(d) = Σ 1 / (60 + rank(d))` across both result lists. Rank-based, not raw-score-based, so it's immune to BM25 and cosine-similarity living on completely different numeric scales — the industry-standard approach for hybrid RAG.
- **Reranking** — the fused top candidates are reranked with a BGE cross-encoder for the final ordering, when available.

## Document-aware retrieval

**File:** `app/services/retrieval/document_resolver.py`

Fuzzy-matches a question against the user's actual uploaded filenames (normalized: strip extension, lowercase, split on any non-alphanumeric run) using both token-overlap and sequence-similarity scoring, so a query like "summarize the resume" resolves to the one relevant file instead of pooling and diluting against every document the user has ever uploaded. Stays a *soft* signal — below the match threshold, it returns `None` and search falls back to searching across all of the user's documents, exactly as before.

---

## Tech stack

| Layer | Technology | Notes |
|---|---|---|
| Frontend | Next.js 15 (JSX) | Chat, document upload, search, memory, settings dashboards |
| Backend | FastAPI (Python 3.12) | Async throughout |
| LLM | Ollama — `qwen2.5:7b` | Local, no API key required |
| Embeddings | Ollama — `nomic-embed-text` | 768-dim vectors |
| Document parsing | Docling (PDF), PyMuPDF (other formats) | Structure-aware parsing + heading/table-scoped chunking for PDFs |
| Vector DB | Qdrant | Per-user namespace isolation |
| Document/chat store | MongoDB | Users, documents, chat sessions, long-term memory |
| Cache / short-term memory | Redis | Session history, query cache |
| Keyword search | BM25 (`rank_bm25`, BM25Plus variant) | Fused with vector scores via RRF |
| Reranking | BGE cross-encoder | Final ordering after RRF fusion |
| Orchestration | LangGraph + LangChain | `StateGraph` with conditional Critic retry loop |
| Auth | JWT (PyJWT) + bcrypt | Rate-limited, validated, audit-logged |
| Testing | pytest + pytest-asyncio | 131 backend unit tests |
| CI/CD | GitHub Actions | Test → build Docker images → push to GHCR → deploy via SSH |
| Containerization | Docker / Docker Compose | Infra services (Mongo/Redis/Qdrant) via compose; backend containerized separately for prod |
| Reverse proxy | nginx | Rate limiting, SSL termination, security headers (production) |

---

## Installation

### Prerequisites

- Docker Desktop (with WSL2 backend on Windows)
- Python 3.12
- Node.js 20+
- Ollama installed natively — [ollama.com](https://ollama.com)

### 1. Clone the repo

```powershell
git clone <your-repo-url>
cd rag-2.0-system
```

### 2. Set up environment variables

Copy the example files and fill in real values:

```powershell
copy .env.example .env
copy backend\.env.example backend\.env
```

Edit both `.env` files. `MONGO_PASSWORD`, `REDIS_PASSWORD`, and `QDRANT_API_KEY` in the root `.env` must match the corresponding `MONGODB_URL` / `REDIS_URL` / `QDRANT_API_KEY` values in `backend\.env` exactly — the root file configures the containers, the backend file configures how the app connects to them.

### 3. Start infrastructure (MongoDB, Qdrant, Redis)

```powershell
docker compose up -d
docker compose ps
```

Confirm `mongodb` and `redis` show `(healthy)` before continuing.

### 4. Start Ollama and pull required models

```powershell
ollama serve
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 5. Set up and run the backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

First-time note: Docling will download its layout and table-structure models (`docling-layout-heron`, `docling-models`) on first PDF upload — this can take a minute or two the very first time.

Backend runs at `http://localhost:8000`.

### 6. Set up and run the frontend

In a new terminal:

```powershell
cd frontend
echo NEXT_PUBLIC_API_URL=http://localhost:8000 > .env.local
npm install
npm run dev
```

Frontend runs at `http://localhost:3000`.

### 7. Verify

Open `http://localhost:3000`, upload a test PDF, and confirm it processes without errors.

### Where things live

- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Qdrant dashboard: http://localhost:6333/dashboard

### Run backend tests

```powershell
cd backend
pytest tests/unit/ -v
```

---

## CI/CD pipeline

Every push to `main` runs through `.github/workflows/deploy.yml`:

1. **`test-backend`** — spins up throwaway MongoDB + Redis containers, runs the full pytest suite
2. **`test-frontend`** — installs, lints, and production-builds the Next.js app
3. **`build-images`** *(only if both above pass, only on `main`)* — builds backend and frontend Docker images, pushes to GitHub Container Registry (`ghcr.io`)
4. **`deploy`** *(only if build succeeds)* — SSHes into the production server, pulls the new images, restarts containers, runs a smoke test against `/health`

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for real issues hit while building this pipeline and how they were diagnosed and fixed.

---

## Project status

Built in sequential phases, each with its own deliverables and tests:

| Phase | Area | Status |
|---|---|---|
| 1 | Infrastructure (Docker, skeleton apps) | ✅ Complete |
| 2 | Authentication (JWT, register/login) | ✅ Complete |
| 3 | Document upload, parsing, chunking, embedding | ✅ Complete |
| 4 | Hybrid search (BM25 + vector) | ✅ Complete |
| 5 | Multi-agent system + orchestrator | ✅ Complete |
| 6 | Memory (short-term Redis, long-term MongoDB) | ✅ Complete |
| 7 | External tools (web search, calculator) | ✅ Complete |
| 8 | Security hardening (rate limiting, headers, validation, audit logs) | ✅ Complete |
| 9 | Production Docker setup | ✅ Complete |
| 10 | Unit tests (131 tests across agents, auth, retrieval, parsers) | ✅ Complete |
| 11 | Performance (parallel agent execution, query cache, profiler) | ✅ Complete |
| 12 | CI/CD pipeline (GitHub Actions, GHCR, automated deploy) | ✅ Complete |
| 13 | Documentation | 🚧 In progress |
| 14 | Hybrid search upgrade (RRF fusion, BGE reranker wiring, Redis-backed cache) | ✅ Complete |
| 15 | Docling-based PDF ingestion (structure-aware parsing/chunking) | ✅ Complete |
| 16 | Structured evaluation harness (Ragas + DeepEval + golden test set) | 🚧 In progress |

### Known gaps

- **Numeric adjacent-value fabrication** — the numeric-claims grounding check confirms a number appears *somewhere* in context, but doesn't yet verify it's attributed to the *right entity*; cross-entity numeric questions can still pick a nearby correct-looking number from the same chunk and attribute it wrongly. Actively being worked on in the eval harness.
- **Query cache key uses raw query text**, not the rewritten question — semantically identical questions phrased differently miss the cache.
- **Production deploy target** — the `deploy` CI job is fully written but requires a provisioned server and GitHub Secrets (`PROD_HOST`, `PROD_USER`, `PROD_SSH_KEY`, `API_URL`) to actually run.

These are tracked deliberately, not hidden — see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) and the test suite's own documented "known gap" tests for specifics.

---

## Performance

- Cold query (no cache): ~28s end-to-end through the full agent pipeline
- Cached query (repeat question): ~2s
- Uncached query after parallelizing Planner + Retriever: ~12s

---

## License

*(Add your license here — MIT is a common default for portfolio projects.)*
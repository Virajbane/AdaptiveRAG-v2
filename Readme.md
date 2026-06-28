# RAG 2.0 — Adaptive Retrieval-Augmented Generation System

An enterprise-style, multi-agent RAG (Retrieval-Augmented Generation) system that doesn't just search documents — it **decides where to look first**. A planner agent analyzes each question, a retriever agent runs hybrid (vector + keyword) search, a tool agent reaches for the web or a calculator when needed, and an answer agent produces a cited response — all running on a local, self-hosted LLM.

Built end-to-end: authentication, document ingestion pipeline, hybrid search, multi-agent orchestration, memory, security hardening, automated testing, and a CI/CD pipeline that tests, containerizes, and deploys on every push.

---

## Why this exists

Most RAG demos are a single retrieval step bolted onto an LLM call. This project instead models RAG as a small **pipeline of specialized agents**, each with one job, coordinated by an orchestrator — closer to how a production system would actually be structured. It's also fully local-first: the default LLM and embedding model both run on [Ollama](https://ollama.com), so the system works without any external API key.

---

## Architecture

```
                         ┌─────────────────────┐
                         │   Next.js Frontend   │
                         │  (chat, docs, auth)  │
                         └──────────┬───────────┘
                                    │ HTTP / JWT
                                    ▼
                         ┌─────────────────────┐
                         │   FastAPI Backend    │
                         │ (auth, rate-limit,   │
                         │  security headers)   │
                         └──────────┬───────────┘
                                    ▼
              ┌─────────────────────────────────────────┐
              │            Agent Orchestrator             │
              │                                            │
              │   Planner ──▶ Retriever (parallel) ──▶    │
              │       Tool Agent (web/calc) ──▶ Answer     │
              │                                            │
              └──────┬───────────────┬──────────────┬──────┘
                     ▼               ▼              ▼
            ┌────────────┐  ┌──────────────┐  ┌───────────┐
            │   Qdrant    │  │   MongoDB    │  │   Redis    │
            │ (vectors)   │  │ (docs, chat, │  │ (sessions, │
            │             │  │  long-term   │  │  query     │
            │             │  │  memory)     │  │  cache)    │
            └────────────┘  └──────────────┘  └───────────┘
                     ▲
                     │ embeddings
              ┌──────┴──────┐
              │    Ollama     │
              │ (qwen2.5:7b,  │
              │ nomic-embed)  │
              └─────────────┘
```

**Request flow:** A user message hits the FastAPI `/agents/chat` endpoint → the **Planner Agent** decides which sources are needed (documents? web? both?) → the **Retriever Agent** runs hybrid search (BM25 keyword + Qdrant vector search, merged and weighted) in parallel with planning → the **Tool Agent** calls external tools (web search via Tavily, a sandboxed calculator) if the plan calls for it → the **Answer Agent** generates a final response using only the top retrieved chunks, with sources traced back to real retrieved documents (never invented by the LLM) → the result is cached in-memory for repeat queries.

---

## Tech stack

| Layer | Technology | Notes |
|---|---|---|
| Frontend | Next.js 15 (JSX) | Chat, document upload, search, memory, settings dashboards |
| Backend | FastAPI (Python 3.12) | Async throughout |
| LLM | Ollama — `qwen2.5:7b` | Local, no API key required |
| Embeddings | Ollama — `nomic-embed-text` | 768-dim vectors |
| Vector DB | Qdrant | Per-user namespace isolation |
| Document/chat store | MongoDB | Users, documents, chat sessions, long-term memory |
| Cache / short-term memory | Redis | Session history, in-memory query cache (Redis migration planned, see Roadmap) |
| Keyword search | BM25 (`rank_bm25`, BM25Plus variant) | Combined with vector scores in hybrid search |
| Auth | JWT (PyJWT) + bcrypt | Rate-limited, validated, audit-logged |
| Testing | pytest + pytest-asyncio | 131 backend unit tests |
| CI/CD | GitHub Actions | Test → build Docker images → push to GHCR → deploy via SSH |
| Containerization | Docker / Docker Compose | Separate dev and production compose files |
| Reverse proxy | nginx | Rate limiting, SSL termination, security headers (production) |

---

## Multi-agent system

| Agent | Responsibility |
|---|---|
| **Planner** | Reads the question, decides which sources are needed (`documents`, `web`, `tools`), sets a confidence level |
| **Retriever** | Runs hybrid search (vector + BM25 keyword) against the user's own document index, in parallel with the Planner |
| **Tool Agent** | Calls web search or the calculator tool when the plan requires it |
| **Critic** | Validates groundedness and flags hallucination risk *(built, not yet wired into the live orchestration path — see Known Gaps)* |
| **Answer** | Generates the final response strictly from the top retrieved chunks, with citations traced to real source documents |

All agents share a single `AgentState` object, run through a `BaseAgent.run()` wrapper that catches and records errors per-agent without crashing the whole pipeline.

---

## Getting started

### Prerequisites
- Docker + Docker Compose
- Python 3.12 (for running tests outside Docker)
- Node.js 20+ (for frontend dev outside Docker)

### Run locally

```powershell
git clone https://github.com/Virajbane/AdaptiveRAG-v2.git
cd AdaptiveRAG-v2
cp .env.example .env   # fill in MONGO_PASSWORD, REDIS_PASSWORD, JWT_SECRET_KEY, etc.
docker compose up -d
```

Once the containers are healthy:
```powershell
# Pull the local LLM models (first run only — this downloads several GB)
docker exec rag-ollama ollama pull qwen2.5:7b
docker exec rag-ollama ollama pull nomic-embed-text
```

- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Run backend tests

```powershell
cd backend
pip install -r requirements.txt
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

This project was built in 12 sequential phases, each with its own deliverables and tests:

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
| 14 | Further optimization (Redis-backed cache, reranker wiring, DB indexes) | ⏳ Planned |

### Known gaps

- **Query cache is in-memory only** — lost on server restart; planned migration to Redis-backed cache (Phase 14)
- **BGE reranker** (`backend/app/services/retrieval/reranker.py`) is implemented but not yet wired into `hybrid_search.py`
- **CriticAgent** is implemented and unit-tested in isolation, but not yet called inside the live orchestrator flow
- **Production deploy target** — the `deploy` CI job is fully written but requires a provisioned server and GitHub Secrets (`PROD_HOST`, `PROD_USER`, `PROD_SSH_KEY`, `API_URL`) to actually run

These are tracked deliberately, not hidden — see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) and the test suite's own documented "known gap" tests for specifics.

---

## Performance

From Phase 11 optimization work:
- Cold query (no cache): ~28s end-to-end through the full agent pipeline
- Cached query (repeat question): ~2s
- New, uncached query after parallelizing Planner + Retriever: ~12s

---

## License

*(Add your license here — MIT is a common default for portfolio projects.)*
# RAG 2.0 — Adaptive Retrieval-Augmented Generation System

An enterprise-style, multi-agent RAG system that doesn't just search documents — it **decides where to look first, checks its own retrieval, and refuses to guess when it isn't sure.**

A planner agent analyzes each question, a retriever agent runs hybrid (vector + keyword) search, a grader agent filters out weak evidence before it ever reaches the LLM, and a critic agent double-checks the final answer for hallucination — all running on a local, self-hosted LLM via Ollama.

Built end-to-end: authentication, a Docling-powered document ingestion pipeline, hybrid search with cross-encoder reranking, multi-agent orchestration with deterministic safety nets, memory, security hardening, automated testing, and a CI/CD pipeline that tests, containerizes, and deploys on every push.

This README is written for someone who has **never seen this repository before**. Follow it top to bottom on first setup — the order of steps matters, especially around Docker and environment variables.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture](#architecture)
3. [Folder Structure](#folder-structure)
4. [Environment Variables](#environment-variables)
5. [Installation (Step-by-Step)](#installation-step-by-step)
6. [Service Health Verification](#service-health-verification)
7. [Running the Project](#running-the-project)
8. [Smoke Test](#smoke-test)
9. [Troubleshooting](#troubleshooting)
10. [Verification Commands Cheat Sheet](#verification-commands-cheat-sheet)
11. [Production Deployment](#production-deployment)
12. [Development Workflow](#development-workflow)
13. [FAQ](#faq)
14. [Contributing](#contributing)
15. [Project Status & Known Gaps](#project-status--known-gaps)
16. [License](#license)

---

## Prerequisites

Install these **before** cloning the repo. Version mismatches are the #1 cause of first-time setup failures.

| Requirement | Minimum Version | Notes |
|---|---|---|
| **Docker Desktop** | Latest stable | On Windows, must run with the **WSL2 backend** enabled (Settings → General → "Use the WSL 2 based engine") |
| **Docker Compose** | v2 (bundled with Docker Desktop) | Verify with `docker compose version` |
| **Python** | 3.12 | Backend will not install correctly on older versions |
| **Node.js** | 20+ | Verify with `node -v` |
| **Ollama** | Latest | Installed **natively on the host**, not in Docker — [ollama.com](https://ollama.com) |
| **Git** | Any recent version | |

**Supported OS:** Windows 10/11 (with WSL2), macOS (Intel or Apple Silicon), Linux. All commands in this README are given in **PowerShell** syntax for Windows; macOS/Linux users should use the equivalent bash commands (noted where they differ meaningfully).

Check your versions before proceeding:

```powershell
docker --version
docker compose version
python --version
node --version
ollama --version
```

---

## Architecture

```
                    ┌───────────────────────┐
                    │   Next.js Frontend      │
                    │  (chat, docs, auth)     │
                    └───────────┬─────────────┘
                                │ HTTP / JWT
                                ▼
                    ┌───────────────────────┐
                    │   FastAPI Backend       │
                    │ (auth, rate-limit,      │
                    │  security headers)      │
                    └───────────┬─────────────┘
                                ▼
          ┌────────────────────────────────────────┐
          │       Agent Orchestrator (LangGraph)     │
          │                                            │
          │  Rewriter → Planner ∥ Retriever →         │
          │  Grader → Tool Agent (web/calc) →         │
          │  Answer → Critic (retry loop)              │
          └──────┬────────────┬────────────┬──────────┘
                 ▼            ▼            ▼
          ┌───────────┐ ┌───────────┐ ┌───────────┐
          │  Qdrant     │ │  MongoDB    │ │  Redis      │
          │  (vectors,  │ │ (docs,      │ │ (sessions,  │
          │  per-user   │ │  chat,      │ │  short-term │
          │  namespace) │ │  long-term  │ │  memory,    │
          │             │ │  memory)    │ │  query      │
          │             │ │             │ │  cache)     │
          └─────────────┘ └─────────────┘ └─────────────┘
                 ▲
                 │ embeddings + generation
          ┌──────┴──────┐
          │   Ollama      │
          │ (qwen2.5:7b,  │
          │ nomic-embed)  │
          └───────────────┘
```

### What each service does

| Service | Role | Runs in Docker? |
|---|---|---|
| **Next.js Frontend** | Chat UI, document upload, auth screens, settings | No — runs as a host `npm` process in dev |
| **FastAPI Backend** | Auth, rate limiting, agent orchestration, API endpoints | No in dev (runs via `uvicorn`); yes in production (own Dockerfile) |
| **MongoDB** | Stores users, uploaded documents, chat sessions, long-term memory | Yes, via `docker-compose.yml` |
| **Redis** | Session cache, short-term conversational memory, query result cache | Yes, via `docker-compose.yml` |
| **Qdrant** | Vector database — per-user namespaced document embeddings | Yes, via `docker-compose.yml` |
| **Ollama** | Local LLM (`qwen2.5:7b`) and embedding model (`nomic-embed-text`) | No — installed natively on the host, not containerized |

**Why Ollama isn't in `docker-compose.yml`:** running an LLM inside Docker on most laptops loses GPU passthrough performance. Ollama is installed and run natively; everything else that's stateful infrastructure (Mongo/Redis/Qdrant) is containerized.

---

## Folder Structure

```
rag-2.0-system/
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── agents/           # Rewriter, Planner, Retriever, Grader, Answer, Critic, Tool
│   │   ├── services/
│   │   │   ├── document/     # Ingestion pipeline (processor.py)
│   │   │   └── retrieval/    # hybrid_search.py, document_resolver.py
│   │   └── main.py
│   ├── tests/unit/           # 131 pytest unit tests
│   ├── requirements.txt
│   └── .env.example          # Backend environment template
├── frontend/                 # Next.js 15 application
│   ├── package.json
│   └── .env.local            # Frontend environment (you create this)
├── docker/                   # Dockerfiles for production images
├── docs/                     # Additional documentation
├── .github/workflows/
│   └── deploy.yml            # CI/CD pipeline
├── docker-compose.yml        # Provisions MongoDB, Redis, Qdrant
├── .env.example               # Root environment template (Docker container config)
├── TROUBLESHOOTING.md
└── README.md                  # This file
```

**There are three separate `.env` files.** Mixing them up is the single most common setup mistake — see the next section.

---

## Environment Variables

This project has **three separate environment files**, each read by a different process. They are not interchangeable, and two of the values inside them **must match exactly** across files.

| File | Read by | Purpose |
|---|---|---|
| `.env` (repo root) | `docker compose` only | Configures the MongoDB / Redis / Qdrant **containers** themselves (initial credentials, ports) |
| `backend/.env` | The FastAPI backend (`uvicorn`) | Configures how the **application** connects to those containers, plus app-level secrets |
| `frontend/.env.local` | The Next.js frontend (`npm run dev`) | Tells the frontend where the backend API lives |

Create them from the provided templates:

```powershell
copy .env.example .env
copy backend\.env.example backend\.env
```

The frontend file has no template — you create it directly (see step 6 of Installation).

### ⚠️ Pre-Flight Check — Do This Before Running `docker compose up`

**If you change the password in the root `.env`, go update it in `backend/.env` too — this step gets skipped most often, and MongoDB is the one that breaks because of it.** Mongo will refuse to authenticate the backend the moment the two passwords disagree, and the error it throws doesn't make the cause obvious.

Before starting anything, open the root `.env` and `backend/.env` side by side and confirm these values are **identical** across both files. This check takes thirty seconds and prevents the majority of first-run failures:

- [ ] Mongo password in root `.env` matches the password embedded in `MONGODB_URL` in `backend/.env`
- [ ] Redis password in root `.env` matches the password embedded in `REDIS_URL` in `backend/.env`
- [ ] Qdrant API key in root `.env` matches `QDRANT_API_KEY` in `backend/.env` (or is blank in both)
- [ ] `JWT_SECRET_KEY` / `SECRET_KEY` is set to your own long random string, not left as the example placeholder
- [ ] `NEXT_PUBLIC_API_URL` in `frontend/.env.local` points at the backend URL you're actually running

If any of these are out of sync, fix the `.env` files **first** — don't start Docker and troubleshoot afterward. If you've already started Docker once with mismatched values and are now changing them, remember that MongoDB won't pick up a new password without a volume reset (see the [critical warning](#️-critical-mongodb-credentials-must-match--and-can-silently-go-stale) below).

### Root `.env` (read by Docker Compose only)

| Variable | Required? | Purpose |
|---|---|---|
| `MONGO_INITDB_ROOT_USERNAME` | Required | Username Docker uses to create the MongoDB root user **on first container initialization only** |
| `MONGO_INITDB_ROOT_PASSWORD` | Required | Password for that root user — see the **critical warning** below |
| `REDIS_PASSWORD` | Required | Password Redis is started with (`--requirepass`) |
| `QDRANT_API_KEY` | Optional | If set, enables `QDRANT__SERVICE__API_KEY` on the container, requiring authenticated requests. Leave blank for local dev with no auth. |
| `MONGO_PORT` / `REDIS_PORT` / `QDRANT_PORT` | Optional | Override default host ports if they're already in use |

**Example root `.env` (production-style):**

```env
# MongoDB
MONGO_PASSWORD=your_secure_mongo_password

# Redis
REDIS_PASSWORD=your_secure_redis_password

# Qdrant
QDRANT_API_KEY=your_qdrant_api_key

# JWT
JWT_SECRET_KEY=your_super_secret_jwt_key_min_32_chars

# Tavily Web Search
TAVILY_API_KEY=your_tavily_api_key

# Environment
ENVIRONMENT=production

# Frontend
NEXT_PUBLIC_API_URL=http://your-domain.com

# Logging
LOG_LEVEL=INFO
```

> ⚠️ Whatever names your local `docker-compose.yml` actually declares (e.g. `MONGO_INITDB_ROOT_PASSWORD` vs. `MONGO_PASSWORD`) are the names that matter — check the compose file itself and use its exact variable names. The values, not the names, are what must match `backend/.env`; a rename on one side with no corresponding rename on the other reproduces the exact "credentials don't match" failure described in the critical warning below.

### `backend/.env` (read by the FastAPI app)

| Variable | Required? | Purpose |
|---|---|---|
| `MONGODB_URL` | Required | Full Mongo connection string, e.g. `mongodb://<user>:<password>@localhost:27017`. `<user>`/`<password>` **must exactly match** `MONGO_INITDB_ROOT_USERNAME` / `MONGO_INITDB_ROOT_PASSWORD` from the root `.env` |
| `REDIS_URL` | Required | Redis connection string including password, e.g. `redis://:<password>@localhost:6379`. `<password>` must match root `.env`'s `REDIS_PASSWORD` |
| `QDRANT_URL` | Required | Usually `http://localhost:6333` |
| `QDRANT_API_KEY` | Optional | Required **only if** the root `.env` set `QDRANT_API_KEY`. If the container has a key configured and the backend doesn't send one, every request gets a `401 Unauthorized` — see [Qdrant Authentication](#qdrant-authentication) |
| `JWT_SECRET_KEY` | Required | Signs auth tokens. Use a long random string — never reuse the example value in production |
| `OLLAMA_BASE_URL` | Required | Usually `http://localhost:11434` |
| `OLLAMA_MODEL` | Optional | Defaults to `qwen2.5:7b` |
| `OLLAMA_EMBED_MODEL` | Optional | Defaults to `nomic-embed-text` |
| `WEB_SEARCH_API_KEY` | Optional | Only needed if the Tool Agent's web-search capability is enabled |

**Example `backend/.env` (development-style):**

```env
APP_NAME=RAG 2.0 System
DEBUG=True

MONGODB_URL=mongodb://admin:password123@localhost:27017/rag_db?authSource=admin
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
EMBEDDING_MODEL=nomic-embed-text

OPENAI_API_KEY=
GROQ_API_KEY=
ENABLE_PICTURE_DESCRIPTION=True
PICTURE_DESCRIPTION_MODEL=moondream

SECRET_KEY=change-this-to-a-long-random-string-for-real-deployment
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_HOURS=24

FRONTEND_URL=http://localhost:3000
TAVILY_API_KEY=

# Security
ENCRYPTION_KEY=your_encryption_key_min_32_chars
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
ENVIRONMENT=development
LOG_LEVEL=INFO
```

> ⚠️ Notice `MONGODB_URL` above embeds `admin:password123` — that username and password **must be exactly what the root `.env` sets** for MongoDB's root user. `password123` is a placeholder; replace it with your own password in **both** files, not just this one. The same applies to `SECRET_KEY`/`JWT_SECRET_KEY` — pick one real value and use it consistently wherever the config expects it.

### `frontend/.env.local` (read by Next.js)

| Variable | Required? | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Required | The backend's base URL, e.g. `http://localhost:8000`. Must be set **before** `npm run dev` — Next.js inlines `NEXT_PUBLIC_*` vars at build/start time |

### ⚠️ Critical: MongoDB credentials must match — and can silently go stale

`MONGO_INITDB_ROOT_USERNAME` / `MONGO_INITDB_ROOT_PASSWORD` (root `.env`) and the credentials embedded in `MONGODB_URL` (`backend/.env`) **must be identical**. If they diverge, the backend fails to authenticate against Mongo.

The trap: **MongoDB only creates the root user the first time the container initializes its data volume.** If you change `MONGO_INITDB_ROOT_PASSWORD` in `.env` *after* the container has already started once, nothing happens on `docker compose up -d` — the existing user in the persisted volume keeps the old password, and the backend (now pointed at the new password) fails to authenticate.

**Fix — recreate the MongoDB volume so initialization runs again:**

```powershell
docker compose down -v
docker compose up -d
```

`-v` removes the named volumes, which wipes all data in MongoDB, Redis, and Qdrant — this forces re-initialization with the current `.env` values. **Use it only when you've intentionally changed credentials and are fine losing local data** (this is normal during initial setup). Do **not** use `-v` on a running dev environment where you want to keep uploaded documents or chat history — in that case, either restore the original password in `.env`, or manually update the Mongo user's password via `mongosh` instead of wiping the volume.

```powershell
# Restart without wiping data (safe — use when credentials already match)
docker compose down
docker compose up -d
```

---

## Installation (Step-by-Step)

Follow this exact order. Skipping ahead (e.g. starting the backend before Docker containers are healthy) is the most common cause of confusing errors.

### 1. Clone the repository

```powershell
git clone <your-repo-url>
cd rag-2.0-system
```

### 2. Create environment files

```powershell
copy .env.example .env
copy backend\.env.example backend\.env
```

Edit both files with a text editor. Set matching Mongo/Redis credentials as described above. Then create the frontend file:

```powershell
echo NEXT_PUBLIC_API_URL=http://localhost:8000 > frontend\.env.local
```

### 3. Start Docker infrastructure

```powershell
docker compose up -d
```

### 4. Verify Docker containers are healthy

```powershell
docker compose ps
```

Confirm `mongodb` and `redis` show `(healthy)` before continuing. If a container shows `Exited` or `Restarting`, check its logs immediately:

```powershell
docker logs <container-name>
docker inspect <container-name>
```

Do not proceed until all three containers (`mongodb`, `redis`, `qdrant`) are up.

### 5. Start Ollama and pull required models

In a separate terminal (leave it running):

```powershell
ollama serve
```

In another terminal:

```powershell
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 6. Install and start the backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> If you see `running scripts is disabled on this system`, see [PowerShell Execution Policy](#powershell-execution-policy) below.

```powershell
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Leave this running. Backend is now at `http://localhost:8000`.

> **First-time note:** Docling downloads its layout and table-structure models (`docling-layout-heron`, `docling-models`) on the **first PDF upload**, not at install time — this can take a minute or two the very first time and is not an error.

### 7. Install and start the frontend

In a new terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend is now at `http://localhost:3000`.

### 8. Verify everything is up

Run through [Service Health Verification](#service-health-verification) before using the app.

### 9. Register a user, log in, upload a PDF, and chat

See the [Smoke Test](#smoke-test) section for the exact sequence.

---

## PowerShell Execution Policy

Windows blocks running the venv activation script by default. If `.\venv\Scripts\Activate.ps1` fails with `running scripts is disabled on this system`, fix it with **one** of:

```powershell
# Persistent fix for your user account (recommended)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

```powershell
# One-time fix for just the current terminal session
Set-ExecutionPolicy Bypass -Scope Process
```

Then re-run `.\venv\Scripts\Activate.ps1`.

---

## Qdrant Authentication

Qdrant can run with or without an API key, controlled by the `QDRANT__SERVICE__API_KEY` environment variable on the container (set via `QDRANT_API_KEY` in the root `.env`).

- **If `QDRANT_API_KEY` is set in the root `.env`** (authenticated mode): the container requires every request to carry that key. The backend must have the **same key** set as `QDRANT_API_KEY` in `backend/.env`. If the backend's key is missing or wrong, every retrieval call fails with `401 Unauthorized` — this failure looks like a retrieval bug but is actually a config mismatch.
- **If `QDRANT_API_KEY` is left blank** (unauthenticated mode, default for local dev): omit `QDRANT_API_KEY` from `backend/.env` too. No key is sent or required.

Rule of thumb: **the key must be present in both files or absent from both files.** A key in one but not the other is the failure state.

---

## Service Health Verification

Run these checks in order after `docker compose up -d` and before starting the backend/frontend.

### MongoDB

```powershell
docker ps
```

Expected: a `mongodb` container listed with status `Up ... (healthy)`.

```powershell
docker compose logs mongodb --tail 20
```

Expected: no `Authentication failed` or `exception in initAndListen` lines.

### Redis

```powershell
docker compose logs redis --tail 20
```

Expected: `Ready to accept connections`. To test connectivity directly:

```powershell
docker exec -it <redis-container-name> redis-cli -a <REDIS_PASSWORD> ping
```

Expected output: `PONG`.

### Qdrant

```powershell
curl http://localhost:6333/collections
```

Expected: a JSON response listing collections (`{"result":{"collections":[...]}...}`), **not** a `401` or connection error. If you get `401`, see [Qdrant Authentication](#qdrant-authentication).

### Backend

Once `uvicorn` is running:

```powershell
curl http://localhost:8000/health
```

Expected output:

```json
{"status": "ok"}
```

If this fails, check the backend terminal output — most failures at this stage are Mongo/Redis/Qdrant connection errors surfaced at startup.

### Frontend

Open `http://localhost:3000` in a browser. Expected: the app's login screen loads with no console errors referencing `NEXT_PUBLIC_API_URL` or a failed fetch to the backend.

---

## Running the Project

Once everything is installed, day-to-day startup only requires:

```powershell
# Terminal 1 — infrastructure
docker compose up -d

# Terminal 2 — Ollama
ollama serve

# Terminal 3 — backend
cd backend
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 4 — frontend
cd frontend
npm run dev
```

Useful URLs while running:

- Frontend: `http://localhost:3000`
- Backend API docs (Swagger): `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Qdrant dashboard: `http://localhost:6333/dashboard`

Run backend tests:

```powershell
cd backend
pytest tests/unit/ -v
```

---

## Smoke Test

A complete first-time walkthrough to confirm the whole stack works end to end:

1. Open `http://localhost:3000`.
2. **Register** a new user account.
3. **Log in** with those credentials.
4. **Upload a PDF** (any text-based PDF — the first upload will take longer while Docling downloads its models).
5. Wait for the document to show as processed.
6. **Ask a question** about the document's content in the chat.
7. Confirm you **receive a grounded answer** (not an error or an empty response).
8. Confirm the answer shows a **citation** back to the uploaded document's real filename (not "Source 1" / "Source N").

If all eight steps succeed, the full pipeline — auth, ingestion, hybrid retrieval, multi-agent orchestration, and citation tracing — is working correctly.

---

## Troubleshooting

### MongoDB — `Authentication failed`

- **Cause:** `MONGODB_URL` credentials in `backend/.env` don't match `MONGO_INITDB_ROOT_USERNAME` / `MONGO_INITDB_ROOT_PASSWORD` in the root `.env` — commonly because the password was changed after the container's first init.
- **Fix:** Align both files, then recreate the volume: `docker compose down -v && docker compose up -d`. See the [critical warning above](#️-critical-mongodb-credentials-must-match--and-can-silently-go-stale) before using `-v`.

### Qdrant — `401 Unauthorized`

- **Cause:** `QDRANT_API_KEY` is set on the container (root `.env`) but missing or mismatched in `backend/.env` — or vice versa.
- **Fix:** Make the key match in both files, or remove it from both for unauthenticated local dev. Restart the backend after editing `backend/.env`.

### Redis — `Connection refused`

- **Cause:** Either the Redis container isn't running (`docker compose ps` shows it stopped), or `REDIS_URL` in `backend/.env` points at the wrong host/port, or its password doesn't match `REDIS_PASSWORD`.
- **Fix:** Confirm the container is `Up`, confirm the port isn't already in use by another local Redis instance, and confirm the password matches.

### Backend — Import errors on startup

- **Cause:** Dependencies installed into the wrong Python environment (venv not activated), or Python version below 3.12.
- **Fix:** Confirm `python --version` is 3.12+, re-activate the venv (`.\venv\Scripts\Activate.ps1`), and reinstall: `pip install -r requirements.txt`.

### Frontend — `npm install` issues

- **Cause:** Node version below 20, or a stale `node_modules`/lockfile from a previous partial install.
- **Fix:** Confirm `node --version` is 20+, delete `node_modules` and `package-lock.json`, then re-run `npm install`.

### Docker — Container exits immediately

- **Cause:** Usually a missing or malformed required variable in the root `.env` (e.g. `MONGO_INITDB_ROOT_PASSWORD` left blank), or a port already in use on the host.
- **Fix:** `docker logs <container-name>` will name the exact missing variable or bind error. Fix `.env` or the conflicting port, then `docker compose up -d` again.

### General: PowerShell execution policy

See [PowerShell Execution Policy](#powershell-execution-policy) above.

---

## Verification Commands Cheat Sheet

Run these in sequence to sanity-check a full setup:

```powershell
# Containers are up and healthy
docker ps

# No error-level logs from infra
docker compose logs

# Python environment is correct
python -c "import sys; print(sys.version)"

# Backend is reachable and healthy
curl http://localhost:8000/health

# Qdrant is reachable
curl http://localhost:6333/collections

# Redis responds
docker exec -it <redis-container-name> redis-cli -a <REDIS_PASSWORD> ping
```

Expected: healthy containers, `{"status": "ok"}` from `/health`, a JSON collections list from Qdrant (no `401`), and `PONG` from Redis.

---

## Production Deployment

Production uses a separate Dockerfile for the backend (containerized independently of the root `docker-compose.yml`, which only provisions infra) and an nginx reverse proxy in front of it.

- **Environment variables & secrets:** Never commit real `.env` files. In production, inject `JWT_SECRET_KEY`, Mongo/Redis/Qdrant credentials, and any API keys via your platform's secrets manager (e.g. GitHub Actions Secrets for CI/CD, or your host's secret store) rather than checked-in files. The CI/CD pipeline expects `PROD_HOST`, `PROD_USER`, `PROD_SSH_KEY`, and `API_URL` as GitHub Secrets.
- **Docker volumes:** Mongo, Redis, and Qdrant data should be on named, persistent volumes (not anonymous ones) so data survives container recreation. Back these up separately from the application containers, which are stateless and safe to redeploy freely.
- **HTTPS:** Terminate TLS at the nginx reverse proxy in front of the backend; don't serve the app over plain HTTP outside of local dev.
- **Reverse proxy:** nginx handles SSL termination, rate limiting, and security headers in front of the FastAPI backend — see the `docker/` directory for the production Dockerfile and nginx config.
- **Backups:** Snapshot the MongoDB and Qdrant volumes on a schedule; Redis data is cache/session state and does not need durable backups.
- **Monitoring:** The `/health` endpoint is suitable for uptime checks and load balancer health probes. Watch backend logs for repeated Critic Agent rejections or Grader "batch too weak" outcomes as a proxy for retrieval quality drift.

### CI/CD Pipeline

Every push to `main` runs through `.github/workflows/deploy.yml`:

1. **`test-backend`** — spins up throwaway MongoDB + Redis containers, runs the full pytest suite.
2. **`test-frontend`** — installs, lints, and production-builds the Next.js app.
3. **`build-images`** *(only if both above pass, only on `main`)* — builds backend and frontend Docker images, pushes to GitHub Container Registry (`ghcr.io`).
4. **`deploy`** *(only if build succeeds)* — SSHes into the production server, pulls the new images, restarts containers, runs a smoke test against `/health`.

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for real issues hit while building this pipeline and how they were diagnosed and fixed.

---

## Development Workflow

- Backend agents live in `backend/app/agents/` — each agent is a single file inheriting from `BaseAgent`, which wraps execution in error handling so one agent's failure doesn't crash the LangGraph orchestrator.
- After changing backend code, `uvicorn --reload` picks up changes automatically.
- Run the unit test suite before pushing: `pytest tests/unit/ -v` (131 tests covering agents, auth, retrieval, and parsers).
- Frontend changes hot-reload under `npm run dev`.
- Read `TROUBLESHOOTING.md` for a log of real bugs found during development (particularly around the deterministic backstops in the Rewriter, Planner, Grader, and Critic agents) — it documents *why* each check exists, not just what it does.

---

## FAQ

**Do I need an OpenAI/Anthropic API key to run this?**
No. The default configuration is fully local via Ollama (`qwen2.5:7b` + `nomic-embed-text`). No external LLM API key is required for core functionality.

**Why does the first PDF upload take so long?**
Docling downloads its layout and table-structure models on first use, not at install time. Subsequent uploads are fast.

**I changed my Mongo password and now nothing works — why?**
See [the critical MongoDB warning](#️-critical-mongodb-credentials-must-match--and-can-silently-go-stale) — MongoDB only applies `MONGO_INITDB_ROOT_PASSWORD` on first container initialization. You need `docker compose down -v` to force re-initialization.

**Do I need Qdrant authentication for local development?**
No — leave `QDRANT_API_KEY` blank in both `.env` files for local dev. Enable it only when exposing Qdrant beyond your own machine.

**Can I use a different LLM instead of `qwen2.5:7b`?**
Yes — set `OLLAMA_MODEL` in `backend/.env` to any model you've pulled with `ollama pull`. Smaller/larger models will change the deterministic backstops' effectiveness, since they were tuned against `qwen2.5:7b`'s specific failure patterns.

**Where do I report a bug or ask a question?**
Open an issue in the repository, or check `TROUBLESHOOTING.md` first — many first-time setup issues are already documented there with root causes.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Make your changes; add or update tests under `backend/tests/unit/` for any backend logic changes.
3. Run the full backend test suite locally: `pytest tests/unit/ -v`.
4. Ensure the frontend builds cleanly: `npm run build` inside `frontend/`.
5. Open a pull request describing what changed and why — for changes to agent logic, include the specific failure case that motivated the change (this project's convention, see `TROUBLESHOOTING.md`, is to document *why* a backstop exists, not just what it does).
6. CI (`test-backend` and `test-frontend`) must pass before merge.

---

## Project Status & Known Gaps

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

### Performance

- Cold query (no cache): ~28s end-to-end through the full agent pipeline
- Cached query (repeat question): ~2s
- Uncached query after parallelizing Planner + Retriever: ~12s

### Tech Stack

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


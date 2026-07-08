# Runbooks

Operational guides for common changes to the RAG 2.0 system. Each runbook assumes you're working from the project root with the dev stack (`docker-compose.yml`) running.

docker exec -it rag-redis redis-cli FLUSHDB
---

## Runbook 1: Adding a new LLM provider

The system currently calls Ollama exclusively through `backend/app/services/llm/provider.py`. To add a second provider (e.g. OpenAI, Groq, Gemini) as an option or fallback:

### 1. Add the dependency
If not already in `backend/requirements.txt`, add the provider's SDK (e.g. `openai`, already present; `groq` if adding Groq).

### 2. Add settings
In `backend/app/config/settings.py`, the relevant API key fields already exist as optional (`OPENAI_API_KEY`, `GROQ_API_KEY`). Add the real key to your `.env` file — never commit it.

### 3. Extend `LLMProvider`
Open `backend/app/services/llm/provider.py`. The class currently has a single `generate()` method that calls Ollama directly. Refactor it to branch on a `provider` setting:

```python
class LLMProvider:
    def __init__(self, model: str = None, provider: str = "ollama"):
        self.provider = provider
        self.model = model or settings.OLLAMA_MODEL
        # existing Ollama client setup stays as-is

    async def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        if self.provider == "ollama":
            return await self._generate_ollama(prompt, max_tokens)
        elif self.provider == "openai":
            return await self._generate_openai(prompt, max_tokens)
        # add further branches as needed

    async def _generate_openai(self, prompt: str, max_tokens: int) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
```

Keep the existing `_generate_ollama` logic exactly as it is — don't touch the working path while adding a new one.

### 4. Wire it through agents (optional)
Every agent inherits from `BaseAgent`, which holds an `LLMProvider` instance via `self.llm`. If you want a *specific* agent (e.g. the Critic) to use a different provider than the rest, instantiate it with the provider override when constructing the orchestrator in `backend/app/agents/orchestrator.py`.

### 5. Test
Add a unit test mirroring the existing pattern in `backend/tests/unit/test_agents.py` — mock the new provider's client the same way `mock_llm()` mocks Ollama, so no real API key or network call is needed in CI.

### 6. Cost awareness
Unlike Ollama (free, local), any hosted provider incurs real per-token cost. If adding one, also extend `backend/app/utils/profiler.py` or add a lightweight token-counting wrapper so usage is visible — don't let this go unmonitored once it's live.

---

## Runbook 2: Adding a new tool

Tools live in `backend/app/services/tools/` and are registered in `tool_registry.py`. The existing tools (`calculator.py`, `web_search.py`, `sql_executor.py`) all follow the same shape.

### 1. Create the tool class
New file, e.g. `backend/app/services/tools/weather.py`:

```python
class WeatherTool:
    """Fetch current weather for a location"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def get_weather(self, location: str) -> dict:
        # call whatever weather API, return a plain dict
        return {"location": location, "temp_c": 24, "condition": "clear"}

weather_tool = None

def init_weather(api_key: str):
    global weather_tool
    weather_tool = WeatherTool(api_key)
```

Follow the exact pattern `web_search.py` uses: a module-level `None` global, initialized later via an `init_*()` function called from `main.py`'s startup event — **not** instantiated at import time. This matters because of how `tool_registry.py` checks for it (see below).

### 2. Register it
In `backend/app/services/tools/tool_registry.py`, the registry already live-checks `web_search_module.web_search_tool` on every call rather than capturing it once at import time — specifically so a tool that initializes *after* the registry exists (during FastAPI's startup event) still shows up correctly. Follow the same pattern: import your new tool's module (not the bare variable), and add a similar live-check entry inside `get_tools()`.

### 3. Initialize it at startup
In `backend/app/main.py`, inside the `@app.on_event("startup")` handler, alongside the existing `init_web_search(settings.TAVILY_API_KEY)` call, add:
```python
init_weather(settings.WEATHER_API_KEY)
```

### 4. Expose it through ToolAgent (optional)
If you want the agent pipeline itself to be able to call this tool (not just the manual `/tools/execute` endpoint), edit `backend/app/agents/tool_agent.py`'s `_execute()` method — it currently checks `sources_needed` for `"web"` or `"tools"` and calls `tool_registry.execute_tool("web_search", ...)`. Add a similar branch, gated on whatever signal the Planner should use to decide this tool is relevant (you may need to extend `PLANNER_PROMPT` in `backend/app/agents/prompts.py` so the Planner agent knows this tool exists and can request it).

### 5. Test
Mirror `backend/tests/unit/test_agents.py`'s `TestToolAgent` class — assert the tool is skipped when not needed, called with correct args when needed, and that an exception from the tool sets `state.error` rather than crashing the whole request.

---

## Runbook 3: Scaling the database layer

This system currently runs single-instance MongoDB, Redis, and Qdrant via Docker Compose — fine for moderate load, but here's what changes as usage grows.

### MongoDB

**Add indexes before anything else.** Per the Phase 14 backlog, `(user_id, created_at)` compound indexes are missing on the `documents` and `chat_sessions` collections. Every query in this system filters by `user_id` for isolation — without an index on that field, MongoDB does a full collection scan per request, which gets slow fast as data grows. Add via `backend/app/db/mongodb/queries.py`'s collection setup, or directly:
```python
await db["documents"].create_index([("user_id", 1), ("created_at", -1)])
await db["chat_sessions"].create_index([("user_id", 1), ("created_at", -1)])
```

**Beyond indexes:** if a single MongoDB instance becomes a bottleneck, move to a replica set (read scaling) before considering sharding — sharding adds real operational complexity and isn't worth it until you're well past what a replica set + good indexes can handle.

### Redis

**Move the query cache off in-memory first.** `backend/app/services/cache/query_cache.py` is currently a Python dict living inside the backend process — meaning (a) it's lost on every restart, and (b) it doesn't work correctly if you ever run more than one backend replica (each replica has its own separate cache, so a cache hit on replica A is a cache miss on replica B). `docker-compose.prod.yml` already runs 2 backend replicas — this is an active correctness gap, not just a performance one, once you're running multiple replicas in production.

Migrate `query_cache.py` to use the existing `redis_client.py` wrapper instead of an in-memory dict — same `get`/`set` interface, just backed by Redis so all replicas share one cache.

**Beyond that:** Redis itself rarely needs scaling before everything else does — it's fast enough that a single instance handles very high throughput. If it ever does become the bottleneck, Redis Cluster is the next step, but this is unlikely to be needed before MongoDB or Qdrant become concerns first.

### Qdrant

**Per-user namespace isolation** is already in place via metadata filtering (see `backend/app/db/qdrant/client.py`'s `search()` method, which filters by `user_id`). This scales reasonably well within a single Qdrant instance for a good while.

**When you do need to scale:** Qdrant supports horizontal scaling via sharding and replication natively — see Qdrant's own clustering documentation. This is a "later" concern; don't reach for it until you have actual evidence (slow query latency under real load) that a single instance is the bottleneck.

### General principle for all three

Add monitoring before scaling blindly. `backend/app/utils/profiler.py` already times each pipeline step — extend that visibility to the database layer specifically (query latency per collection/operation) before deciding which datastore to scale first. Scaling the wrong layer wastes effort; the profiler data tells you which layer is actually slow.

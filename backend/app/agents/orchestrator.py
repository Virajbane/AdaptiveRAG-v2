from app.agents.state import AgentState
from app.agents.graph import build_agent_graph
from app.services.cache.query_cache import query_cache
import app.services.memory.manager as mm_module
from app.db.mongodb.client import get_db
from langsmith import traceable

# 2026-07-04 fix: the pipeline was caching failed answers exactly like
# successful ones. Root cause chain: an Ollama /api/generate crash inside
# AnswerAgent set state.answer to this fallback string with
# confidence_final=0.0 (correct) - but CriticAgent then re-scored that
# fallback string as if it were a real answer (now fixed separately in
# critic.py), and regardless of that, THIS file cached the result
# unconditionally. Once cached, every future ask of the same question
# for the same user returned the stale failure straight from Redis,
# bypassing the graph entirely - so even after fixing planner.py and
# critic.py, the bad answer kept coming back via [CACHE HIT].
#
# Fix: only cache when the pipeline actually produced a real, non-error,
# non-fallback, non-zero-confidence answer. Caching is an optimization -
# skipping it for a failure just means the next ask reruns the graph
# fresh, which is exactly what we want after a transient crash.
_FAILED_ANSWER = "Sorry, I couldn't generate an answer."


class AgentOrchestrator:
    """
    Thin wrapper around the compiled LangGraph agent graph.

    Responsibilities NOT part of the agent graph itself (deliberately
    kept here, not as graph nodes):
    - Query cache check/store - this is a request-level optimization,
      not an agent decision
    - Conversation memory save - a side effect after the graph
      completes, not part of the reasoning pipeline
    """

    def __init__(self):
        self.graph = None

    async def _ensure_graph(self):
        if self.graph is None:
            db = await get_db()
            self.graph = build_agent_graph(db=db)

    @traceable(name="rag-2.0-system-run")
    async def _run_graph(self, initial_state: AgentState):
        """Runs the compiled agent graph. Decorated once at class
        definition time (not rebuilt on every request like the old
        inline-closure version was)."""
        return await self.graph.ainvoke(initial_state)

    async def process(
        self,
        question: str,
        user_id: str,
        session_id: str = "default_session",
    ) -> dict:
        """
        Process a question through the agent graph.

        session_id identifies the conversation thread for short-term
        memory lookups (used by RewriterAgent for context resolution,
        and to save this turn back into the same thread below).
        Defaults to "default_session" for backward compatibility with
        callers that don't yet pass one explicitly.
        """
        await self._ensure_graph()

        # -----------------------------
        # Cache Check
        # -----------------------------
        cached_result = await query_cache.get(question, user_id)
        print(f"[DEBUG] Cache hit: {cached_result is not None}")

        if cached_result:
            print("\n[CACHE HIT]")
            return cached_result

        print(f"\n{'=' * 50}")
        print(f"QUESTION: {question}")
        print(f"{'=' * 50}\n")

        # -----------------------------
        # Run the agent graph
        # -----------------------------
        initial_state = AgentState(
            question=question,
            user_id=user_id,
            session_id=session_id,
        )

        raw = await self._run_graph(initial_state)
        final_state = AgentState(**raw) if isinstance(raw, dict) else raw

        # IMPORTANT: `error` can be set by a node early in the pipeline
        # (e.g. Planner JSON parse failure) that the router already
        # recovered from gracefully (fallback to ["documents"], pipeline
        # kept running). That's NOT a fatal failure — Critic may have
        # gone on to validate a perfectly good answer. Only treat this
        # as a hard failure if the pipeline genuinely produced nothing
        # usable; otherwise return the real answer and let the stale
        # error sit in logs/state for debugging, not in the user response.
        if final_state.error and not final_state.answer:
            return self._error_response(final_state)

        # -----------------------------
        # Save Conversation Memory
        # -----------------------------
        if mm_module.memory_manager:
            await mm_module.memory_manager.save_interaction(
                user_id=user_id,
                session_id=session_id,
                user_message=question,
                assistant_response=final_state.answer,
            )

        # -----------------------------
        # Final Response
        # -----------------------------
        result = {
            "answer": final_state.answer,
            "sources": final_state.sources,
            "sources_needed": final_state.sources_needed,
            # Defensive default: avoids a TypeError crash if a pipeline
            # path ever leaves confidence_final unset (None) while still
            # producing a usable answer.
            "confidence": round(final_state.confidence_final or 0.0, 3),
            "search_time_ms": final_state.search_time_ms,
            "is_valid": final_state.is_valid,
        }

        # -----------------------------
        # Cache Result — only if it's a REAL answer
        # -----------------------------
        # See module-level 2026-07-04 note above. A failed/fallback
        # answer must never be cached, or it gets served verbatim to
        # every future identical question for this user until the TTL
        # expires, even after the underlying bug is fixed.
        should_cache = (
            not final_state.error
            and final_state.answer
            and final_state.answer.strip() != _FAILED_ANSWER
            and (final_state.confidence_final or 0.0) > 0.0
        )

        if should_cache:
            await query_cache.set(question, user_id, result)
        else:
            print(
                f"[CACHE SKIP] Not caching failed/low-confidence result "
                f"(error={final_state.error!r}, "
                f"confidence={final_state.confidence_final})"
            )

        return result

    def _error_response(self, state: AgentState) -> dict:
        """Return standardized error response."""

        return {
            "answer": f"Error: {state.error}",
            "sources": [],
            "confidence": 0.0,
            "search_time_ms": 0,
            "is_valid": False,
        }
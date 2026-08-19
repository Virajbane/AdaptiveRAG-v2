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
    
    2026-08-09: Added knowledge_version parameter to cache key for
    document RAG safety. Same canonical question now resolves to same
    answer only when the underlying knowledge base hasn't changed.
    Caching now uses rewritten_question (canonical form) after graph
    execution to avoid redundant reruns on rephrased queries.
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
        knowledge_version: str = "",
        include_diagnostics: bool = False,
    ) -> dict:
        """
        Process a question through the agent graph.

        Args:
            question: The user's question.
            user_id: Identifies the user for cache scope and memory.
            session_id: Conversation thread ID for short-term memory lookups
                (used by RewriterAgent for context resolution, and to save
                this turn back into the same thread). Defaults to "default_session"
                for backward compatibility.
            knowledge_version: Optional document/index/knowledge base version
                identifier. Should be set by callers when the underlying
                knowledge base version is known (e.g., document index version,
                database schema revision). Prevents stale cached answers after
                knowledge updates. If not provided, defaults to empty string.
                IMPORTANT: Do not invent a version number; only pass if
                available from the application context.
            include_diagnostics: Internal evaluation flag. When true, bypasses
                the query cache and returns retrieval/answer-context trace
                fields. Never enable this for the public API response.

        Returns:
            dict with keys:
            - answer: generated response text
            - sources: list of evidence sources
            - sources_needed: original planner source decision
            - confidence: final confidence score (0-1)
            - search_time_ms: execution time
            - is_valid: critic validation result
        """
        await self._ensure_graph()

        # -----------------------------
        # Cache Check (initial lookup)
        # -----------------------------
        # Use original question for cache lookup; after graph execution,
        # we'll cache using the canonical (rewritten) form to avoid
        # redundant reruns on differently-phrased queries.
        if include_diagnostics:
            # A cached public response contains no internal evidence trace.
            # Evaluation must execute the graph to observe this invocation.
            cached_result = None
            print("[DEBUG] Cache bypassed for diagnostic run")
        else:
            cached_result = await query_cache.get(
                question, user_id, knowledge_version=knowledge_version
            )
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

        if include_diagnostics:
            result["rewritten_question"] = final_state.rewritten_question
            result["retrieved_docs"] = final_state.retrieved_docs
            result["answer_context"] = final_state.answer_context
            result["answer_context_docs"] = final_state.answer_context_docs
            result["answer_context_dropped_docs"] = final_state.answer_context_dropped_docs

        # -----------------------------
        # Cache Result — only if it's a REAL answer
        # -----------------------------
        # See module-level 2026-07-04 note above. A failed/fallback
        # answer must never be cached, or it gets served verbatim to
        # every future identical question for this user until the TTL
        # expires, even after the underlying bug is fixed.
        #
        # 2026-08-09: Cache using canonical question (rewritten_question)
        # so that differently-phrased versions of the same semantic query
        # hit the same cache entry after the first run.
        should_cache = (
            not final_state.error
            and final_state.answer
            and final_state.answer.strip() != _FAILED_ANSWER
            and (final_state.confidence_final or 0.0) > 0.0
        )

        if should_cache and not include_diagnostics:
            # Use rewritten_question (canonical form) as cache key; fall back
            # to original question if no rewrite occurred.
            canonical_question = final_state.rewritten_question or question
            await query_cache.set(
                canonical_question,
                user_id,
                result,
                knowledge_version=knowledge_version,
            )
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

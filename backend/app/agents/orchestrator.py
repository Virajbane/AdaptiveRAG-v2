from app.agents.state import AgentState
from app.agents.graph import build_agent_graph
from app.services.cache.query_cache import query_cache
import app.services.memory.manager as mm_module


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
        self.graph = build_agent_graph()

    async def process(
        self,
        question: str,
        user_id: str
    ) -> dict:
        """
        Process a question through the agent graph.
        """

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
            user_id=user_id
        )

        final_state: AgentState = await self.graph.ainvoke(initial_state)

        if final_state.error:
            return self._error_response(final_state)

        # -----------------------------
        # Save Conversation Memory
        # -----------------------------
        if mm_module.memory_manager:
            await mm_module.memory_manager.save_interaction(
                user_id=user_id,
                session_id="default_session",
                user_message=question,
                assistant_response=final_state.answer,
            )

        # -----------------------------
        # Final Response
        # -----------------------------
        result = {
            "answer": final_state.answer,
            "sources": final_state.sources,
            "confidence": round(final_state.confidence_final, 3),
            "search_time_ms": final_state.search_time_ms,
            "is_valid": final_state.is_valid,
        }

        # -----------------------------
        # Cache Result
        # -----------------------------
        await query_cache.set(question, user_id, result)

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
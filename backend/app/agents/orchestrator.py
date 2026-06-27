import asyncio

from app.agents.state import AgentState
from app.agents.planner import PlannerAgent
from app.agents.retriever import RetrieverAgent
from app.agents.tool_agent import ToolAgent
from app.agents.critic import CriticAgent
from app.agents.answer import AnswerAgent

from app.services.llm.provider import LLMProvider
from app.services.cache.query_cache import query_cache
import app.services.memory.manager as mm_module


class AgentOrchestrator:
    """
    Main orchestrator: Coordinates all agents

    Flow:
    1. Planner
    2. Retriever (Parallel with Planner)
    3. Answer
    """

    def __init__(self):
        llm = LLMProvider()

        self.planner = PlannerAgent(llm)
        self.retriever = RetrieverAgent(llm)
        self.tool_agent = ToolAgent(llm)
        self.critic = CriticAgent(llm)
        self.answer = AnswerAgent(llm)

    async def process(
        self,
        question: str,
        user_id: str
    ) -> dict:
        """
        Process a question through all agents.

        Optimization:
        - Query cache
        - Planner + Retriever run in parallel
        """

        # -----------------------------
        # Cache Check
        # -----------------------------
        cached_result = await query_cache.get(question, user_id)
        print(f"[DEBUG] Cache hit: {cached_result is not None}")

        if cached_result:
            print("\n[CACHE HIT]")
            return cached_result

        from app.utils.profiler import profiler

        print(f"\n{'=' * 50}")
        print(f"QUESTION: {question}")
        print(f"{'=' * 50}\n")

        # -----------------------------
        # Initial State
        # -----------------------------
        state = AgentState(
            question=question,
            user_id=user_id
        )

        print("[1/3] PLANNER + RETRIEVER (PARALLEL)")

        profiler.start("planner")
        profiler.start("retriever")

        planner_task = self.planner.run(
            state.copy()
            if hasattr(state, "copy")
            else AgentState(question=question, user_id=user_id)
        )

        retriever_task = self.retriever.run(
            state.copy()
            if hasattr(state, "copy")
            else AgentState(question=question, user_id=user_id)
        )

        state_from_planner, state_from_retriever = await asyncio.gather(
            planner_task,
            retriever_task,
            return_exceptions=False
        )

        planner_time = profiler.end("planner")
        retriever_time = profiler.end("retriever")

        # -----------------------------
        # Merge Results
        # -----------------------------
        state.plan = state_from_planner.plan
        state.sources_needed = state_from_planner.sources_needed
        state.confidence = state_from_planner.confidence

        state.retrieved_docs = state_from_retriever.retrieved_docs
        state.search_time_ms = state_from_retriever.search_time_ms

        # -----------------------------
        # Error Handling
        # -----------------------------
        if state_from_planner.error:
            return self._error_response(state_from_planner)

        if state_from_retriever.error:
            return self._error_response(state_from_retriever)

        print(f"Planner   : {planner_time:.0f} ms")
        print(f"Retriever : {retriever_time:.0f} ms")

        # -----------------------------
        # Answer Agent
        # -----------------------------
        print("\n[2/3] ANSWER")

        profiler.start("answer")

        state = await self.answer.run(state)

        profiler.end("answer")

        if state.error:
            return self._error_response(state)

        print("\n[3/3] COMPLETE\n")

        # -----------------------------
        # Save Conversation Memory
        # -----------------------------
        if mm_module.memory_manager:
            await mm_module.memory_manager.save_interaction(
                user_id=user_id,
                session_id="default_session",
                user_message=question,
                assistant_response=state.answer,
            )

        profiler.print_summary()

        # -----------------------------
        # Final Response
        # -----------------------------
        result = {
            "answer": state.answer,
            "sources": state.sources,
            "confidence": round(state.confidence_final, 3),
            "search_time_ms": state.search_time_ms,
            "is_valid": state.is_valid,
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
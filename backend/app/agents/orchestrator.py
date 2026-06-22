import asyncio
from app.agents.state import AgentState
from app.agents.planner import PlannerAgent
from app.agents.retriever import RetrieverAgent
from app.agents.tool_agent import ToolAgent
from app.agents.critic import CriticAgent
from app.agents.answer import AnswerAgent
from app.services.llm.provider import LLMProvider
import app.services.memory.manager as mm_module

class AgentOrchestrator:
    """
    Main orchestrator: Coordinates all agents
    
    Flow:
    1. PLANNER: Decide strategy
    2. RETRIEVER + TOOL: Parallel search
    3. CRITIC: Validate
    4. ANSWER: Generate response
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
        Process a question through all agents
        
        LATENCY OPTIMIZATION (Fix #3):
        Skip Tool Agent when:
        - It's not explicitly needed
        - Web search/calculator aren't really configured
        Saves ~5 seconds per request
        """
        
        print(f"\n{'='*50}")
        print(f"QUESTION: {question}")
        print(f"{'='*50}\n")
        
        # Create initial state
        state = AgentState(
            question=question,
            user_id=user_id
        )
        
        # Step 1: Planner decides strategy
        print("[1/4] PLANNER")
        state = await self.planner.run(state)
        if state.error:
            return self._error_response(state)
        
        # Step 2: Retriever search (always run)
        print("\n[2/4] RETRIEVER")
        state = await self.retriever.run(state)
        if state.error:
            return self._error_response(state)
        
        # Step 3: Skip Tool Agent for now
        # (web_search returns mock data, so calling LLM to decide is wasted latency)
        # TODO: Enable when Tavily API is configured
        print("\n[3/4] ANSWER (Tool Agent skipped - no real web search configured)")
        
        # Step 4: Generate answer
        state = await self.answer.run(state)
        if state.error:
            return self._error_response(state)
        
        print("\n[4/4] COMPLETE\n")

        # Save interaction to memory
        if mm_module.memory_manager:
            await mm_module.memory_manager.save_interaction(
                user_id=user_id,
                session_id="default_session",  # Or get from request
                user_message=question,
                assistant_response=state.answer
            )
        
        return {
            "answer": state.answer,
            "sources": state.sources,
            "confidence": round(state.confidence_final, 3),
            "search_time_ms": state.search_time_ms,
            "is_valid": state.is_valid
        }
    
    def _error_response(self, state: AgentState) -> dict:
        """Return error response"""
        return {
            "answer": f"Error: {state.error}",
            "sources": [],
            "confidence": 0.0,
            "search_time_ms": 0,
            "is_valid": False
        }
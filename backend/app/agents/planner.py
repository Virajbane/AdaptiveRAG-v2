import json
from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.agents.prompts import PLANNER_PROMPT

class PlannerAgent(BaseAgent):
    """
    Planner Agent: Decides what sources to use
    
    - Analyzes user intent
    - Chooses retrieval strategy
    - Sets confidence level
    """
    
    async def _execute(self, state: AgentState) -> AgentState:
        """Plan the search strategy"""
        
        # Call LLM with prompt
        prompt = PLANNER_PROMPT.format(question=state.question)
        response = await self.call_llm(prompt)
        
        try:
            # Parse JSON response
            plan = self.parse_json_response(response)
            
            state.plan = plan.get("strategy", "")
            state.sources_needed = plan.get("sources", ["documents"])
            state.confidence = plan.get("confidence", 0.5)
            
            print(f"[PLANNER] Plan: {state.plan}")
            print(f"[PLANNER] Sources: {state.sources_needed}")
            print(f"[PLANNER] Confidence: {state.confidence}")
            
        except json.JSONDecodeError:
            state.error = "Failed to parse planner response"
            state.sources_needed = ["documents"]
        
        return state
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

    Note: model selection (fast vs. main LLM) is handled by
    AgentOrchestrator, which injects the right LLMProvider instance.
    This class does not need its own __init__.
    """

    async def _execute(self, state: AgentState) -> AgentState:
        """Plan the search strategy"""

        prompt = PLANNER_PROMPT.format(question=state.question)

        try:
            response = await self.call_llm(prompt)
        except Exception as e:
            # Surface the REAL failure (e.g. model not found, connection
            # refused, timeout) instead of letting it fall through to a
            # JSON parse error that hides the actual cause.
            state.error = f"Planner LLM call failed: {str(e)}"
            state.sources_needed = ["documents"]
            print(f"[PLANNER] LLM call failed: {e}")
            return state

        try:
            plan = self.parse_json_response(response)

            state.plan = plan.get("strategy", "")
            state.sources_needed = plan.get("sources", ["documents"])
            state.confidence = plan.get("confidence", 0.5)

            print(f"[PLANNER] Plan: {state.plan}")
            print(f"[PLANNER] Sources: {state.sources_needed}")
            print(f"[PLANNER] Confidence: {state.confidence}")

        except json.JSONDecodeError:
            state.error = "Failed to parse planner response as JSON"
            state.sources_needed = ["documents"]
            print(f"[PLANNER] Could not parse response as JSON: {response[:200]!r}")

        return state
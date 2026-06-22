from abc import ABC, abstractmethod
from app.agents.state import AgentState
from app.services.llm.provider import LLMProvider
import json
import re

class BaseAgent(ABC):
    """
    Base class for all agents
    
    Each agent:
    - Takes AgentState as input
    - Modifies state as needed
    - Returns updated state
    """
    
    def __init__(self, llm: LLMProvider = None):
        self.llm = llm or LLMProvider()
        self.name = self.__class__.__name__
    
    async def run(self, state: AgentState) -> AgentState:
        """
        Execute agent logic
        
        Args:
            state: Current agent state
        
        Returns:
            Updated state
        """
        try:
            # Call agent-specific logic
            result = await self._execute(state)
            return result
        except Exception as e:
            state.error = f"{self.name} error: {str(e)}"
            return state
    
    @abstractmethod
    async def _execute(self, state: AgentState) -> AgentState:
        """Implement agent-specific logic"""
        pass
    
    async def call_llm(self, prompt: str, **kwargs) -> str:
        """Helper to call LLM"""
        response = await self.llm.generate(prompt, **kwargs)
        return response

    def parse_json_response(self, response: str) -> dict:
        """
        Robustly extract a JSON object from an LLM response.
        
        Local models sometimes wrap JSON in markdown fences, add
        explanation text before/after, or include multiple JSON-like
        blocks. This grabs just the first complete {...} block.
        """
        # Try straightforward parse first (fastest path, works often)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Strip markdown code fences if present (```json ... ```)
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Last resort: find the first balanced {...} block and parse just that
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Give up - raise so the caller's except block handles it
        raise json.JSONDecodeError("Could not extract valid JSON", response, 0)
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

    def _extract_balanced_json(self, text: str) -> str | None:
        """
        Scan for the first complete top-level {...} object using brace
        depth tracking, instead of a greedy regex. This correctly stops
        at the end of the FIRST JSON object even if the model emits a
        second JSON-looking block right after it (e.g. a few-shot example
        followed by the real answer) — a greedy `\\{.*\\}` regex would
        incorrectly span from the first '{' to the LAST '}' across both
        blocks, producing invalid JSON.

        Tracks whether we're inside a string literal so braces inside
        quoted values (e.g. "issues": ["unexpected '{' in text"]) don't
        throw off the depth count. Handles escaped quotes (\\").

        Returns the matched substring, or None if no balanced object
        was found (e.g. truncated/malformed output).
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            char = text[i]

            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

        return None  # never closed — malformed/truncated

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
        
        # Balanced-brace scan instead of greedy regex. The old
        # `re.search(r'\{.*\}', response, re.DOTALL)` spans from the
        # first '{' to the LAST '}' in the whole string, so when the
        # model emits two JSON-looking blocks back to back (e.g. a
        # few-shot example block followed by the real answer block),
        # it merges them into one invalid blob. Scanning brace depth
        # stops at the end of the first complete object instead.
        candidate = self._extract_balanced_json(response)
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        
        # Give up - raise so the caller's except block handles it
        raise json.JSONDecodeError("Could not extract valid JSON", response, 0)
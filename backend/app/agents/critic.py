import json
from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.agents.prompts import CRITIC_PROMPT

class CriticAgent(BaseAgent):
    """
    Critic Agent: Validates answer quality
    
    - Checks for hallucinations
    - Verifies evidence grounding
    - Detects missing info
    - Returns confidence score
    """
    
    async def _execute(self, state: AgentState) -> AgentState:
        """Validate the answer before sending to user"""
        
        # Format context from retrieved docs
        context = "\n".join([
            f"[{i}] {doc['text'][:200]}..."
            for i, doc in enumerate(state.retrieved_docs, 1)
        ])
        
        # Generate initial answer first (from answer agent)
        # For now, this is a placeholder
        if not state.answer:
            state.answer = "Searching for relevant information..."
        
        try:
            prompt = CRITIC_PROMPT.format(
                question=state.question,
                context=context,
                answer=state.answer
            )
            
            response = await self.call_llm(prompt)
            criticism = self.parse_json_response(response)
            
            state.is_valid = criticism.get("valid", False)
            state.validation_issues = criticism.get("issues", [])
            
            print(f"[CRITIC] Valid: {state.is_valid}")
            if state.validation_issues:
                print(f"[CRITIC] Issues: {state.validation_issues}")
        
        except Exception as e:
            state.error = f"Critic error: {str(e)}"
            state.is_valid = False
        
        return state
from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.agents.prompts import ANSWER_PROMPT

class AnswerAgent(BaseAgent):
    """
    Answer Agent: Generates final response

    CRITICAL FIXES:
    1. Sources ONLY from real retrieved_docs (never LLM-invented)
    2. Confidence calculation - fixed operator precedence
    3. LLM asked for PLAIN TEXT only, no JSON - local models were
       inconsistently malforming the JSON contract (missing commas,
       double-nesting, "Sources:" instead of "sources":, etc.) across
       different calls. Since sources never came from the LLM's JSON
       anyway (see FIX #2), there was no reason to ask for it in JSON
       in the first place - this removes that whole failure surface.
    """

    async def _execute(self, state: AgentState) -> AgentState:
        """Generate final answer with citations"""

        # FIX #1: Check if we have ANY real documents
        if not state.retrieved_docs:
            state.answer = (
                "I don't have any documents to search. "
                "Please upload documents first, then ask your question."
            )
            state.sources = []
            state.confidence_final = state.confidence * 0.3  # Low confidence
            print("[ANSWER] No documents - returning explicit message")
            return state

        # Format context from REAL retrieved documents only
        context = "\n".join([
            f"[{i}] {doc['text']}"
            for i, doc in enumerate(state.retrieved_docs, 1)
        ])

        try:
            prompt = ANSWER_PROMPT.format(
                question=state.question,
                context=context
            )

            # FIX #3: plain text in, plain text out - no JSON parsing needed
            response = await self.call_llm(prompt)
            state.answer = response.strip()

            # FIX #2: Sources ONLY from REAL retrieved_docs
            # NEVER use LLM's invented JSON sources - they hallucinate
            state.sources = [
                {
                    "doc_id": doc['doc_id'],
                    "chunk_index": doc['chunk_index'],
                    "text": doc['text'][:200],
                    "score": doc['combined_score']
                }
                for doc in state.retrieved_docs
            ]

            # FIX #3: Confidence calculation - FIXED operator precedence
            # Guarded if/else instead of ternary to avoid precedence bug
            if state.retrieved_docs:
                avg_doc_score = sum(
                    doc['combined_score'] for doc in state.retrieved_docs
                ) / len(state.retrieved_docs)
                state.confidence_final = (
                    state.confidence * 0.5 +  # Planner confidence
                    avg_doc_score * 0.5        # Document match quality
                )
            else:
                state.confidence_final = state.confidence * 0.3

            state.confidence_final = min(state.confidence_final, 1.0)

            print(f"[ANSWER] Generated response with {len(state.sources)} real sources")
            print(f"[ANSWER] Confidence: {state.confidence_final:.2f}")

        except Exception as e:
            state.error = f"Answer agent error: {str(e)}"
            state.answer = "Sorry, I couldn't generate an answer."
            state.sources = []
            state.confidence_final = 0.0

        return state
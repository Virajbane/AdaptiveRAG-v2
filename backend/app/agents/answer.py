from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.agents.prompts import ANSWER_PROMPT


class AnswerAgent(BaseAgent):
    """
    Answer Agent: Generates final response

    CRITICAL FIXES:
    1. Sources ONLY from real retrieved_docs (never LLM-invented)
    2. Confidence calculation fixed
    3. Plain text output from LLM
    4. Performance optimization:
       - Only top 3 docs sent to LLM
       - Only first 300 chars of each doc
    """

    async def _execute(self, state: AgentState) -> AgentState:
        """Generate final answer with citations"""

        # No documents found
        if not state.retrieved_docs:
            state.answer = (
                "I don't have any documents to search. "
                "Please upload documents first, then ask your question."
            )
            state.sources = []
            state.confidence_final = state.confidence * 0.3

            print("[ANSWER] No documents - returning explicit message")
            return state

        # --------------------------------------------------
        # PERFORMANCE OPTIMIZATION
        # Use only Top-3 retrieved documents
        # Truncate each to 300 characters
        # --------------------------------------------------
        top_docs = state.retrieved_docs[:3]

        context = "\n".join([
            f"[{i}] {doc['text'][:300]}..."
            for i, doc in enumerate(top_docs, 1)
        ])

        print(f"[ANSWER] Using {len(top_docs)} top documents")

        try:
            prompt = ANSWER_PROMPT.format(
                question=state.question,
                context=context
            )

            # Generate answer
            response = await self.call_llm(prompt)
            state.answer = response.strip()

            # Sources from REAL retrieved docs
            state.sources = [
                {
                    "doc_id": doc["doc_id"],
                    "chunk_index": doc["chunk_index"],
                    "text": doc["text"][:200],
                    "score": doc["combined_score"],
                }
                for doc in state.retrieved_docs
            ]

            # Confidence calculation
            avg_doc_score = (
                sum(doc["combined_score"] for doc in state.retrieved_docs)
                / len(state.retrieved_docs)
            )

            state.confidence_final = min(
                state.confidence * 0.5 +
                avg_doc_score * 0.5,
                1.0
            )

            print(f"[ANSWER] Generated response with {len(state.sources)} sources")
            print(f"[ANSWER] Confidence: {state.confidence_final:.2f}")

        except Exception as e:
            state.error = f"Answer agent error: {str(e)}"
            state.answer = "Sorry, I couldn't generate an answer."
            state.sources = []
            state.confidence_final = 0.0

        return state
from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.agents.prompts import ANSWER_PROMPT


class AnswerAgent(BaseAgent):
    async def _execute(self, state: AgentState) -> AgentState:

        if not state.retrieved_docs:
            state.answer = (
                "I don't have any documents to search. "
                "Please upload documents first, then ask your question."
            )
            state.sources = []
            state.confidence_final = state.confidence * 0.3
            print("[ANSWER] No documents - returning explicit message")
            return state

        # Use top 6 docs, full text — no truncation
        top_docs = state.retrieved_docs[:6]

        context = "\n\n".join([
            f"[Source {i}]\n{doc['text']}"
            for i, doc in enumerate(top_docs, 1)
        ])

        print(f"[ANSWER] Using {len(top_docs)} top documents")

        try:
            prompt = ANSWER_PROMPT.format(
                question=state.question,
                context=context
            )

            response = await self.call_llm(prompt)
            state.answer = response.strip()

            # NOTE: sources built from top_docs (the docs actually sent to
            # the LLM), not all retrieved_docs - previously these could
            # diverge if Retriever's top_k ever changed independently of
            # AnswerAgent's top-6 slice.
            #
            # filename comes from RetrieverAgent's batched Mongo lookup
            # (attached as doc['filename']). Falls back to a generic
            # "Source N" label only if the lookup didn't run (e.g. db
            # wasn't wired through) or the doc_id had no matching record.
            state.sources = [
                {
                    "doc_id": doc["doc_id"],
                    "chunk_index": doc["chunk_index"],
                    "filename": doc.get("filename") or f"Source {i}",
                    "text": doc["text"][:200],
                    "score": doc.get("rerank_score", doc.get("combined_score", 0.0)),
                }
                for i, doc in enumerate(top_docs, 1)
            ]

            # Confidence fix: combined_score is the RRF fusion score, which
            # is rank-based and deliberately tiny/tightly-clustered
            # (RRF_K=60 means scores live in roughly the 0.01-0.05 range
            # no matter how good or bad retrieval actually is). Averaging
            # that directly into a 0-1 confidence score mathematically
            # guarantees confidence_final lands near ~0.42-0.45 for EVERY
            # query regardless of answer quality - which is exactly the
            # symptom observed (confidence stuck at 0.44-0.47 across
            # multiple different, correct answers).
            #
            # rerank_score (the BGE cross-encoder score) carries real
            # signal about retrieval relevance and should be used instead.
            # Falls back to combined_score only if reranking didn't run.
            scored_docs = [
                doc.get("rerank_score", doc.get("combined_score", 0.0))
                for doc in state.retrieved_docs
            ]
            avg_doc_score = sum(scored_docs) / len(scored_docs) if scored_docs else 0.0

            # rerank_score from a cross-encoder isn't naturally bounded to
            # [0, 1] - clamp defensively so confidence_final stays sane
            # even if the model's raw score is outside that range.
            avg_doc_score = max(0.0, min(avg_doc_score, 1.0))

            state.confidence_final = min(
                state.confidence * 0.5 + avg_doc_score * 0.5, 1.0
            )

            print(f"[ANSWER] Generated response with {len(state.sources)} sources")
            print(f"[ANSWER] Confidence: {state.confidence_final:.2f}")

        except Exception as e:
            state.error = f"Answer agent error: {str(e)}"
            state.answer = "Sorry, I couldn't generate an answer."
            state.sources = []
            state.confidence_final = 0.0

        return state
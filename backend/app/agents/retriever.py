from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.services.retrieval.hybrid_search import HybridSearchEngine
import time

class RetrieverAgent(BaseAgent):
    """
    Retriever Agent: Searches documents

    - Performs hybrid search (vector + keyword)
    - Retrieves top documents
    - Passes context to other agents

    NOTE: Runs in PARALLEL with PlannerAgent, so it always searches.
    The orchestrator decides whether to use the results based on
    planner's sources_needed AFTER both complete.
    """

    async def _execute(self, state: AgentState) -> AgentState:
        """Search for relevant documents — always runs, results used conditionally."""

        try:
            start_time = time.time()

            search_engine = HybridSearchEngine()
            results = await search_engine.search(
                query=state.question,
                user_id=state.user_id,
                top_k=5
            )

            state.retrieved_docs = results
            state.search_time_ms = (time.time() - start_time) * 1000

            print(f"[RETRIEVER] Found {len(results)} documents")
            for i, doc in enumerate(results, 1):
                # rerank_score is the BGE cross-encoder score and is what
                # actually determines final ranking (when reranking ran).
                # combined_score is the RRF fusion score from BM25+vector -
                # it's rank-based and tightly clustered by design (RRF_K=60),
                # so it will always look "flat" at 2 decimal places even
                # when fusion is working correctly. Don't use it to judge
                # whether retrieval quality is good or bad.
                if 'rerank_score' in doc:
                    score_label = "rerank"
                    score_value = doc['rerank_score']
                else:
                    # Reranker was unavailable/skipped - fall back to RRF score,
                    # but flag it so it's obvious which scoring path was used.
                    score_label = "rrf (no rerank)"
                    score_value = doc.get('combined_score', 0.0)

                print(f"  {i}. [{score_label}] Score: {score_value:.4f} | {doc['text'][:60]}...")

        except Exception as e:
            import traceback
            traceback.print_exc()
            state.error = f"Retriever error: {str(e)}"

        return state
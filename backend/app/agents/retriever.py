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
    """
    
    async def _execute(self, state: AgentState) -> AgentState:
        """Search for relevant documents"""
        
        # Check if documents search is needed
        if "documents" not in state.sources_needed:
            return state
        
        try:
            start_time = time.time()
            
            # Perform hybrid search
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
                print(f"  {i}. Score: {doc['combined_score']:.2f}")
            
        except Exception as e:
            state.error = f"Retriever error: {str(e)}"
        
        return state
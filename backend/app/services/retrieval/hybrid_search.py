import asyncio
from typing import List
from app.services.retrieval.vector_search import VectorSearchEngine
from app.services.retrieval.keyword_search import keyword_manager
from app.services.retrieval.reranker import bge_reranker

class HybridSearchEngine:
    """Combine vector and keyword search, with optional BGE reranking"""
    
    def __init__(self):
        self.vector_engine = VectorSearchEngine()
        self.keyword_engine = keyword_manager
        # bge_reranker is a module-level singleton (see reranker.py) -
        # the model is ~500MB and loaded once at import time, not
        # re-created per HybridSearchEngine instance.
        self.reranker = bge_reranker
    
    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        rerank_pool_size: int = 10
    ) -> List[dict]:
        """
        Hybrid search combining vector + keyword, then reranked.
        
        Args:
            query: Search query
            user_id: User ID
            top_k: Number of final results to return
            vector_weight: Weight for vector search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            rerank_pool_size: How many combined candidates to hand to the
                reranker before cutting down to top_k. Larger than top_k
                on purpose - the cheap vector+keyword combined_score is
                used to narrow the field first, then the more expensive
                but more accurate cross-encoder reranks within that
                narrowed pool. If the reranker is unavailable, this has
                no effect - behavior falls back to the original
                combined_score-sorted result exactly as before.
        
        Returns:
            Top-k results, reranked by BGE cross-encoder when available
        """
        
        # Run both searches in parallel
        vector_results, keyword_results = await asyncio.gather(
            self.vector_engine.search(query, user_id, top_k=10),
            self.keyword_engine.search(user_id, query, top_k=10)
        )
        
        # Combine results
        combined = self._combine_results(
            vector_results,
            keyword_results,
            vector_weight,
            keyword_weight
        )
        
        # Deduplicate (keep highest score for each chunk)
        deduped = self._deduplicate(combined)
        
        # Sort by combined score (vector + keyword blend) - this is the
        # cheap first-pass ranking used to select the candidate pool
        deduped.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Narrow to a candidate pool before the expensive rerank step.
        # If there are fewer candidates than the pool size, this is a
        # no-op slice.
        candidate_pool = deduped[:rerank_pool_size]
        
        # Rerank the candidate pool. CrossEncoder.predict() is a
        # blocking, CPU/GPU-bound call (not I/O), so it's run via
        # run_in_executor to avoid freezing the event loop for every
        # other concurrent request - same pattern already used for the
        # blocking Ollama call in services/document/embedder.py.
        if self.reranker.available and candidate_pool:
            loop = asyncio.get_event_loop()
            reranked = await loop.run_in_executor(
                None,
                self.reranker.rerank,
                query,
                candidate_pool,
                top_k
            )
            return reranked
        
        # Reranker unavailable (model failed to load) or empty pool -
        # fall back to the original combined_score ranking, unchanged
        # from pre-reranker behavior.
        return candidate_pool[:top_k]
    
    def _combine_results(
        self,
        vector_results: List[dict],
        keyword_results: List[dict],
        vector_weight: float,
        keyword_weight: float
    ) -> List[dict]:
        """
        Combine vector and keyword results with weighted scores
        """
    
        combined = {}
    
    # Add vector results
        for result in vector_results:
            key = f"{result['doc_id']}:{result['chunk_index']}"
            combined[key] = {
                **result,
                'vector_score': result['score'],
                'keyword_score': 0,
                'combined_score': result['score'] * vector_weight
            }
    
    # Add/update with keyword results
        for result in keyword_results:
            key = f"{result['doc_id']}:{result['chunk_index']}"
            # Clip negative BM25 scores to 0 - a weak/negative keyword score
            # should contribute nothing, not actively reduce relevance
            clipped_keyword_score = max(0, result['score'])
            
            if key in combined:
                # Update with keyword score
                combined[key]['keyword_score'] = clipped_keyword_score
                combined[key]['combined_score'] = (
                    (combined[key]['vector_score'] * vector_weight) +
                    (clipped_keyword_score * keyword_weight)
                )
            else:
                # New result from keyword search
                combined[key] = {
                    **result,
                    'vector_score': 0,
                    'keyword_score': clipped_keyword_score,
                    'combined_score': clipped_keyword_score * keyword_weight
                }
    
        return list(combined.values())
    
    def _deduplicate(self, results: List[dict]) -> List[dict]:
        """
        Remove duplicate chunks, keeping highest score
        """
        seen = {}
        for result in results:
            key = f"{result['doc_id']}:{result['chunk_index']}"
            if key not in seen or result['combined_score'] > seen[key]['combined_score']:
                seen[key] = result
        
        return list(seen.values())
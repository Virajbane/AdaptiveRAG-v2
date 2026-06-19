from typing import List
from app.services.retrieval.vector_search import VectorSearchEngine
from app.services.retrieval.keyword_search import keyword_manager

class HybridSearchEngine:
    """Combine vector and keyword search"""
    
    def __init__(self):
        self.vector_engine = VectorSearchEngine()
        self.keyword_engine = keyword_manager
    
    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[dict]:
        """
        Hybrid search combining vector + keyword
        
        Args:
            query: Search query
            user_id: User ID
            top_k: Number of results to return
            vector_weight: Weight for vector search (0-1)
            keyword_weight: Weight for keyword search (0-1)
        
        Returns:
            Top-k combined and reranked results
        """
        
        # Run both searches in parallel
        import asyncio
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
        
        # Sort by combined score
        deduped.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top-k
        return deduped[:top_k]
    
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
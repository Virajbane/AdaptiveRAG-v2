import asyncio
from typing import List
from app.services.retrieval.vector_search import VectorSearchEngine
from app.services.retrieval.keyword_search import keyword_manager
from app.services.retrieval.reranker import bge_reranker


class HybridSearchEngine:
    """
    Hybrid search: BM25 + Vector + Reciprocal Rank Fusion (RRF).
    RRF is rank-based, not score-based — immune to score scale mismatches
    between BM25 and cosine similarity. Industry standard for hybrid RAG.
    """

    RRF_K = 60  # standard constant, prevents top rank from dominating

    def __init__(self):
        self.vector_engine = VectorSearchEngine()
        self.keyword_engine = keyword_manager
        self.reranker = bge_reranker

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 6,
        rerank_pool_size: int = 20
    ) -> List[dict]:

        # Run both searches in parallel, fetch more candidates
        vector_results, keyword_results = await asyncio.gather(
            self.vector_engine.search(query, user_id, top_k=20),
            self.keyword_engine.search(user_id, query, top_k=20)
        )

        # DEBUG
        print(f"[HYBRID DEBUG] user_id: {user_id}")
        print(f"[HYBRID DEBUG] vector results: {len(vector_results)}")
        print(f"[HYBRID DEBUG] keyword results: {len(keyword_results)}")
        print(f"[HYBRID DEBUG] BM25 indexed users: {list(self.keyword_engine.user_indexes.keys())}")

        # Fuse with RRF
        fused = self._reciprocal_rank_fusion(vector_results, keyword_results)

        # Rerank top candidates with BGE cross-encoder if available
        candidate_pool = fused[:rerank_pool_size]

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

        return candidate_pool[:top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[dict],
        keyword_results: List[dict],
    ) -> List[dict]:
        """
        RRF formula: score(d) = sum(1 / (k + rank(d)))
        Each result list contributes a rank-based score.
        A chunk appearing in both lists gets scores from both — no weighting needed.
        """
        scores = {}   # key -> rrf_score
        docs = {}     # key -> doc dict

        for rank, doc in enumerate(vector_results):
            key = f"{doc['doc_id']}:{doc['chunk_index']}"
            scores[key] = scores.get(key, 0) + 1 / (self.RRF_K + rank + 1)
            docs[key] = {**doc, 'vector_score': doc['score'], 'keyword_score': 0}

        for rank, doc in enumerate(keyword_results):
            key = f"{doc['doc_id']}:{doc['chunk_index']}"
            scores[key] = scores.get(key, 0) + 1 / (self.RRF_K + rank + 1)
            if key in docs:
                docs[key]['keyword_score'] = max(0, doc['score'])
            else:
                docs[key] = {**doc, 'vector_score': 0, 'keyword_score': max(0, doc['score'])}

        # Attach RRF score as combined_score (keeps downstream code unchanged)
        for key, doc in docs.items():
            doc['combined_score'] = scores[key]

        return sorted(docs.values(), key=lambda x: x['combined_score'], reverse=True)

    def _deduplicate(self, results: List[dict]) -> List[dict]:
        seen = {}
        for result in results:
            key = f"{result['doc_id']}:{result['chunk_index']}"
            if key not in seen or result['combined_score'] > seen[key]['combined_score']:
                seen[key] = result
        return list(seen.values())
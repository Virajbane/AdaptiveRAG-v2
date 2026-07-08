from typing import List
from app.config.settings import settings

class BGEReranker:
    """BGE Cross-Encoder for result reranking"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Initialize reranker
        Note: First load will download model (~500MB)

        2026-07-08: skip loading entirely when settings.ENABLE_RERANKER
        is False. Added because the ~500MB model load pushes memory
        usage past Render free tier's 512MB cap, crashing the whole
        process with an OOM kill right after startup - not a caught
        exception, so the existing try/except never even got a chance
        to run. Setting ENABLE_RERANKER=false in that environment skips
        the load attempt entirely; hybrid search (BM25 + vector + RRF)
        still runs normally without the reranking pass.
        """
        if not settings.ENABLE_RERANKER:
            print("Reranker disabled via ENABLE_RERANKER=false — skipping model load.")
            self.available = False
            return

        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.available = True
        except Exception as e:
            print(f"Warning: Could not load reranker model: {e}")
            self.available = False
    
    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: int = 5
    ) -> List[dict]:
        """
        Rerank documents using BGE cross-encoder
        
        Args:
            query: Search query
            documents: List of documents with 'text' field
            top_k: Number of results to keep
        
        Returns:
            Reranked documents with scores
        """
        if not self.available or not documents:
            return documents[:top_k]
        
        # Extract texts
        doc_texts = [doc['text'] for doc in documents]
        
        # Create query-document pairs
        pairs = [[query, doc_text] for doc_text in doc_texts]
        
        # Score pairs
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score
        documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return documents[:top_k]


# Module-level singleton, same convention as web_search_tool in
# services/tools/web_search.py. Loaded once at import time (the
# ~500MB model download/load only happens once for the process'
# lifetime), rather than being re-created every time a
# HybridSearchEngine is instantiated.
#
# Unlike web_search_tool, this does NOT need a separate init_*()
# function called from main.py's startup event - CrossEncoder()
# loading is a one-time local model load with no API key or async
# setup required, so it's safe to construct directly at import time.
bge_reranker = BGEReranker()
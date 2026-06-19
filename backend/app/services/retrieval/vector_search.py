from app.db.qdrant.client import QdrantVectorDB
from app.services.document.embedder import EmbeddingGenerator
from typing import List

class VectorSearchEngine:
    """Vector search using Qdrant"""
    
    def __init__(self):
        self.vector_db = QdrantVectorDB()
        self.embedder = EmbeddingGenerator()
    
    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 10
    ) -> List[dict]:
        """
        Search using vector similarity
        
        Args:
            query: Search query string
            user_id: User ID (for isolation)
            top_k: Number of results
        
        Returns:
            List of similar chunks with scores
        """
        # Embed the query
        query_embedding = await self.embedder.embed_text(query)
        
        # Search Qdrant
        results = await self.vector_db.search(
            query_vector=query_embedding,
            user_id=user_id,
            top_k=top_k
        )
        
        # Add search type
        for result in results:
            result['search_type'] = 'vector'
        
        return results
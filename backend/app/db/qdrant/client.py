from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List
import uuid

class QdrantVectorDB:
    """Qdrant vector database client"""
    
    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url)
        self.collection_name = "documents_embeddings"
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection_name)
        except:
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,  # nomic-embed-text dimension
                    distance=Distance.COSINE
                )
            )
            print(f"Created Qdrant collection: {self.collection_name}")
    
    async def store_vectors(
        self,
        doc_id: str,
        user_id: str,
        chunks: List[dict],
        embeddings: List[List[float]]
    ) -> int:
        """
        Store document chunks as vectors in Qdrant
        
        Args:
            doc_id: Document ID
            user_id: User ID (for namespace isolation)
            chunks: List of chunk dicts with 'text' and 'tokens'
            embeddings: List of embedding vectors
        
        Returns:
            Number of vectors stored
        """
        points = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "chunk_index": i,
                    "chunk_text": chunk["text"],
                    "tokens": chunk["tokens"],
                    "namespace": f"user_{user_id}"
                }
            )
            points.append(point)
        
        # Upsert points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return len(points)
    
    async def search(
        self,
        query_vector: List[float],
        user_id: str,
        top_k: int = 5
    ) -> List[dict]:
        """
        Search for similar vectors (user-isolated)
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter={
                "must": [
                    {
                        "key": "user_id",
                        "match": {"value": user_id}
                    }
                ]
            },
            limit=top_k
        )
        
        return [
            {
                "score": result.score,
                "doc_id": result.payload["doc_id"],
                "chunk_index": result.payload["chunk_index"],
                "text": result.payload["chunk_text"],
                "tokens": result.payload["tokens"]
            }
            for result in results
        ]
    
    async def delete_document_vectors(self, doc_id: str, user_id: str) -> int:
        """Delete all vectors for a document"""
        # Delete by filter
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector={
                "filter": {
                    "must": [
                        {
                            "key": "doc_id",
                            "match": {"value": doc_id}
                        },
                        {
                            "key": "user_id",
                            "match": {"value": user_id}
                        }
                    ]
                }
            }
        )
        return result.deleted
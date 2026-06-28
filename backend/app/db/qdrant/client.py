from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
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
        collections = self.client.get_collections()
        existing = [c.name for c in collections.collections]

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                )
            )
            print(f"Created Qdrant collection: {self.collection_name}")
        else:
            print(f"Qdrant collection already exists: {self.collection_name}")

    async def store_vectors(
        self,
        doc_id: str,
        user_id: str,
        chunks: List[dict],
        embeddings: List[List[float]]
    ) -> int:
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
        """Search for similar vectors filtered by user_id"""
        response = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            ),
            limit=top_k
        )

        return [
            {
                "score": point.score,
                "doc_id": point.payload["doc_id"],
                "chunk_index": point.payload["chunk_index"],
                "text": point.payload["chunk_text"],
                "tokens": point.payload["tokens"]
            }
            for point in response
        ]

    async def delete_document_vectors(self, doc_id: str, user_id: str) -> int:
        """Delete all vectors for a document"""
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector={
                "filter": {
                    "must": [
                        {"key": "doc_id", "match": {"value": doc_id}},
                        {"key": "user_id", "match": {"value": user_id}}
                    ]
                }
            }
        )
        return result.deleted
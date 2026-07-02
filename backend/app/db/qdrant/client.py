from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
from typing import List
import uuid

class QdrantVectorDB:
    """Qdrant vector database client"""

    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url)
        self.collection_name = "documents_embeddings"
        self._ensure_collection()

    def _ensure_collection(self):
        """
        Create collection if it doesn't exist.

        IMPORTANT: this used to swallow connection errors silently and
        fall through to create_collection() as if the collection simply
        didn't exist yet. That's wrong — a connection failure (bad URL,
        Qdrant not started, auth issue, network timeout) is a completely
        different situation from "the collection legitimately doesn't
        exist." Treating them the same means a misconfigured/unreachable
        Qdrant produces no error at all — retrieval just silently starts
        returning zero results later, with nothing in the logs pointing
        back to the real cause.

        Now: only UnexpectedResponse / ResponseHandlingException (Qdrant's
        actual client-side connection/response error types) are caught,
        and they're logged loudly with the real exception before
        re-raising — so a bad connection fails fast and visibly at
        startup instead of failing silently and showing up as a confusing
        "no documents found" symptom minutes later.
        """
        try:
            collections = self.client.get_collections()
        except (UnexpectedResponse, ResponseHandlingException) as e:
            print(
                f"[QDRANT ERROR] Failed to connect to Qdrant at startup. "
                f"Check that Qdrant is running and the URL is correct.\n"
                f"  Underlying error: {type(e).__name__}: {e}"
            )
            # Re-raise rather than falling through to create_collection().
            # A connection failure should stop startup loudly, not silently
            # continue as if everything's fine — continuing here would
            # mean every later operation on self.client also fails, just
            # without ever explaining why.
            raise

        existing = [c.name for c in collections.collections]

        if self.collection_name not in existing:
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=768,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {self.collection_name}")
            except (UnexpectedResponse, ResponseHandlingException) as e:
                print(
                    f"[QDRANT ERROR] Failed to create collection "
                    f"'{self.collection_name}': {type(e).__name__}: {e}"
                )
                raise
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
        top_k: int = 5,
        document_id: str = None,   # NEW — optional, scopes search to one doc
    ) -> List[dict]:
        """Search for similar vectors filtered by user_id, optionally also by doc_id"""

        must_conditions = [
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id)
            )
        ]

        if document_id:
            must_conditions.append(
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=document_id)
                )
            )

        response = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=Filter(must=must_conditions),
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
            points_selector=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                ]
            )
        )
        return result.status
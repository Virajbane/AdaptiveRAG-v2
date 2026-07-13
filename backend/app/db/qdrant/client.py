from app.config.settings import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
from typing import List
import uuid

class QdrantVectorDB:
    """Qdrant vector database client"""

    def __init__(self, url: str = None, api_key: str = None):
        self.client = QdrantClient(
            url=url or settings.QDRANT_URL,
            api_key=api_key or settings.QDRANT_API_KEY
        )
        self.collection_name = "documents_embeddings"
        self._ensure_collection()

    def _ensure_collection(self):
        """
        Create collection if it doesn't exist.
        (unchanged — see previous version for full comment)
        """
        try:
            collections = self.client.get_collections()
        except (UnexpectedResponse, ResponseHandlingException) as e:
            print(
                f"[QDRANT ERROR] Failed to connect to Qdrant at startup. "
                f"Check that Qdrant is running and the URL is correct.\n"
                f"  Underlying error: {type(e).__name__}: {e}"
            )
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
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> dict:
        """
        Upsert chunk vectors to Qdrant in batches, so one batch failing
        doesn't lose vectors that would have stored fine.

        Returns:
            {
                "stored_count": int,
                "failed_count": int,
                "failed_chunk_indices": [int, ...],  # ORIGINAL chunk_index
                                                       # values (from chunk["chunk_index"]),
                                                       # not positions within this call's
                                                       # `chunks` list -- matters when the
                                                       # caller has already filtered chunks
                                                       # before calling this (which
                                                       # document_processor.py now does).
            }
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                f"length mismatch -- refusing to silently truncate"
            )

        # Build all points up front. IMPORTANT: the payload's chunk_index
        # must be chunk["chunk_index"] (the ORIGINAL index stamped back in
        # DocumentProcessor.process), NOT the loop position `i` here.
        #
        # Bug this fixes: document_processor.py filters out chunks that
        # failed to embed BEFORE calling this function -- so `chunks` here
        # may already be missing entries (e.g. original chunk #3 dropped).
        # If we stored `i` (the position in this now-shorter list) as
        # chunk_index, every chunk after a gap would get the WRONG
        # chunk_index in Qdrant -- corrupting citations/search() results
        # for any document that ever had a partial embedding failure.
        # `i` is still used below for BATCH tracking (which batch a point
        # belongs to for failure reporting) -- that usage is fine, since
        # document_processor.py already un-shifts those back to real
        # chunk_index values before returning failed_chunk_indices. Only
        # the STORED PAYLOAD needed this fix.
        all_points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "chunk_index": chunk["chunk_index"],  # FIXED: was `i`
                    "chunk_text": chunk["text"],
                    "tokens": chunk["tokens"],
                    "namespace": f"user_{user_id}"
                }
            )
            # Track by chunk["chunk_index"] too, so failure reporting is
            # already in terms of the real index -- no downstream
            # un-shifting needed for this list.
            all_points.append((chunk["chunk_index"], point))

        stored_count = 0
        failed_indices = []

        for batch_start in range(0, len(all_points), batch_size):
            batch = all_points[batch_start:batch_start + batch_size]
            batch_indices = [idx for idx, _ in batch]
            batch_points = [pt for _, pt in batch]

            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
                stored_count += len(batch_points)
            except (UnexpectedResponse, ResponseHandlingException) as e:
                print(
                    f"[QDRANT ERROR] Batch upsert failed for doc_id={doc_id}, "
                    f"chunk_indices={batch_indices}: "
                    f"{type(e).__name__}: {e}"
                )
                failed_indices.extend(batch_indices)

        return {
            "stored_count": stored_count,
            "failed_count": len(failed_indices),
            "failed_chunk_indices": failed_indices,
        }

    async def search(
        self,
        query_vector: List[float],
        user_id: str,
        top_k: int = 5,
        document_id: str = None,
    ) -> List[dict]:
        """Search for similar vectors filtered by user_id, optionally also by doc_id"""

        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        ]

        if document_id:
            must_conditions.append(
                FieldCondition(key="doc_id", match=MatchValue(value=document_id))
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
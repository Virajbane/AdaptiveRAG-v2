# app/services/retrieval/bm25_bootstrap.py
"""
Shared BM25 index bootstrap logic.

Extracted from main.py's startup event so any process that needs a working
keyword_manager (the real FastAPI app, eval_rag.py, a one-off script, etc.)
calls the exact same rebuild path instead of re-implementing it. Prior to
this, eval_rag.py never rebuilt BM25 at all -- it imported the same
module-level `keyword_manager` singleton as the app, but since it's a
separate Python process, that singleton starts with empty user_indexes/
user_chunks every run, and nothing in eval_rag.py populated it. This is
independent of the earlier connect_to_mongo() fix: this function only reads
from Qdrant, never touches Mongo.
"""

import asyncio


async def rebuild_bm25_indexes():
    """Rebuild in-memory BM25 indexes from Qdrant so keyword search works
    without requiring a document re-upload or a running FastAPI app."""
    from app.db.qdrant.client import QdrantVectorDB
    from app.services.retrieval.keyword_search import keyword_manager

    try:
        qdrant = QdrantVectorDB()
        all_points = []
        offset = None
        loop = asyncio.get_event_loop()

        while True:
            result = await loop.run_in_executor(
                None,
                lambda o=offset: qdrant.client.scroll(
                    collection_name="documents_embeddings",
                    limit=100,
                    offset=o,
                    with_payload=True,
                    with_vectors=False,
                )
            )
            points, next_offset = result
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        if not all_points:
            print("INFO: No vectors in Qdrant - BM25 index empty (upload docs first)")
            return

        user_chunks: dict = {}
        for point in all_points:
            uid = point.payload.get("user_id")
            if not uid:
                continue
            chunk_text = point.payload.get("chunk_text", "")
            if not chunk_text:
                continue
            user_chunks.setdefault(uid, []).append({
                "doc_id": point.payload.get("doc_id", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "text": chunk_text,
            })

        for uid, chunks in user_chunks.items():
            await keyword_manager.rebuild_from_chunks(uid, chunks)
            print(f"SUCCESS: BM25 rebuilt for user {uid[:8]}...: {len(chunks)} chunks")

    except Exception as e:
        print(f"WARNING: BM25 rebuild failed (non-fatal): {e}")
        import traceback
        traceback.print_exc()
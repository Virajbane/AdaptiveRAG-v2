# diag_vector_pool.py
import asyncio
from app.services.retrieval.vector_search import VectorSearchEngine

async def main():
    engine = VectorSearchEngine()
    results = await engine.search(
        query="Approximately how many full-duplex dialogue instances are in Lychee-FD's final training dataset?",
        user_id="6a42254db9e494f692bbeb9e",
        top_k=20,
        document_id="6a4f3ceb757ca528ed374b2d",
    )
    print(f"Vector top-20 returned: {len(results)}")
    for i, r in enumerate(results, 1):
        marker = " <-- TARGET" if r.get("chunk_index") == 55 else ""
        print(f"  [{i}] chunk_index={r.get('chunk_index')} score={r.get('score'):.4f}{marker}")

asyncio.run(main())
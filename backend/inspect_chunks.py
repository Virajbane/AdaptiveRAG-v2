"""
inspect_chunks.py

Pulls FULL (untruncated) chunk text for a given question, straight from
the same HybridSearchEngine used in production — so we can visually
confirm whether a specific number/value actually exists in any
retrieved chunk, or was never captured as text at all.

Run:
    python inspect_chunks.py --user-id <id> --question "..." --top-k 12
"""

import argparse
import asyncio


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--search-term", default=None,
                         help="Optional: also grep chunk text for this exact string "
                              "(e.g. '4.44' or '36.0') and flag which chunks contain it.")
    args = parser.parse_args()

    from app.services.retrieval.hybrid_search import HybridSearchEngine
    from app.db.mongodb.client import connect_to_mongo, get_db
    from app.services.retrieval.bm25_bootstrap import rebuild_bm25_indexes

    await connect_to_mongo()
    await rebuild_bm25_indexes()
    db = await get_db()

    engine = HybridSearchEngine()
    results = await engine.search(
        query=args.question,
        user_id=args.user_id,
        top_k=args.top_k,
        document_id=None,
    )

    print(f"\n{'='*70}")
    print(f"QUESTION: {args.question}")
    print(f"Retrieved {len(results)} chunks (top_k={args.top_k})")
    print(f"{'='*70}\n")

    for i, doc in enumerate(results, 1):
        score = doc.get("rerank_score", doc.get("combined_score", 0.0))
        text = doc.get("text", "")
        hit_marker = ""
        if args.search_term:
            hit_marker = "  <<< CONTAINS SEARCH TERM" if args.search_term in text else ""
        print(f"--- Chunk {i}  (score={score:.4f}){hit_marker} ---")
        print(text)
        print()

    if args.search_term:
        any_hit = any(args.search_term in doc.get("text", "") for doc in results)
        print(f"{'='*70}")
        print(f"Search term {args.search_term!r} found in ANY retrieved chunk: {any_hit}")
        print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
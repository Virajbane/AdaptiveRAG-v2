# diag_failed_retrieval.py
import asyncio
from app.services.retrieval.hybrid_search import HybridSearchEngine

async def main():
    engine = HybridSearchEngine()
    for q in [
        "What accuracy did Lychee-FD achieve on TriviaQA in the speech-to-speech (S->S) setting?",
        "Approximately how many full-duplex dialogue instances are in Lychee-FD's final training dataset?",
    ]:
        print("="*80)
        print("Q:", q)
        results = await engine.search(query=q, user_id="6a42254db9e494f692bbeb9e", top_k=6)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {str(r.get('chunk_text') or r.get('text',''))[:150]}")

asyncio.run(main())
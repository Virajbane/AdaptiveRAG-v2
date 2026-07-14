"""
test_retriever_patch.py

Quick standalone check that RetrieverAgent's metric-query top_k patch is
actually working -- NOT run through eval_rag.py, because
eval_rag.py's run_retrieval_eval() calls hybrid_engine.search() directly,
bypassing RetrieverAgent (and this patch) entirely. This script goes
through RetrieverAgent itself, same as the real pipeline does.

Run:
    python test_retriever_patch.py --user-id <owner_id>
"""

import argparse
import asyncio

from app.agents.state import AgentState
from app.agents.retriever import RetrieverAgent  # adjust path if this differs
from app.services.retrieval.hybrid_search import HybridSearchEngine
from app.services.retrieval.document_resolver import resolve_document_filter
from app.db.mongodb.client import connect_to_mongo, get_db
from app.services.retrieval.bm25_bootstrap import rebuild_bm25_indexes

TEST_QUESTIONS = [
    ("retrieval_figure_utmos_stepaudio_1",
     "What UTMOS score did the Step-Audio-2 backbone achieve?",
     "4.44"),
]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    args = parser.parse_args()

    await connect_to_mongo()
    await rebuild_bm25_indexes()
    db = await get_db()

    agent = RetrieverAgent(db=db)

    print("\n" + "=" * 60)
    print("RETRIEVER AGENT PATCH TEST (goes through the real agent)")
    print("=" * 60)

    search_engine = HybridSearchEngine()

    for qid, question, expected_value in TEST_QUESTIONS:
        # --- RAW CANDIDATE POOL (before any narrowing) ---
        # Shows exactly where the expected value lands, pre- and post-
        # rerank-score, across all 12 candidates -- not just whichever
        # 5 the agent narrowed down to.
        document_id = await resolve_document_filter(question, args.user_id, db)
        raw_candidates = await search_engine.search(
            query=question, user_id=args.user_id, top_k=12, document_id=document_id,
        )
        print(f"\n{'#' * 60}")
        print(f"[{qid}] RAW 12 CANDIDATES (pre-narrowing)")
        print(f"{'#' * 60}")
        for i, doc in enumerate(raw_candidates, 1):
            score = doc.get("rerank_score", doc.get("combined_score", 0.0))
            hit = "<-- EXPECTED VALUE HERE" if expected_value in doc.get("text", "") else ""
            print(f"  {i}. score={score:.4f} {hit}")
            print(f"     FULL TEXT: {doc.get('text', '')!r}")

        # --- ACTUAL AGENT PATH (with narrowing applied) ---
        state = AgentState(question=question, user_id=args.user_id)
        state = await agent._execute(state)

        found = any(expected_value in doc.get("text", "") for doc in (state.retrieved_docs or []))
        status = "FOUND" if found else "NOT FOUND"

        print(f"\n[{qid}] expected value: {expected_value!r} -> AFTER NARROWING: {status}")
        print(f"  question: {question}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
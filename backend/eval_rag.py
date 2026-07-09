"""
eval_rag.py

Lightweight, zero-new-dependency eval harness for the RAG pipeline.

RETRIEVAL EVAL (ready to run now):
    Calls HybridSearchEngine.search() directly. A question is scored as
    "found" if ANY of the top-k retrieved chunks' text contains the
    expected keyword(s) -- matching by CONTENT, not by exact chunk_index.
    This is deliberately more robust than index-matching: your chunker's
    real tokenizer produces different chunk boundaries than any
    approximation used to build the golden set, so exact-index matching
    would silently fail even when retrieval actually works correctly.
    What matters is whether the right information came back, not which
    numbered slot it landed in.

ANSWER EVAL (needs one more piece from you):
    AnswerAgent operates on a shared AgentState populated by upstream
    agents (Rewriter -> Planner -> Retriever -> Grader -> ToolAgent ->
    Critic -> Answer), not a plain question string. This script can't
    guess that orchestration correctly, so it calls a single
    `run_pipeline_fn(question, user_id)` callable that YOU wire up below
    to however your API route actually invokes the full agent graph.
    Until PIPELINE_ENTRYPOINT is set, answer-type questions are skipped
    with a clear message rather than silently failing on a wrong
    assumption about AnswerAgent's interface.

Run:
    python eval_rag.py --golden golden_set.json --user-id <test_user_id>
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Awaitable


@dataclass
class RetrievalResult:
    id: str
    question: str
    found: bool
    rank: Optional[int]


@dataclass
class AnswerResult:
    id: str
    question: str
    answer: str
    keyword_hits: int
    keyword_total: int
    faithfulness_score: Optional[float]
    unsupported_claims: List[str] = field(default_factory=list)


def load_golden_set(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    placeholders = [it["id"] for it in items if "REPLACE" in json.dumps(it) or "SET_AFTER" in json.dumps(it)]
    if placeholders:
        print(f"[WARN] {len(placeholders)} golden set entries still contain "
              f"placeholder values and will be skipped: {placeholders}")
    return [it for it in items if "REPLACE" not in json.dumps(it) and "SET_AFTER" not in json.dumps(it)]


def _content_match(text: str, expected_keywords: List[str]) -> bool:
    """A chunk 'matches' if it contains ANY one of the expected keywords
    (case-insensitive). Using ANY (not ALL) because expected_answer_contains
    often lists alternative acceptable phrasings (e.g. "140K" or "140,000"),
    not a checklist every chunk must satisfy simultaneously."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in expected_keywords)


async def run_retrieval_eval(item: dict, hybrid_engine, user_id: str, top_k: int = 6) -> RetrievalResult:
    results = await hybrid_engine.search(
        query=item["question"],
        user_id=user_id,
        top_k=top_k,
        document_id=item.get("document_id"),
    )

    expected_keywords = item.get("expected_answer_contains", [])
    found = False
    rank = None
    for i, r in enumerate(results, start=1):
        if _content_match(r.get("text", ""), expected_keywords):
            found = True
            rank = i
            break

    return RetrievalResult(id=item["id"], question=item["question"], found=found, rank=rank)


FAITHFULNESS_JUDGE_PROMPT = """You are grading whether an AI-generated answer is fully supported by the given context.

Context:
{context}

Answer to grade:
{answer}

Instructions:
- List any claims in the Answer that are NOT directly supported by the Context.
- If every claim is supported, output exactly: NONE
- Otherwise output each unsupported claim on its own line, prefixed with "- "
- Do not explain your reasoning, just list the claims or output NONE.
"""


async def judge_faithfulness(llm, context: str, answer: str) -> tuple:
    """Uses your own LLMProvider.generate() as judge -- confirmed interface
    match, no guessing needed here."""
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(context=context, answer=answer)
    try:
        judge_response = await llm.generate(prompt)
    except Exception as e:
        print(f"[JUDGE] LLM call failed: {e}")
        return None, []

    text = judge_response.strip()
    if text.upper().startswith("NONE"):
        return 1.0, []

    claims = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
    claims = [c for c in claims if c]
    if not claims:
        return 1.0, []

    approx_answer_sentences = max(answer.count(".") + answer.count("\n"), 1)
    score = max(0.0, 1 - len(claims) / approx_answer_sentences)
    return score, claims


async def run_answer_eval(
    item: dict,
    run_pipeline_fn,
    hybrid_engine,
    llm,
    user_id: str,
) -> Optional[AnswerResult]:
    if run_pipeline_fn is None:
        print(f"[SKIP] '{item['id']}' -- no PIPELINE_ENTRYPOINT wired up yet. "
              f"AnswerAgent needs full AgentState from your orchestrator; "
              f"paste that entrypoint and I'll wire this in.")
        return None

    state = await run_pipeline_fn(item["question"], user_id)
    answer_text = state.get("answer", "") if isinstance(state, dict) else getattr(state, "answer", "")
    sources = state.get("sources", []) if isinstance(state, dict) else getattr(state, "sources", [])

    context_text = "\n\n".join(s.get("text", "") for s in sources) if sources else ""
    if not context_text:
        retrieved = await hybrid_engine.search(query=item["question"], user_id=user_id, top_k=6)
        context_text = "\n\n".join(r["text"] for r in retrieved)

    expected_keywords = item.get("expected_answer_contains", [])
    answer_lower = answer_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)

    score, unsupported = await judge_faithfulness(llm, context_text, answer_text)

    return AnswerResult(
        id=item["id"], question=item["question"], answer=answer_text,
        keyword_hits=hits, keyword_total=len(expected_keywords),
        faithfulness_score=score, unsupported_claims=unsupported,
    )


def print_report(retrieval_results, answer_results, top_k: int):
    print("\n" + "=" * 60)
    print("RETRIEVAL EVAL (content-based matching)")
    print("=" * 60)
    if retrieval_results:
        recall = sum(1 for r in retrieval_results if r.found) / len(retrieval_results)
        mrr = sum((1 / r.rank) if r.found else 0 for r in retrieval_results) / len(retrieval_results)
        print(f"Recall@{top_k}: {recall:.2%}  ({sum(1 for r in retrieval_results if r.found)}/{len(retrieval_results)})")
        print(f"MRR:        {mrr:.3f}")
        for r in retrieval_results:
            status = f"rank {r.rank}" if r.found else "NOT FOUND"
            print(f"  [{r.id}] {status} -- {r.question[:60]}")
    else:
        print("(no retrieval-type questions in golden set)")

    print("\n" + "=" * 60)
    print("ANSWER EVAL")
    print("=" * 60)
    if answer_results:
        for r in answer_results:
            kw_str = f"{r.keyword_hits}/{r.keyword_total} keywords" if r.keyword_total else "n/a"
            faith_str = f"{r.faithfulness_score:.2f}" if r.faithfulness_score is not None else "judge failed"
            print(f"  [{r.id}] keywords={kw_str}  faithfulness={faith_str}")
            if r.unsupported_claims:
                print(f"      unsupported claims flagged: {r.unsupported_claims}")
    else:
        print("(no answer-type questions ran -- see [SKIP] messages above)")
    print()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="golden_set.json")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    golden_items = load_golden_set(args.golden)
    if not golden_items:
        print("No usable golden set entries. Fill in golden_set.json first.")
        sys.exit(1)

    from app.services.retrieval.hybrid_search import HybridSearchEngine
    from app.services.llm.provider import LLMProvider  # confirmed: adjust path if this differs
    from app.db.mongodb.client import connect_to_mongo   # NEW

    await connect_to_mongo()   # NEW — without this, db=None flows through
                                # AgentOrchestrator -> build_agent_graph -> planner/retriever,
                                # breaking metadata lookup and document_resolver silently
                                # (confirmed via "[DOC_RESOLVER] db is None, skipping filter")

    hybrid_engine = HybridSearchEngine()
    llm = LLMProvider()

    # --- PIPELINE_ENTRYPOINT -------------------------------------------
    # AgentOrchestrator.process() runs the full LangGraph pipeline
    # (rewriter -> planner/retriever -> join -> grader -> route ->
    # tool_agent/no_answer/metadata_answer -> answer -> critic) and
    # returns a plain dict with "answer" / "sources" keys already —
    # matches run_answer_eval's state.get("answer","") / .get("sources",[])
    # exactly, no AgentState reconstruction needed here.
    from app.agents.orchestrator import AgentOrchestrator
    _orchestrator = AgentOrchestrator()

    async def PIPELINE_ENTRYPOINT(question: str, user_id: str):
        return await _orchestrator.process(question, user_id)
    # ---------------------------------------------------------------------

    retrieval_results = []
    answer_results = []

    for item in golden_items:
        if item["type"] == "retrieval":
            retrieval_results.append(
                await run_retrieval_eval(item, hybrid_engine, args.user_id, top_k=args.top_k)
            )
        elif item["type"] == "answer":
            result = await run_answer_eval(item, PIPELINE_ENTRYPOINT, hybrid_engine, llm, args.user_id)
            if result:
                answer_results.append(result)

    print_report(retrieval_results, answer_results, args.top_k)


if __name__ == "__main__":
    asyncio.run(main())
"""
Minimal evaluation harness for Adaptive RAG v2.

WHAT THIS CHECKS (deliberately small scope - v1):
  1. Routing accuracy: did PlannerAgent.sources_needed match what the
     question actually needs? (This is the exact bug class found in
     production on 2026-06-30 - "what are my skills in docs mentioned"
     routed to ['web'] instead of ['documents'].)
  2. Basic faithfulness: does the final answer avoid forbidden phrases
     (e.g. "I don't have documents" when documents WERE retrieved) and
     contain expected keywords when specified.

WHAT THIS DOES NOT DO YET (intentional - see PROGRESS_TRACKER.md item #2):
  - No LLM-as-judge scoring of faithfulness/groundedness
  - No precision/recall over retrieval rankings
  - Not wired into CI

HOW TO RUN:
  cd backend
  python -m eval.run_eval

REQUIRES:
  - Backend's normal runtime deps (Mongo, Redis, Qdrant, Ollama) all running,
    same as `uvicorn app.main:app` - this hits the REAL pipeline, not mocks.
  - backend/.env populated (same file your app already uses).
  - dataset.json's "user_id" field set to a real user_id with indexed docs.
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Allow running as `python -m eval.run_eval` from backend/ root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.mongodb.client import connect_to_mongo, close_mongo_connection, get_db
from app.services.memory.redis_client import redis_client
from app.services.memory.long_term import LongTermMemory
from app.services.memory.manager import MemoryManager
import app.services.memory.manager as mm_module
from app.services.tools.web_search import init_web_search
from app.config.settings import settings
from app.agents.orchestrator import AgentOrchestrator
from app.agents.state import AgentState


DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"


@dataclass
class CaseResult:
    case_id: str
    question: str
    category: str
    expected_sources: list
    actual_sources: list
    routing_pass: bool
    answer: str
    confidence: float
    faithfulness_pass: bool
    faithfulness_notes: list = field(default_factory=list)
    error: str = ""
    elapsed_s: float = 0.0

    @property
    def overall_pass(self) -> bool:
        if self.error:
            return False
        if self.category == "routing":
            return self.routing_pass
        if self.category == "faithfulness":
            return self.faithfulness_pass
        return self.routing_pass and self.faithfulness_pass


async def _startup():
    """Mirror app.main's startup sequence so the eval run uses the exact
    same initialization path as the real server - no shortcuts that could
    make eval results diverge from production behavior."""
    await connect_to_mongo()
    await redis_client.connect()
    init_web_search(settings.TAVILY_API_KEY)

    db = await get_db()
    if db is not None:
        long_term_mem = LongTermMemory(db)
        memory_mgr = MemoryManager(long_term_mem)
        mm_module.memory_manager = memory_mgr
        mm_module.long_term_memory = long_term_mem


async def _shutdown():
    await close_mongo_connection()
    await redis_client.disconnect()


def _check_faithfulness(answer: str, must_contain: list, must_not_contain: list) -> tuple:
    """Case-insensitive substring checks. Returns (passed, notes)."""
    notes = []
    passed = True
    answer_lower = answer.lower()

    for phrase in must_contain:
        if phrase.lower() not in answer_lower:
            passed = False
            notes.append(f"MISSING expected phrase: {phrase!r}")

    for phrase in must_not_contain:
        if phrase.lower() in answer_lower:
            passed = False
            notes.append(f"FOUND forbidden phrase: {phrase!r}")

    return passed, notes


async def run_case(orchestrator: AgentOrchestrator, case: dict, user_id: str) -> CaseResult:
    """
    Run one eval case through the REAL orchestrator.

    NOTE: AgentOrchestrator.process() doesn't currently return
    sources_needed in its result dict (see orchestrator.py - result only
    has answer/sources/confidence/search_time_ms/is_valid). To capture
    routing decisions for this harness without modifying production
    return shape, we bypass process()'s caching wrapper and invoke the
    graph directly, mirroring process()'s own logic minus the cache.
    This keeps the eval honest (same graph, same agents) while still
    being able to inspect intermediate state.
    """
    t0 = time.perf_counter()
    try:
        initial_state = AgentState(question=case["question"], user_id=user_id)
        raw = await orchestrator.graph.ainvoke(initial_state)
        final_state = AgentState(**raw) if isinstance(raw, dict) else raw

        elapsed = time.perf_counter() - t0

        if final_state.error:
            return CaseResult(
                case_id=case["id"],
                question=case["question"],
                category=case["category"],
                expected_sources=case["expected_sources"],
                actual_sources=final_state.sources_needed,
                routing_pass=False,
                answer="",
                confidence=0.0,
                faithfulness_pass=False,
                error=final_state.error,
                elapsed_s=elapsed,
            )

        expected = set(case["expected_sources"])
        actual = set(final_state.sources_needed)
        # Routing pass = exact set match. Strict on purpose - a Planner
        # that says ["documents", "web"] when only ["documents"] was
        # needed is doing unnecessary work (and burning ~real seconds
        # on a Tavily call), so partial credit would hide that cost.
        routing_pass = expected == actual

        faithfulness_pass, faithfulness_notes = _check_faithfulness(
            final_state.answer,
            case.get("must_contain", []),
            case.get("must_not_contain", []),
        )

        return CaseResult(
            case_id=case["id"],
            question=case["question"],
            category=case["category"],
            expected_sources=case["expected_sources"],
            actual_sources=list(final_state.sources_needed),
            routing_pass=routing_pass,
            answer=final_state.answer,
            confidence=round(final_state.confidence_final, 3),
            faithfulness_pass=faithfulness_pass,
            faithfulness_notes=faithfulness_notes,
            elapsed_s=elapsed,
        )

    except Exception as e:
        elapsed = time.perf_counter() - t0
        return CaseResult(
            case_id=case["id"],
            question=case["question"],
            category=case.get("category", "unknown"),
            expected_sources=case.get("expected_sources", []),
            actual_sources=[],
            routing_pass=False,
            answer="",
            confidence=0.0,
            faithfulness_pass=False,
            error=f"{type(e).__name__}: {e}",
            elapsed_s=elapsed,
        )


def _print_report(results: list):
    print("\n" + "=" * 70)
    print("EVAL REPORT")
    print("=" * 70)

    for r in results:
        status = "PASS" if r.overall_pass else "FAIL"
        print(f"\n[{status}] {r.case_id}  ({r.elapsed_s:.1f}s)")
        print(f"  Question: {r.question}")
        if r.error:
            print(f"  ERROR: {r.error}")
            continue
        print(f"  Expected sources: {r.expected_sources}  |  Actual: {r.actual_sources}  "
              f"-> routing {'OK' if r.routing_pass else 'MISMATCH'}")
        if r.category == "faithfulness" or r.faithfulness_notes:
            print(f"  Faithfulness: {'OK' if r.faithfulness_pass else 'FAILED'}")
            for note in r.faithfulness_notes:
                print(f"    - {note}")
        print(f"  Confidence: {r.confidence}")
        print(f"  Answer (first 150 chars): {r.answer[:150]!r}")

    total = len(results)
    passed = sum(1 for r in results if r.overall_pass)
    routing_cases = [r for r in results if r.category == "routing"]
    routing_correct = sum(1 for r in routing_cases if r.routing_pass)

    print("\n" + "-" * 70)
    print(f"TOTAL: {passed}/{total} cases passed")
    if routing_cases:
        print(f"ROUTING ACCURACY: {routing_correct}/{len(routing_cases)} "
              f"({100 * routing_correct / len(routing_cases):.0f}%)")
    print("=" * 70 + "\n")


async def main():
    with open(DATASET_PATH) as f:
        data = json.load(f)

    user_id = os.environ.get("EVAL_USER_ID") or data.get("user_id")
    if not user_id:
        print(
            "ERROR: No user_id set. Either add a top-level \"user_id\" key to "
            "dataset.json, or run with EVAL_USER_ID=<your_user_id> python -m eval.run_eval\n"
            "Use a user_id that already has documents indexed (check startup "
            "logs for 'BM25 rebuilt for user ...')."
        )
        sys.exit(1)

    print(f"Running eval against user_id={user_id}")
    print(f"Loaded {len(data['cases'])} case(s) from {DATASET_PATH.name}\n")

    await _startup()
    try:
        orchestrator = AgentOrchestrator()
        results = []
        for case in data["cases"]:
            print(f"Running: {case['id']}...")
            result = await run_case(orchestrator, case, user_id)
            results.append(result)
        _print_report(results)
    finally:
        await _shutdown()


if __name__ == "__main__":
    asyncio.run(main())

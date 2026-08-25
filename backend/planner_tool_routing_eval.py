"""
Planner + Tool Routing Evaluation Suite
=========================================

Comprehensive testing of ONLY:
  User Question → Rewriter → Planner → Tool Router → Tool Execution → Answer

Does NOT test:
  - Document retrieval, embeddings, BM25, reranking, Qdrant
  - RAG quality, faithfulness, hallucination detection
  - Answer quality (only whether tool result was used)

Main goal:
  1. Did planner choose the correct source?
  2. Did router select the correct concrete tool?
  3. Did the tool actually execute?
  4. Did the answer use the tool result?
  5. Was document retrieval avoided for non-document queries?

Test coverage:
  - Calculator (5+ tests)
  - Weather (8+ tests with different intents)
  - Web (5+ tests for current/external info)
  - Database (5+ tests or decline gracefully)
  - Direct LLM (5+ general knowledge tests)
  - Ambiguous/negative tests (routing correctness under confusion)
  - Rewriter evaluation (does rewriting preserve routing intent?)
  - Tool execution verification
  - Document retrieval isolation
  - Structured planner output validation

Run from backend directory:
  python planner_tool_routing_eval.py --user-id YOUR_USER_ID [--skip-slow]
   python planner_tool_routing_eval.py --user-id YOUR_USER_ID --planner-only

Output:
  - Console summary with PASS/FAIL per test
  - Machine-readable JSON: planner_tool_routing_results.json
  - Test matrix with metrics
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import re
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Result tracking
# ============================================================================

@dataclass
class RoutingTestResult:
    """Single routing test result."""
    category: str
    test_id: str
    question: str
    rewritten_question: Optional[str]
    expected_source: str
    actual_source: Optional[str]
    expected_tool: Optional[str]
    actual_tool: Optional[str]
    tool_executed: bool
    tool_result_present: bool
    answer_present: bool
    document_retrieval_executed: bool
    passed: bool
    failure_type: str = ""  # REWRITER, PLANNER, ROUTER, TOOL_SELECTION, TOOL_EXECUTION, ANSWER_GENERATION, DOCUMENT_LEAK, OTHER
    root_cause: str = ""
    latency_s: Optional[float] = None


RESULTS: List[RoutingTestResult] = []


def record_test(
    category: str,
    test_id: str,
    question: str,
    rewritten_question: Optional[str],
    expected_source: str,
    actual_source: Optional[str],
    expected_tool: Optional[str],
    actual_tool: Optional[str],
    tool_executed: bool,
    tool_result_present: bool,
    answer_present: bool,
    document_retrieval_executed: bool,
    passed: bool,
    failure_type: str = "",
    root_cause: str = "",
    latency_s: Optional[float] = None,
) -> None:
    """Record a test result."""
    result = RoutingTestResult(
        category=category,
        test_id=test_id,
        question=question,
        rewritten_question=rewritten_question,
        expected_source=expected_source,
        actual_source=actual_source,
        expected_tool=expected_tool,
        actual_tool=actual_tool,
        tool_executed=tool_executed,
        tool_result_present=tool_result_present,
        answer_present=answer_present,
        document_retrieval_executed=document_retrieval_executed,
        passed=passed,
        failure_type=failure_type,
        root_cause=root_cause,
        latency_s=latency_s,
    )
    RESULTS.append(result)

    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n[{status}] {test_id}")
    print(f"       Category:  {category}")
    print(f"       Question:  {question[:70]}")
    if rewritten_question and rewritten_question != question:
        print(f"       Rewritten: {rewritten_question[:70]}")
    print(f"       Expected:  source={expected_source}, tool={expected_tool}")
    print(f"       Actual:    source={actual_source}, tool={actual_tool}")
    if tool_executed:
        print(f"       Tool execution: YES")
    if document_retrieval_executed:
        print(f"       ⚠️  Document retrieval executed (may be unnecessary)")
    if failure_type:
        print(f"       Failure type: {failure_type}")
    if root_cause:
        print(f"       Root cause: {root_cause}")
    if latency_s is not None:
        print(f"       Latency: {latency_s:.2f}s")


# ============================================================================
# Helper functions
# ============================================================================

def state_get(state: Any, key: str, default: Any = None) -> Any:
    """Get value from state object or dict."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def normalize_sources(value: Any) -> List[str]:
    """Normalize various source value types to list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, list):
        return [str(x) for x in value]
    return []


def call_sync_or_async(fn, *args, **kwargs):
    """Call fn synchronously or asynchronously as needed."""
    value = fn(*args, **kwargs)
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


async def call_maybe_async(fn, *args, **kwargs):
    """Call fn, awaiting if needed."""
    value = fn(*args, **kwargs)
    if inspect.isawaitable(value):
        return await value
    return value


# ============================================================================
# Planner invocation (reuses stage10_eval.py logic)
# ============================================================================

def _build_minimal_state(question: str, user_id: str) -> Any:
    """Build minimal AgentState for planner testing."""
    from app.agents.state import AgentState
    return AgentState(
        question=question,
        rewritten_question=question,
        user_id=user_id,
    )


async def invoke_planner(
    planner: Any, question: str, state: Any = None, user_id: str = "routing_test_user"
) -> Any:
    """
    Invoke planner with proper fallback order.
    
    Prefers _execute(state) first (production routing path) before
    falling back to other methods.
    """
    # Try production entry point first
    execute = getattr(planner, "_execute", None)
    if execute is not None:
        try:
            exec_state = state if state is not None else _build_minimal_state(question, user_id)
            return await call_maybe_async(execute, exec_state)
        except Exception as exc:
            print(f"[PLANNER] _execute() failed: {exc!r}, trying fallbacks...")

    # Fallback candidates
    for name in ["run", "execute", "classify_sources", "_classify_sources"]:
        fn = getattr(planner, name, None)
        if fn is None:
            continue
        try:
            return await call_maybe_async(fn, question)
        except TypeError:
            if state is not None:
                return await call_maybe_async(fn, state)
            raise

    raise RuntimeError("Could not invoke planner with any known method")


def extract_sources_from_planner_result(result: Any) -> List[str]:
    """Extract planner sources from various result formats."""
    if result is None:
        return []

    # Direct list/tuple
    if isinstance(result, (list, tuple)):
        return normalize_sources(result)

    # JSON string
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            return extract_sources_from_planner_result(parsed)
        except Exception:
            match = re.search(
                r'"sources(?:_needed)?"\s*:\s*\[(.*?)\]',
                result,
                flags=re.DOTALL,
            )
            if match:
                return re.findall(r'"([^"]+)"', match.group(1))
            return []

    # Dict / AgentState / object
    # Check sources_needed FIRST (planner's actual output field)
    for key in ("sources_needed", "sources"):
        value = state_get(result, key)
        if value:
            return normalize_sources(value)

    # Check nested plan
    plan = state_get(result, "plan")
    if plan:
        return extract_sources_from_planner_result(plan)

    return []


# ============================================================================
# Test case definitions
# ============================================================================
#
# SINGLE SOURCE OF TRUTH:
# planner_decision_eval_tests.json
#
# The Python evaluator intentionally contains NO hardcoded routing questions.
# All Calculator, Weather, Web, Database, Slack, Direct LLM, and
# Ambiguous/Negative routing questions are loaded from PLAN_* entries
# in the JSON file.
# ============================================================================

PLANNER_TEST_FILE = "planner_decision_eval_tests.json"


def load_planner_decision_tests(path: str = PLANNER_TEST_FILE) -> List[Dict[str, Any]]:
    """Load the complete planner-decision JSON suite."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    tests = payload.get("tests", [])
    if not isinstance(tests, list) or not tests:
        raise ValueError(f"No planner tests found in {path}")

    normalized = []
    for case in tests:
        decision = case.get("expected_planner_decision", {})
        normalized.append({
            "id": case["id"],
            "question": case["question"],
            "expected_source": decision.get("source"),
            "expected_tool": decision.get("tool"),
            "not_expected_sources": case.get("must_not_select", []),
            "reason": case.get("reason", ""),
        })
    return normalized


def load_routing_test_cases(path: str = PLANNER_TEST_FILE) -> List[Dict[str, Any]]:
    """Load only PLAN_* cases from the shared JSON suite."""
    return [
        case for case in load_planner_decision_tests(path)
        if case["id"].startswith("PLAN_")
    ]


def routing_category(test_id: str) -> str:
    """Map PLAN_* IDs to the report category."""
    if test_id.startswith("PLAN_CALC_"):
        return "Calculator"
    if test_id.startswith("PLAN_WEATHER_"):
        return "Weather"
    if test_id.startswith("PLAN_WEB_"):
        return "Web Search"
    if test_id.startswith("PLAN_DB_"):
        return "Database"
    if test_id.startswith("PLAN_SLACK_"):
        return "Slack"
    if test_id.startswith("PLAN_LLM_"):
        return "Direct LLM"
    if test_id.startswith("PLAN_AMB_"):
        return "Ambiguous/Negative"
    return "Planner Routing"


# ============================================================================
# Test execution
# ============================================================================


async def test_planner_only(
    planner: Any,
    user_id: str,
    test_cases: List[Dict[str, Any]],
) -> None:
    """
    Evaluate planner decisions only, using the shared JSON test suite.

    No orchestrator/tool execution is performed here. This verifies that the
    planner selected the expected source and, when the planner exposes it,
    the expected concrete tool.
    """
    print(f"\n{'='*80}")
    print(f"Testing: Planner Decision ({len(test_cases)} tests)")
    print(f"{'='*80}")

    for case in test_cases:
        test_id = case["id"]
        question = case["question"]
        expected_source = case["expected_source"]
        expected_tool = case.get("expected_tool")
        forbidden_sources = case.get("not_expected_sources") or []

        try:
            t0 = time.perf_counter()
            planner_state = await invoke_planner(
                planner,
                question,
                user_id=user_id,
            )
            latency = time.perf_counter() - t0

            actual_sources = extract_sources_from_planner_result(planner_state)
            actual_source = actual_sources[0] if actual_sources else None

            # Different planner implementations expose the selected tool under
            # different state keys. Support the common representations without
            # changing the production planner.
            actual_tool = None
            for key in (
                "tool",
                "selected_tool",
                "tool_name",
                "selected_tool_name",
                "planned_tool",
            ):
                value = state_get(planner_state, key)
                if value:
                    actual_tool = str(value)
                    break

            # Some planners put the decision inside plan.
            if actual_tool is None:
                plan = state_get(planner_state, "plan")
                if plan:
                    for key in (
                        "tool",
                        "selected_tool",
                        "tool_name",
                        "selected_tool_name",
                        "planned_tool",
                    ):
                        value = state_get(plan, key)
                        if value:
                            actual_tool = str(value)
                            break

            passed = expected_source in actual_sources

            if expected_tool is not None:
                # If the planner exposes a concrete tool, enforce it.
                passed = passed and actual_tool == expected_tool
            else:
                passed = passed and actual_tool in (None, "", "none", "None")

            if forbidden_sources:
                passed = passed and all(
                    forbidden not in actual_sources
                    for forbidden in forbidden_sources
                )

            failure_type = ""
            root_cause = ""

            if not passed:
                if not actual_sources:
                    failure_type = "PLANNER"
                    root_cause = "Planner returned empty sources."
                elif expected_source not in actual_sources:
                    failure_type = "PLANNER"
                    root_cause = (
                        f"Expected source={expected_source!r}, "
                        f"got {actual_sources!r}."
                    )
                elif expected_tool is not None and actual_tool != expected_tool:
                    failure_type = "TOOL_ROUTING"
                    root_cause = (
                        f"Expected tool={expected_tool!r}, "
                        f"got {actual_tool!r}."
                    )
                else:
                    bad = next(
                        (
                            forbidden
                            for forbidden in forbidden_sources
                            if forbidden in actual_sources
                        ),
                        None,
                    )
                    if bad:
                        failure_type = "PLANNER"
                        root_cause = (
                            f"Planner selected forbidden source {bad!r}."
                        )

            record_test(
                category=routing_category(test_id),
                test_id=test_id,
                question=question,
                rewritten_question=state_get(
                    planner_state, "rewritten_question"
                ),
                expected_source=expected_source,
                actual_source=actual_source,
                expected_tool=expected_tool,
                actual_tool=actual_tool,
                tool_executed=False,
                tool_result_present=False,
                answer_present=False,
                document_retrieval_executed=False,
                passed=passed,
                failure_type=failure_type,
                root_cause=root_cause,
                latency_s=latency,
            )

        except Exception as exc:
            record_test(
                category=routing_category(test_id),
                test_id=test_id,
                question=question,
                rewritten_question=None,
                expected_source=expected_source,
                actual_source=None,
                expected_tool=expected_tool,
                actual_tool=None,
                tool_executed=False,
                tool_result_present=False,
                answer_present=False,
                document_retrieval_executed=False,
                passed=False,
                failure_type="OTHER",
                root_cause=f"Exception: {exc!r}\n{traceback.format_exc(limit=2)}",
                latency_s=None,
            )


async def test_routing(
    planner: Any,
    orchestrator: Any,
    user_id: str,
    test_cases: List[Dict[str, Any]],
    category: str,
) -> None:
    """Test planner routing on a set of test cases."""
    print(f"\n{'='*80}")
    print(f"Testing: {category}")
    print(f"{'='*80}")

    for case in test_cases:
        test_id = case["id"]
        question = case["question"]
        expected_source = case["expected_source"]
        expected_tool = case.get("expected_tool")
        not_expected_source = case.get("not_expected_source")

        try:
            t0 = time.perf_counter()

            # Invoke planner
            planner_state = await invoke_planner(planner, question, user_id=user_id)
            planner_sources = set(extract_sources_from_planner_result(planner_state))

            # Invoke orchestrator end-to-end
            session_id = f"routing_test_{uuid.uuid4().hex[:8]}"
            orch_state = await orchestrator.process(
                question, user_id, session_id=session_id
            )
            latency = time.perf_counter() - t0

            # Extract what actually happened
            actual_source = planner_sources
            actual_tool = state_get(orch_state, "actual_tool")  # if populated by tool_agent
            tool_executed = bool(state_get(orch_state, "tool_results"))
            tool_result_present = bool(state_get(orch_state, "tool_results"))
            answer_present = bool(state_get(orch_state, "answer"))
            document_retrieval = bool(state_get(orch_state, "retrieved_docs"))

            # Evaluate source AND concrete tool correctness.
            passed = expected_source in actual_source

            if expected_tool is not None:
                passed = passed and actual_tool == expected_tool
            else:
                passed = passed and actual_tool in (None, "", "none", "None")

            if not_expected_source:
                passed = passed and not_expected_source not in actual_source

            # Determine failure type
            failure_type = ""
            root_cause = ""

            if not passed:
                if not actual_source:
                    failure_type = "PLANNER"
                    root_cause = f"Planner returned empty sources"
                elif expected_source not in actual_source:
                    failure_type = "PLANNER"
                    root_cause = (
                        f"Expected {expected_source}, got {actual_source}. "
                        f"Planner misclassified intent."
                    )
                elif expected_tool is not None and actual_tool != expected_tool:
                    failure_type = "TOOL_ROUTING"
                    root_cause = (
                        f"Expected tool={expected_tool!r}, got {actual_tool!r}"
                    )
                elif expected_tool is None and actual_tool not in (None, "", "none", "None"):
                    failure_type = "TOOL_ROUTING"
                    root_cause = (
                        f"Expected no concrete tool, got {actual_tool!r}"
                    )
                elif not_expected_source in actual_source:
                    failure_type = "PLANNER"
                    root_cause = (
                        f"Planner incorrectly selected forbidden source "
                        f"{not_expected_source}"
                    )

            # Document isolation check
            if document_retrieval and expected_source != "documents":
                if not failure_type:
                    failure_type = "DOCUMENT_LEAK"
                    root_cause = (
                        f"Documents were retrieved even though "
                        f"expected source is {expected_source}"
                    )
                passed = False

            record_test(
                category=category,
                test_id=test_id,
                question=question,
                rewritten_question=state_get(planner_state, "rewritten_question"),
                expected_source=expected_source,
                actual_source=list(actual_source)[0] if actual_source else None,
                expected_tool=expected_tool,
                actual_tool=actual_tool,
                tool_executed=tool_executed,
                tool_result_present=tool_result_present,
                answer_present=answer_present,
                document_retrieval_executed=document_retrieval,
                passed=passed,
                failure_type=failure_type,
                root_cause=root_cause,
                latency_s=latency,
            )

        except Exception as exc:
            record_test(
                category=category,
                test_id=test_id,
                question=question,
                rewritten_question=None,
                expected_source=expected_source,
                actual_source=None,
                expected_tool=expected_tool,
                actual_tool=None,
                tool_executed=False,
                tool_result_present=False,
                answer_present=False,
                document_retrieval_executed=False,
                passed=False,
                failure_type="OTHER",
                root_cause=f"Exception: {exc!r}\n{traceback.format_exc(limit=2)}",
                latency_s=None,
            )


# ============================================================================
# Report generation
# ============================================================================

def print_test_matrix() -> None:
    """Print results as a table."""
    print("\n" + "="*140)
    print("TEST MATRIX")
    print("="*140)

    header = (
        f"{'#':<3} | {'ID':<15} | {'Category':<15} | "
        f"{'Expected':<15} | {'Actual':<15} | {'Tool':<15} | "
        f"{'Exec':<5} | {'Docs':<5} | {'Status':<6}"
    )
    print(header)
    print("-" * 140)

    for i, r in enumerate(RESULTS, 1):
        status = "PASS" if r.passed else "FAIL"
        tool_exec = "Y" if r.tool_executed else "N"
        docs = "Y" if r.document_retrieval_executed else "N"

        row = (
            f"{i:<3} | {r.test_id:<15} | {r.category:<15} | "
            f"{r.expected_source:<15} | {r.actual_source or 'None':<15} | "
            f"{r.expected_tool or '-':<15} | "
            f"{tool_exec:<5} | {docs:<5} | {status:<6}"
        )
        print(row)

    print("=" * 140)


def compute_metrics() -> Dict[str, float]:
    """Compute summary metrics."""
    if not RESULTS:
        return {}

    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r.passed)

    planner_correct = sum(
        1 for r in RESULTS
        if r.expected_source == r.actual_source
    )

    tool_correct = sum(
        1 for r in RESULTS
        if r.expected_tool is not None and r.actual_tool == r.expected_tool
    )

    tool_tests = sum(
        1 for r in RESULTS
        if r.expected_tool is not None
    )

    no_docs_leaked = sum(
        1 for r in RESULTS
        if r.expected_source != "documents" and not r.document_retrieval_executed
    )

    doc_isolation_tests = sum(
        1 for r in RESULTS
        if r.expected_source != "documents"
    )

    answer_present = sum(1 for r in RESULTS if r.answer_present)

    return {
        "Total Tests": total,
        "Passed": passed,
        "Failed": total - passed,
        "Pass Rate %": (passed / total * 100) if total else 0,
        "Planner Accuracy %": (planner_correct / total * 100) if total else 0,
        "Tool Selection Accuracy %": (tool_correct / tool_tests * 100) if tool_tests else 0,
        "Answer Generation %": (answer_present / total * 100) if total else 0,
        "Document Isolation %": (no_docs_leaked / doc_isolation_tests * 100) if doc_isolation_tests else 0,
    }


def print_summary_report() -> None:
    """Print summary statistics and metrics."""
    metrics = compute_metrics()

    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)

    for metric, value in metrics.items():
        if "%" in metric:
            print(f"{metric:<30}: {value:.1f}%")
        else:
            print(f"{metric:<30}: {value}")

    # Failure classification
    print("\n" + "="*80)
    print("FAILURE CLASSIFICATION")
    print("="*80)

    failures = [r for r in RESULTS if not r.passed]
    if not failures:
        print("✓ All tests passed!")
    else:
        failure_counts = {}
        for r in failures:
            ft = r.failure_type or "UNKNOWN"
            failure_counts[ft] = failure_counts.get(ft, 0) + 1

        for failure_type, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"  {failure_type:<20}: {count}")

        print("\nFailed tests:")
        for r in failures:
            print(f"\n  [{r.test_id}] {r.question[:60]}")
            print(f"    Expected: {r.expected_source} (tool: {r.expected_tool})")
            print(f"    Actual:   {r.actual_source} (tool: {r.actual_tool})")
            if r.failure_type:
                print(f"    Type:     {r.failure_type}")
            if r.root_cause:
                print(f"    Cause:    {r.root_cause[:100]}")

    # Category breakdown
    print("\n" + "="*80)
    print("RESULTS BY CATEGORY")
    print("="*80)

    categories = {}
    for r in RESULTS:
        if r.category not in categories:
            categories[r.category] = {"passed": 0, "total": 0}
        categories[r.category]["total"] += 1
        if r.passed:
            categories[r.category]["passed"] += 1

    for category in sorted(categories.keys()):
        stats = categories[category]
        pct = (stats["passed"] / stats["total"] * 100) if stats["total"] else 0
        print(f"  {category:<20}: {stats['passed']}/{stats['total']} passed ({pct:.0f}%)")


def save_json_report(path: str = "planner_tool_routing_results.json") -> None:
    """Save machine-readable JSON report."""
    metrics = compute_metrics()

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "results": [asdict(r) for r in RESULTS],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Results saved to: {path}")


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Planner + Tool Routing Evaluation Suite"
    )
    parser.add_argument("--user-id", required=True, help="User ID for testing")
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow tests (web search, database, Slack)",
    )
    parser.add_argument(
        "--planner-only",
        action="store_true",
        help=(
            "Run only planner decision tests from "
            "planner_decision_eval_tests.json. No tools, retrieval, "
            "or answer generation."
        ),
    )
    parser.add_argument(
        "--planner-test-file",
        default=PLANNER_TEST_FILE,
        help="Path to the shared planner/routing JSON test suite.",
    )
    args = parser.parse_args()

    # Initialize application
    print("Initializing application...")
    try:
        from app.db.mongodb.client import connect_to_mongo
        from app.services.retrieval.bm25_bootstrap import rebuild_bm25_indexes
        from app.services.tools.web_search import init_web_search
        from app.config.settings import settings

        await connect_to_mongo()
        print("✓ MongoDB connected")

        await rebuild_bm25_indexes()
        print("✓ BM25 indexes rebuilt")

        if settings.TAVILY_API_KEY:
            init_web_search(settings.TAVILY_API_KEY)
            print("✓ Web search initialized")
        else:
            print("⚠️  Web search not configured (TAVILY_API_KEY missing)")

    except Exception as exc:
        print(f"❌ Initialization failed: {exc}")
        traceback.print_exc()
        return

    # Load components
    print("\nLoading components...")
    try:
        from app.agents.planner import PlannerAgent
        from app.agents.orchestrator import AgentOrchestrator
        from app.db.mongodb.client import db
        from app.services.llm.provider import LLMProvider
        from app.config.settings import settings

        fast_llm = LLMProvider(model=settings.OLLAMA_FAST_MODEL)
        planner = PlannerAgent(fast_llm, db=db)
        orchestrator = AgentOrchestrator()

        print("✓ Components loaded")
    except Exception as exc:
        print(f"❌ Component loading failed: {exc}")
        traceback.print_exc()
        return

    # Run tests
    print("\n" + "="*80)
    print("PLANNER + TOOL ROUTING EVALUATION SUITE")
    print("="*80)

    try:
        if args.planner_only:
            planner_cases = load_planner_decision_tests(
                args.planner_test_file
            )
            await test_planner_only(
                planner,
                args.user_id,
                planner_cases,
            )
        else:
            # Full pipeline routing uses the SAME JSON test file.
            routing_cases = load_routing_test_cases(args.planner_test_file)

            grouped_cases = {}
            for case in routing_cases:
                category = routing_category(case["id"])
                grouped_cases.setdefault(category, []).append(case)

            category_order = [
                "Calculator",
                "Weather",
                "Web Search",
                "Database",
                "Slack",
                "Direct LLM",
                "Ambiguous/Negative",
            ]

            for category in category_order:
                cases = grouped_cases.get(category, [])
                if not cases:
                    continue

                if args.skip_slow and category in {"Web Search", "Database", "Slack"}:
                    print(f"\n(Skipping {category} tests due to --skip-slow)")
                    continue

                await test_routing(
                    planner,
                    orchestrator,
                    args.user_id,
                    cases,
                    category,
                )

    except Exception as exc:
        print(f"\n❌ Test execution failed: {exc}")
        traceback.print_exc()
        return

    # Print reports
    print_test_matrix()
    print_summary_report()

    if args.planner_only:
        save_json_report("planner_decision_results.json")
    else:
        save_json_report("planner_tool_routing_results.json")


if __name__ == "__main__":
    asyncio.run(main())
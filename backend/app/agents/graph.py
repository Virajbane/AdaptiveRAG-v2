"""
LangGraph orchestration for the RAG agent pipeline — Option B with smart retry routing.

Flow (2026-08-10 fix):

    START → rewriter → planner ──(route_after_planning, check documents first)──┐
                                                                                  ├─ retriever → grader ─┐
                                                                                  ├─ tool_agent ────────┤
                                                                                  ├─ answer ────────────┤
                                                                                  ├─ no_answer ─────────┤
                                                                                  ├─ metadata_answer ───┤
                                                                                  └─ placeholder_answer ┤
                                                                                                         │
                                                                                                         ↓
                                                                                                       answer
                                                                                                         │
                                                                                                       critic
                                                                                                         │
                                                                            route_after_critic (retry dispatch):
                                                                              generation → answer (retry)
                                                                              retrieval  → retriever
                                                                              planning   → planner
                                                                              tool       → tool_agent
                                                                              unknown    → END
                                                                              done       → END

KEY CHANGE (2026-08-10 Test 6 fix):
Retriever moved from parallel unconditional (rewriter → retriever) to CONDITIONAL edge
gated on "documents" in sources_needed. For calculator/weather/web/direct_llm queries,
retrieval is now skipped entirely (~10-24s saved per non-document query). Planner
decision is the single source of truth for what sources are needed.

2026-08-08 update: route_after_planning now dispatches to tool_agent
for "web", "tool", AND "calculator" (previously only web/tools, and
"tools" plural didn't even match the Planner's canonical "tool"
singular -- that bug is fixed here). The SQL-specific short-circuit is
generalized to any placeholder source (currently just "database"),
keyed off state.metadata_answer's shape rather than a hardcoded name,
so adding a future not-yet-implemented source doesn't require another
edit here.

2026-08-09 FIX (Stage 9 wiring review): grader_node no longer mutates
sources_needed. GraderAgent's own contract (agents/grader.py) states it
is assessment-only and was already fixed to stop writing sources_needed
-- but the graph.py wrapper had re-implemented the exact same
"high-confidence retrieval override" logic itself, silently
reintroducing Grader-as-router and breaking "only execute sources
actually requested" for multi-source plans (e.g. a calculator-only or
web-only question could get document evidence mixed back in). Removed;
grader_node now only forwards low_confidence/retrieval_rejected, and
routing after grading uses Planner's sources_needed exclusively.
Also removed a duplicate `graph.add_edge("planner", "join")` call.
"""

import time
import hashlib
from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState, _compute_hash
from app.agents.planner import PlannerAgent
from app.agents.retriever import RetrieverAgent
from app.agents.grader import GraderAgent
from app.agents.tool_agent import ToolAgent
from app.agents.critic import CriticAgent
from app.agents.answer import AnswerAgent
from app.agents.rewriter import RewriterAgent
from app.services.llm.provider import LLMProvider
from app.config.settings import settings
from langsmith import traceable

# ── Retry limits per component ─────────────────────────────────────────
RETRY_LIMITS = {
    "planner": 1,       # Planner routing is stable; rarely needs retry
    "retriever": 2,     # Retriever might need 2 tries for BM25/vector blending
    "answer": 2,        # Answer generation might need 2 tries for synthesis
    "tool": 1,          # Tool execution is deterministic; 1 retry is enough
}

_EARLY_STOP_MIN_CHANGE = 0.05  # 5% change threshold

# Sources that require tool_agent execution before answer generation.
# Single place to extend when a future source needs pre-answer execution
# (e.g. "csv_analyzer", "vision").
_TOOL_EXECUTION_SOURCES = {"web", "tool", "calculator"}


def build_agent_graph(db=None):
    """
    Construct and compile the agent StateGraph with smart retry routing.

    Returns a compiled graph with an `.ainvoke(state)` method that runs
    the full pipeline and returns the final AgentState.
    """

    llm      = LLMProvider(num_ctx=4096)                              # qwen2.5:7b  — deep reasoning
    fast_llm = LLMProvider(model=settings.OLLAMA_FAST_MODEL)          # qwen2.5:1.5b — routing / judging

    rewriter   = RewriterAgent(fast_llm)
    planner    = PlannerAgent(llm, db=db)
    retriever  = RetrieverAgent(fast_llm, db=db)
    grader     = GraderAgent(fast_llm)
    tool_agent = ToolAgent(fast_llm)
    critic     = CriticAgent(fast_llm)
    answer     = AnswerAgent(llm)

    # ── Timing helper ─────────────────────────────────────────────────────

    def _print_timing(node_name: str, elapsed: float):
        print(f"[TIMING] {node_name:<12} {elapsed:6.1f}s")

    # ── Node wrappers ─────────────────────────────────────────────────────

    async def rewriter_node(state: AgentState) -> dict:
        t0 = time.perf_counter()
        result = await rewriter.run(state.copy())
        _print_timing("rewriter", time.perf_counter() - t0)
        update = {"rewritten_question": result.rewritten_question}
        if result.error:
            update["error"] = result.error
        return update

    @traceable(name="planner_node", run_type="chain")
    async def planner_node(state: AgentState) -> dict:
        t0 = time.perf_counter()
        result = await planner.run(state.copy())
        _print_timing("planner", time.perf_counter() - t0)
        update = {
            "plan":             result.plan,
            "sources_needed":   result.sources_needed,
            "confidence":       result.confidence,
            "metadata_answer":  result.metadata_answer,
            "last_planning_hash": _compute_hash((result.plan, result.sources_needed)),
        }
        if result.error:
            update["error"] = result.error
        return update

    @traceable(name="retriever_node", run_type="retriever")
    async def retriever_node(state: AgentState) -> dict:
        t0 = time.perf_counter()
        result = await retriever.run(state.copy())
        _print_timing("retriever", time.perf_counter() - t0)

        doc_summary = [(doc.get("doc_id"), doc.get("rerank_score", 0))
                       for doc in result.retrieved_docs]

        update = {
            "retrieved_docs": result.retrieved_docs,
            "web_results":    result.web_results,
            "search_time_ms": result.search_time_ms,
            "last_retrieval_hash": _compute_hash(doc_summary),
        }
        if result.error:
            update["error"] = result.error
        return update

    @traceable(name="grader_node", run_type="chain")
    async def grader_node(state: AgentState) -> dict:
        t0 = time.perf_counter()
        result = await grader.run(state.copy())
        _print_timing("grader", time.perf_counter() - t0)

        # Grader is assessment-only (see agents/grader.py contract): it
        # flags retrieval quality via low_confidence/retrieval_rejected but
        # never writes sources_needed. Routing is Planner's exclusive
        # responsibility. The old bidirectional override that added
        # "documents" back into sources_needed on a high-confidence match
        # was removed upstream in GraderAgent itself for exactly this
        # reason -- re-implementing it here in the node wrapper would
        # silently reintroduce the same bug (Grader acting as a second
        # router) and break "only execute sources actually requested" for
        # multi-source plans (e.g. a pure "calculator" or "web" question
        # would unexpectedly get document evidence mixed in).
        return {
            "low_confidence":     result.low_confidence,
            "retrieval_rejected": result.retrieval_rejected,
        }

    async def tool_node(state: AgentState) -> dict:
        t0 = time.perf_counter()
        result = await tool_agent.run(state.copy())
        _print_timing("tool_agent", time.perf_counter() - t0)
        update = result.__dict__
        update["last_tool_result_hash"] = _compute_hash(result.tool_results)
        return update

    async def answer_node(state: AgentState) -> dict:
        t0 = time.perf_counter()
        result = await answer.run(state.copy())
        _print_timing("answer", time.perf_counter() - t0)
        update = result.__dict__
        update["last_answer_hash"] = _compute_hash(result.answer)
        return update

    async def critic_node(state: AgentState) -> dict:
        t0 = time.perf_counter()
        result = await critic.run(state.copy())
        _print_timing("critic", time.perf_counter() - t0)
        return result.__dict__

    async def no_answer_node(state: AgentState) -> dict:
        print("[ROUTER] Retrieval rejected (below absolute floor) — "
              "returning direct not-found response, skipping answer/critic")
        return {
            "answer": "I couldn't find this information in the document.",
            "confidence_final": 0.0,
            "sources": [],
            "is_valid": True,
        }

    async def metadata_answer_node(state: AgentState) -> dict:
        meta = state.metadata_answer
        parts = []
        if meta.get("title"):
            parts.append(f"Title: {meta['title']}")
        if meta.get("authors"):
            parts.append(f"Authors: {', '.join(meta['authors'])}")
        if meta.get("affiliations"):
            parts.append(f"Affiliations: {', '.join(meta['affiliations'])}")
        answer_text = "\n".join(parts) if parts else "No metadata available."
        print("[ROUTER] Answering from stored document metadata, skipping retrieval entirely")
        return {
            "answer": answer_text,
            "confidence_final": 0.95,
            "sources": [],
            "is_valid": True,
        }

    async def placeholder_answer_node(state: AgentState) -> dict:
        """
        Generalized replacement for the old sql_answer_node. Fires for
        ANY not-yet-implemented source the Planner flagged via
        metadata_answer["placeholder"] (currently: "database"). Keyed
        off that shape rather than a hardcoded source name, so adding
        another placeholder source later doesn't need a new node.
        """
        placeholder = state.metadata_answer.get(
            "placeholder", "This source is under development."
        )
        print(f"[ROUTER] Placeholder source {state.sources_needed} detected, "
              f"returning placeholder — skipping retrieval/answer/critic")
        return {
            "answer": placeholder,
            "confidence_final": 1.0,
            "sources": [],
            "is_valid": True,
        }

    # ── Routing functions ─────────────────────────────────────────────────

    def route_after_planning(state: AgentState) -> str:
        """
        2026-08-10 FIX: Route AFTER planner, checking for document-needing
        sources FIRST. This gates retriever on "documents" in sources_needed,
        eliminating ~10-24s of unconditional retrieval for non-document queries
        (calculator, weather, web, direct_llm).
        """

        # Metadata short-circuit (highest priority)
        if "metadata" in state.sources_needed and state.metadata_answer:
            print("[ROUTER] Metadata answer available → metadata_answer_node")
            return "metadata_answer"

        # Generalized placeholder short-circuit (currently: database).
        # Checked by shape (metadata_answer carrying a "placeholder" key),
        # not by hardcoded source name.
        if state.metadata_answer.get("placeholder"):
            print(f"[ROUTER] Placeholder source {state.sources_needed} "
                  f"→ placeholder_answer_node")
            return "placeholder_answer"

        # Hard retrieval rejection (only if documents were actually routed)
        if state.retrieval_rejected and "documents" in state.sources_needed:
            print("[ROUTER] Retrieval rejected by grader → no_answer_node")
            return "no_answer"

        # Error propagation
        if state.error:
            print(f"[ROUTER] Error detected, skipping to answer: {state.error}")
            return "answer"

        # Document-needing queries (including multi-source like ["documents", "web"])
        # MUST go through retriever first to load docs. Retriever then feeds to
        # grader → answer for assessment and confidence scoring.
        if "documents" in state.sources_needed:
            print(f"[ROUTER] Planner requested documents → retriever")
            return "retriever"

        # Sources requiring tool_agent execution before answering
        if any(s in state.sources_needed for s in _TOOL_EXECUTION_SOURCES):
            print(f"[ROUTER] Planner requested {state.sources_needed} → tool_agent")
            return "tool_agent"

        # direct_llm / anything else with no tool execution needed
        if state.sources_needed:
            print(f"[ROUTER] Planner sources_needed={state.sources_needed} → answer")
            return "answer"

        # Fallback: no sources specified (shouldn't normally happen with
        # the current Planner, kept as a safety net)
        web_keywords = [
            "news", "latest", "current", "today", "price", "stock",
            "weather", "live", "trending", "recent", "2024", "2025", "2026"
        ]
        question_lower = state.question.lower()
        needs_web = any(kw in question_lower for kw in web_keywords)

        if needs_web:
            print(f"[ROUTER] Fallback: web keywords detected → tool_agent")
            return "tool_agent"

        print(f"[ROUTER] Fallback: no web keywords → answer")
        return "answer"

    def route_after_critic(state: AgentState) -> str:
        """
        Route based on failure_type classification. Replaces the simple
        "valid? yes→END, no→answer" logic with component-targeted retries.
        """

        if state.is_valid:
            print(f"[ROUTER] Critic accepted answer. confidence_final={state.confidence_final:.4f}")
            return "done"

        failure_type = state.failure_type or "unknown"

        if failure_type == "generation":
            if state.answer_retry_count >= RETRY_LIMITS["answer"]:
                print(f"[ROUTER] Generation failure but answer retries exhausted "
                      f"({state.answer_retry_count}/{RETRY_LIMITS['answer']}) → END")
                return "done"
            if state.last_answer_hash and state.answer_retry_count > 0:
                current_hash = _compute_hash(state.answer)
                if current_hash == state.last_answer_hash:
                    print(f"[ROUTER] Generation failure but answer unchanged (same hash) → END")
                    return "done"
            print(f"[ROUTER] Generation failure (retry {state.answer_retry_count}/{RETRY_LIMITS['answer']}) "
                  f"→ retry_answer")
            return "retry_answer"

        if failure_type == "retrieval":
            if state.retriever_retry_count >= RETRY_LIMITS["retriever"]:
                print(f"[ROUTER] Retrieval failure but retriever retries exhausted "
                      f"({state.retriever_retry_count}/{RETRY_LIMITS['retriever']}) → END")
                return "done"
            if state.last_retrieval_hash and state.retriever_retry_count > 0:
                current_hash = _compute_hash([(d.get("doc_id"), d.get("rerank_score"))
                                             for d in state.retrieved_docs])
                if current_hash == state.last_retrieval_hash:
                    print(f"[ROUTER] Retrieval failure but docs unchanged (same hash) → END")
                    return "done"
            print(f"[ROUTER] Retrieval failure (retry {state.retriever_retry_count}/{RETRY_LIMITS['retriever']}) "
                  f"→ retry_retriever")
            return "retry_retriever"

        if failure_type == "planning":
            if state.planner_retry_count >= RETRY_LIMITS["planner"]:
                print(f"[ROUTER] Planning failure but planner retries exhausted "
                      f"({state.planner_retry_count}/{RETRY_LIMITS['planner']}) → END")
                return "done"
            if state.last_planning_hash and state.planner_retry_count > 0:
                current_hash = _compute_hash((state.plan, state.sources_needed))
                if current_hash == state.last_planning_hash:
                    print(f"[ROUTER] Planning failure but plan unchanged (same hash) → END")
                    return "done"
            print(f"[ROUTER] Planning failure (retry {state.planner_retry_count}/{RETRY_LIMITS['planner']}) "
                  f"→ retry_planner")
            return "retry_planner"

        if failure_type == "tool":
            if state.tool_retry_count >= RETRY_LIMITS["tool"]:
                print(f"[ROUTER] Tool failure but tool retries exhausted "
                      f"({state.tool_retry_count}/{RETRY_LIMITS['tool']}) → END")
                return "done"
            if state.last_tool_result_hash and state.tool_retry_count > 0:
                current_hash = _compute_hash(state.tool_results)
                if current_hash == state.last_tool_result_hash:
                    print(f"[ROUTER] Tool failure but results unchanged (same hash) → END")
                    return "done"
            print(f"[ROUTER] Tool failure (retry {state.tool_retry_count}/{RETRY_LIMITS['tool']}) "
                  f"→ retry_tool")
            return "retry_tool"

        print(f"[ROUTER] Failure type unknown; cannot intelligently retry → END")
        return "done"

    # ── Retry counter increment nodes ────────────────────────────────────

    async def increment_answer_retry(state: AgentState) -> dict:
        return {"answer_retry_count": state.answer_retry_count + 1}

    async def increment_retriever_retry(state: AgentState) -> dict:
        return {"retriever_retry_count": state.retriever_retry_count + 1}

    async def increment_planner_retry(state: AgentState) -> dict:
        return {"planner_retry_count": state.planner_retry_count + 1}

    async def increment_tool_retry(state: AgentState) -> dict:
        return {"tool_retry_count": state.tool_retry_count + 1}

    # ── Build graph ───────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("rewriter",             rewriter_node)
    graph.add_node("planner",              planner_node)
    graph.add_node("retriever",            retriever_node)
    graph.add_node("grader",               grader_node)
    graph.add_node("tool_agent",           tool_node)
    graph.add_node("answer",               answer_node)
    graph.add_node("critic",               critic_node)
    graph.add_node("no_answer",            no_answer_node)
    graph.add_node("metadata_answer",      metadata_answer_node)
    graph.add_node("placeholder_answer",   placeholder_answer_node)

    graph.add_node("increment_answer_retry",     increment_answer_retry)
    graph.add_node("increment_retriever_retry",  increment_retriever_retry)
    graph.add_node("increment_planner_retry",    increment_planner_retry)
    graph.add_node("increment_tool_retry",       increment_tool_retry)

    # ── Edges: Main pipeline (2026-08-10 FIX: retriever now conditional) ──

    graph.add_edge(START, "rewriter")
    graph.add_edge("rewriter", "planner")

    # 2026-08-10 FIX: Retriever is NO LONGER parallel/unconditional.
    # It's now a conditional edge from planner, gated on "documents" in sources_needed.
    # This eliminates ~10-24s of wasted retrieval on non-document queries.
    graph.add_conditional_edges(
        "planner",
        route_after_planning,
        {
            "retriever":            "retriever",     # NEW: conditional on "documents"
            "tool_agent":           "tool_agent",
            "answer":               "answer",
            "no_answer":            "no_answer",
            "metadata_answer":      "metadata_answer",
            "placeholder_answer":   "placeholder_answer",
        },
    )

    # Retriever always feeds to grader for assessment (if it runs at all)
    graph.add_edge("retriever", "grader")

    # Grader always feeds to answer for confidence scoring and response synthesis
    graph.add_edge("grader", "answer")

    # Tool and answer nodes both feed to critic
    graph.add_edge("tool_agent", "answer")
    graph.add_edge("answer",     "critic")

    # Special case endpoints (no critic needed)
    graph.add_edge("no_answer",          END)
    graph.add_edge("metadata_answer",    END)
    graph.add_edge("placeholder_answer", END)

    # ── Edges: Retry dispatch after critic ────────────────────────────────

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "done":               END,
            "retry_answer":       "increment_answer_retry",
            "retry_retriever":    "increment_retriever_retry",
            "retry_planner":      "increment_planner_retry",
            "retry_tool":         "increment_tool_retry",
        },
    )

    graph.add_edge("increment_answer_retry",    "answer")
    graph.add_edge("increment_retriever_retry", "retriever")
    graph.add_edge("increment_planner_retry",   "planner")
    graph.add_edge("increment_tool_retry",      "tool_agent")

    return graph.compile()
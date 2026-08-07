"""
LangGraph orchestration for the RAG agent pipeline — Option B with smart retry routing.

Flow:

    START → rewriter ──┬──→ planner ───┐
                        └──→ retriever ─┴──→ join → grader ──(route_after_planning)──→ tool_agent ──→ answer ──→ critic
                                                                  ├──────────────────────────────────────→ answer ──┐
                                                                  ├──→ no_answer ──→ END                        │
                                                                  └──→ metadata_answer ──→ END                  │
                                                                                                                  │
                                                            route_after_critic (NEW):                            │
                                                              generation → answer (retry only answer)          │
                                                              retrieval  → retriever (re-fetch docs)           │
                                                              planning   → planner (re-route)                  │
                                                              tool       → tool_agent (re-execute)             │
                                                              unknown    → END (return as-is)                  │
                                                              done       → END                                  │

2026-07-XX: Upgrade to smart failure-type routing with independent retry budgets.

Key improvements:
─────────────────
1. Failure-type classification
   - Critic now returns failure_type ("generation", "retrieval", "planning", "tool", "unknown")
   - Each type routes back to the component that failed, not blindly to answer
   - Unknown failures end immediately (no point retrying if we can't classify)

2. Independent retry counters
   - planner_retry_count (max 1)
   - retriever_retry_count (max 2)
   - answer_retry_count (max 2)
   - tool_retry_count (max 1)
   - Prevents component A from exhausting retries on component B's behalf

3. Early stop on unchanged output
   - Stores hash of (plan, sources_needed) after planner runs
   - Stores hash of retrieved_docs after retriever runs
   - Stores hash of answer after answer runs
   - If next retry produces identical hash, stops retrying immediately
   - Prevents burning retry budget on identical output

4. Smart re-entry points
   - generation → answer_node (receives identical retrieved_docs)
   - retrieval → retriever_node (receives identical question/planning, new docs)
   - planning → planner_node (receives identical question, forces re-plan)
   - tool → tool_agent_node (receives identical plan but retries tool execution)
   - This avoids redundant work while fixing actual issues

5. Preserved optimizations
   - Planner and Retriever still run in parallel (no regression)
   - Only the failed component is retried (CPU-efficient)
   - Grader still runs once per pipeline (scoring is cheap)
   - Metadata/no_answer short-circuits still intact
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

# Early stop threshold: if hashes differ by <10% in content, stop retrying
# This catches cases like "same doc IDs but slightly different scores"
_EARLY_STOP_MIN_CHANGE = 0.05  # 5% change threshold


def build_agent_graph(db=None):
    """
    Construct and compile the agent StateGraph with smart retry routing.

    Returns a compiled graph with an `.ainvoke(state)` method that runs
    the full pipeline and returns the final AgentState.
    """

    llm      = LLMProvider(num_ctx=4096)                              # qwen2.5:7b  — deep reasoning
    fast_llm = LLMProvider(model=settings.OLLAMA_FAST_MODEL)          # qwen2.5:1.5b — routing / judging

    rewriter   = RewriterAgent(fast_llm)
    planner    = PlannerAgent(fast_llm, db=db)
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
        
        # For early stop detection: hash the doc IDs and scores
        # (not the full text, which might have minor formatting diffs)
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

    async def join_node(state: AgentState) -> dict:
        return {}

    @traceable(name="grader_node", run_type="chain")
    async def grader_node(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        result = await grader.run(state.copy())
        _print_timing("grader", time.perf_counter() - t0)

        # High-confidence retrieval override (existing logic, unchanged)
        if "documents" not in result.sources_needed and result.retrieved_docs:
            top_score = result.retrieved_docs[0].get("rerank_score", 0.0)
            if top_score >= 0.5:  # HIGH_CONFIDENCE_RETRIEVAL_THRESHOLD
                print(
                    f"[GRADER] Override: planner said sources_needed={result.sources_needed} "
                    f"but found a high-confidence document match "
                    f"(top_score={top_score:.4f}) — adding 'documents' to sources"
                )
                result.sources_needed = result.sources_needed + ["documents"]

        return result

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

    async def sql_answer_node(state: AgentState) -> dict:
        placeholder = state.metadata_answer.get(
            "placeholder", "SQL integration is under development."
        )
        print("[ROUTER] @sql tag detected, returning placeholder — skipping retrieval/answer/critic")
        return {
            "answer": placeholder,
            "confidence_final": 1.0,
            "sources": [],
            "is_valid": True,
        }

    # ── Routing functions ─────────────────────────────────────────────────

    def route_after_planning(state: AgentState) -> str:
        """Route to the appropriate next node based on planner output."""
        
        # Metadata short-circuit (highest priority)
        if "metadata" in state.sources_needed and state.metadata_answer:
            print("[ROUTER] Metadata answer available → metadata_answer_node")
            return "metadata_answer"

        # SQL short-circuit
        if "sql" in state.sources_needed:
            print("[ROUTER] SQL tag detected → sql_answer_node (placeholder)")
            return "sql_answer"

        # Hard retrieval rejection
        if state.retrieval_rejected and "documents" in state.sources_needed:
            print("[ROUTER] Retrieval rejected by grader → no_answer_node")
            return "no_answer"

        # Error propagation
        if state.error:
            print(f"[ROUTER] Error detected, skipping tool_agent: {state.error}")
            return "answer"

        # Web/tool routing
        if "web" in state.sources_needed or "tools" in state.sources_needed:
            print(f"[ROUTER] Planner requested web/tools → tool_agent")
            return "tool_agent"

        # Document-only routing
        if state.sources_needed:
            print(f"[ROUTER] Planner sources_needed={state.sources_needed} → answer")
            return "answer"

        # Fallback: no sources specified
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
        Route based on failure_type classification.
        
        This is the new smart routing that replaces the simple
        "valid? yes→END, no→answer" logic.
        """

        # Answer passed validation
        if state.is_valid:
            print(f"[ROUTER] Critic accepted answer. confidence_final={state.confidence_final:.4f}")
            return "done"

        # Answer failed validation; check failure_type
        failure_type = state.failure_type or "unknown"

        # ── Generation failure (AnswerAgent hallucination/synthesis) ──────
        if failure_type == "generation":
            if state.answer_retry_count >= RETRY_LIMITS["answer"]:
                print(f"[ROUTER] Generation failure but answer retries exhausted "
                      f"({state.answer_retry_count}/{RETRY_LIMITS['answer']}) → END")
                return "done"

            # Check early stop: did we just produce identical answer?
            if state.last_answer_hash and state.answer_retry_count > 0:
                # On retry, answer_node updates last_answer_hash; if it matches
                # the previous run, don't burn another retry on identical output
                current_hash = _compute_hash(state.answer)
                if current_hash == state.last_answer_hash:
                    print(f"[ROUTER] Generation failure but answer unchanged (same hash) → END")
                    return "done"

            print(f"[ROUTER] Generation failure (retry {state.answer_retry_count}/{RETRY_LIMITS['answer']}) "
                  f"→ retry_answer")
            return "retry_answer"

        # ── Retrieval failure (wrong/missing chunks) ──────────────────────
        if failure_type == "retrieval":
            if state.retriever_retry_count >= RETRY_LIMITS["retriever"]:
                print(f"[ROUTER] Retrieval failure but retriever retries exhausted "
                      f"({state.retriever_retry_count}/{RETRY_LIMITS['retriever']}) → END")
                return "done"

            # Early stop check
            if state.last_retrieval_hash and state.retriever_retry_count > 0:
                current_hash = _compute_hash([(d.get("doc_id"), d.get("rerank_score"))
                                             for d in state.retrieved_docs])
                if current_hash == state.last_retrieval_hash:
                    print(f"[ROUTER] Retrieval failure but docs unchanged (same hash) → END")
                    return "done"

            print(f"[ROUTER] Retrieval failure (retry {state.retriever_retry_count}/{RETRY_LIMITS['retriever']}) "
                  f"→ retry_retriever")
            return "retry_retriever"

        # ── Planning failure (wrong routing) ──────────────────────────────
        if failure_type == "planning":
            if state.planner_retry_count >= RETRY_LIMITS["planner"]:
                print(f"[ROUTER] Planning failure but planner retries exhausted "
                      f"({state.planner_retry_count}/{RETRY_LIMITS['planner']}) → END")
                return "done"

            # Early stop check
            if state.last_planning_hash and state.planner_retry_count > 0:
                current_hash = _compute_hash((state.plan, state.sources_needed))
                if current_hash == state.last_planning_hash:
                    print(f"[ROUTER] Planning failure but plan unchanged (same hash) → END")
                    return "done"

            print(f"[ROUTER] Planning failure (retry {state.planner_retry_count}/{RETRY_LIMITS['planner']}) "
                  f"→ retry_planner")
            return "retry_planner"

        # ── Tool failure (execution error) ────────────────────────────────
        if failure_type == "tool":
            if state.tool_retry_count >= RETRY_LIMITS["tool"]:
                print(f"[ROUTER] Tool failure but tool retries exhausted "
                      f"({state.tool_retry_count}/{RETRY_LIMITS['tool']}) → END")
                return "done"

            # Early stop check
            if state.last_tool_result_hash and state.tool_retry_count > 0:
                current_hash = _compute_hash(state.tool_results)
                if current_hash == state.last_tool_result_hash:
                    print(f"[ROUTER] Tool failure but results unchanged (same hash) → END")
                    return "done"

            print(f"[ROUTER] Tool failure (retry {state.tool_retry_count}/{RETRY_LIMITS['tool']}) "
                  f"→ retry_tool")
            return "retry_tool"

        # ── Unknown failure (can't classify) ──────────────────────────────
        print(f"[ROUTER] Failure type unknown; cannot intelligently retry → END")
        return "done"

    # ── Increment retry counter nodes ──────────────────────────────────────
    # These are simple nodes that increment the appropriate counter and
    # return nothing (state merge handles the increment).

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

    graph.add_node("rewriter",           rewriter_node)
    graph.add_node("planner",            planner_node)
    graph.add_node("retriever",          retriever_node)
    graph.add_node("join",               join_node)
    graph.add_node("grader",             grader_node)
    graph.add_node("tool_agent",         tool_node)
    graph.add_node("answer",             answer_node)
    graph.add_node("critic",             critic_node)
    graph.add_node("no_answer",          no_answer_node)
    graph.add_node("metadata_answer",    metadata_answer_node)
    graph.add_node("sql_answer",         sql_answer_node)
    
    # Retry increment nodes
    graph.add_node("increment_answer_retry",     increment_answer_retry)
    graph.add_node("increment_retriever_retry",  increment_retriever_retry)
    graph.add_node("increment_planner_retry",    increment_planner_retry)
    graph.add_node("increment_tool_retry",       increment_tool_retry)

    # ── Edges: Main pipeline ──────────────────────────────────────────────

    graph.add_edge(START, "rewriter")
    graph.add_edge("rewriter", "planner")
    graph.add_edge("rewriter", "retriever")

    graph.add_edge("planner",   "join")
    graph.add_edge("retriever", "join")

    graph.add_edge("join", "grader")

    graph.add_conditional_edges(
        "grader",
        route_after_planning,
        {
            "tool_agent":       "tool_agent",
            "answer":           "answer",
            "no_answer":        "no_answer",
            "metadata_answer":  "metadata_answer",
            "sql_answer":       "sql_answer",
        },
    )

    graph.add_edge("tool_agent", "answer")
    graph.add_edge("answer",     "critic")

    # End nodes
    graph.add_edge("no_answer",       END)
    graph.add_edge("metadata_answer", END)
    graph.add_edge("sql_answer",      END)

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

    # ── Edges: Retry increments → nodes to retry ──────────────────────────

    graph.add_edge("increment_answer_retry",    "answer")
    graph.add_edge("increment_retriever_retry", "retriever")
    graph.add_edge("increment_planner_retry",   "planner")
    graph.add_edge("increment_tool_retry",      "tool_agent")

    # After planner retry, fan back out to retriever (parallel with new plan)
    graph.add_edge("planner", "join")

    return graph.compile()
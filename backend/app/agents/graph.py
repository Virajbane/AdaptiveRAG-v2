"""
LangGraph orchestration for the RAG agent pipeline — Option B.

Flow:

    START → rewriter ──┬──→ planner ───┐
                        └──→ retriever ─┴──→ join → grader ──(route_after_planning)──→ tool_agent ──→ answer ──→ critic
                                                                  └──────────────────────────────────────→ answer ──↑    │
                                                                                                                          │
                                                                     route_after_critic:                                │
                                                                       "done"  ──→ END                                  │
                                                                       "retry" ──→ answer (only!) ──────────────────────┘

Option B improvements over the original:
─────────────────────────────────────────
1. Smart routing after join
   - No web keywords in question  →  skip tool_agent, go straight to answer
   - Web keywords detected        →  tool_agent first
   - Error in planner/retriever   →  skip tool_agent, let answer surface the error

2. Surgical retry — answer-only
   - On retry, only answer_node reruns (retrieval was fine, answer quality was low)
   - Full pipeline re-run (planner + retriever) only on the very first pass
   - Saves ~7 minutes per retry on CPU-bound hardware

3. Critic confidence wired into confidence_final
   - confidence_final = 0.7 * critic_confidence + 0.3 * rerank_score
   - CriticAgent sets both critic_confidence and confidence_final directly

4. Per-node timing
   - Every node prints its own wall-clock duration via _timed() wrapper
   - Useful for identifying bottlenecks, especially on CPU-bound hardware

5. Query rewriting + document-scoped retrieval
   - rewriter runs first: resolves conversational context + fixes typos
   - retriever uses RewriterAgent's output, and resolves a document_id
     filter (via document_resolver) when the question names a specific
     uploaded file, so search doesn't pool chunks across all documents

6. Pre-retrieval relevance grading
   - grader runs right after join, before routing — drops chunks whose
     rerank/RRF score is far below the best match for this query, so
     AnswerAgent never sees near-irrelevant chunks. No extra LLM call;
     reuses scores already computed during retrieval. See grader.py.

Design notes:
─────────────
- join is a no-op sync node. LangGraph waits for ALL predecessors before
  running it, giving us the planner+retriever fan-in we need.
- grader runs unconditionally after join (regardless of which path
  route_after_planning picks next), since both tool_agent→answer and
  the direct→answer path end up at AnswerAgent, which reads
  state.retrieved_docs either way.
- Retry is capped at MAX_RETRIES. critic_node increments retry_count.
- planner_node and retriever_node return PARTIAL dicts (only the keys they
  own) to avoid InvalidUpdateError when both write in the same superstep.
- error field uses an Annotated reducer (_merge_error) in AgentState so
  two parallel error writes in the same step merge instead of crashing.
- retriever needs a Mongo db handle (for document_resolver's filename
  lookup) so build_agent_graph() now takes an optional db param, passed
  down from AgentOrchestrator._ensure_graph().
"""

import time
from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.planner import PlannerAgent
from app.agents.retriever import RetrieverAgent
from app.agents.grader import GraderAgent
from app.agents.tool_agent import ToolAgent
from app.agents.critic import CriticAgent
from app.agents.answer import AnswerAgent
from app.agents.rewriter import RewriterAgent
from app.services.llm.provider import LLMProvider
from app.config.settings import settings

MAX_RETRIES = 2


def build_agent_graph(db=None):
    """
    Construct and compile the agent StateGraph.

    Returns a compiled graph with an `.ainvoke(state)` method that runs
    the full pipeline and returns the final AgentState.
    """

    llm      = LLMProvider()                                      # qwen2.5:7b  — deep reasoning
    fast_llm = LLMProvider(model=settings.OLLAMA_FAST_MODEL)      # qwen2.5:1.5b — routing / judging

    rewriter   = RewriterAgent(fast_llm)
    planner    = PlannerAgent(fast_llm)
    retriever  = RetrieverAgent(fast_llm, db=db)   # db needed for document_resolver
    grader     = GraderAgent(fast_llm)             # no LLM call made, but BaseAgent needs an llm arg
    tool_agent = ToolAgent(fast_llm)
    critic     = CriticAgent(fast_llm)
    answer     = AnswerAgent(fast_llm)

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

    async def planner_node(state: AgentState) -> dict:
        t0 = time.perf_counter()
        result = await planner.run(state.copy())
        _print_timing("planner", time.perf_counter() - t0)
        update = {
            "plan":           result.plan,
            "sources_needed": result.sources_needed,
            "confidence":     result.confidence,
        }
        if result.error:
            update["error"] = result.error
        return update

    async def retriever_node(state: AgentState) -> dict:
        t0 = time.perf_counter()
        result = await retriever.run(state.copy())
        _print_timing("retriever", time.perf_counter() - t0)
        update = {
            "retrieved_docs": result.retrieved_docs,
            "web_results":    result.web_results,
            "search_time_ms": result.search_time_ms,
        }
        if result.error:
            update["error"] = result.error
        return update

    async def join_node(state: AgentState) -> dict:
        return {}

    async def grader_node(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        result = await grader.run(state.copy())
        _print_timing("grader", time.perf_counter() - t0)
        return result

    async def tool_node(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        result = await tool_agent.run(state.copy())
        _print_timing("tool_agent", time.perf_counter() - t0)
        return result

    async def answer_node(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        result = await answer.run(state.copy())
        _print_timing("answer", time.perf_counter() - t0)
        return result

    async def critic_node(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        result = await critic.run(state.copy())
        _print_timing("critic", time.perf_counter() - t0)
        if not result.is_valid:
            result.retry_count = state.retry_count + 1
        return result

    # ── Routing functions ─────────────────────────────────────────────────

    def route_after_planning(state: AgentState) -> str:
        if state.error:
            print(f"[ROUTER] Error detected, skipping tool_agent: {state.error}")
            return "answer"

        # Trust the Planner's own decision first. PlannerAgent already ran
        # an LLM call to decide sources_needed (see PLANNER_PROMPT) - this
        # router used to re-derive routing from a hardcoded keyword list
        # instead, silently discarding the Planner's actual output.
        if "web" in state.sources_needed or "tools" in state.sources_needed:
            print(f"[ROUTER] Planner requested web/tools → tool_agent "
                f"(sources_needed={state.sources_needed})")
            return "tool_agent"

        if state.sources_needed:
            print(f"[ROUTER] Planner sources_needed={state.sources_needed} "
                f"→ answer (documents only)")
            return "answer"

        # Fallback ONLY if the Planner produced no sources at all (e.g. its
        # JSON parse failed before sources_needed could be set to anything -
        # PlannerAgent defaults to ["documents"] on parse failure, so reaching
        # here should be rare).
        web_keywords = [
            "news", "latest", "current", "today", "price", "stock",
            "weather", "live", "trending", "recent", "2024", "2025", "2026"
        ]
        question_lower = state.question.lower()
        needs_web = any(kw in question_lower for kw in web_keywords)

        if needs_web:
            print(f"[ROUTER] Fallback: no sources_needed, web keywords detected → tool_agent")
            return "tool_agent"

        print(f"[ROUTER] Fallback: no sources_needed, no web keywords → answer")
        return "answer"

    def route_after_critic(state: AgentState) -> str:
        if state.is_valid:
            print(f"[ROUTER] Critic accepted answer. confidence_final={state.confidence_final:.4f}")
            return "done"

        if state.retry_count >= MAX_RETRIES:
            print(f"[ROUTER] Max retries ({MAX_RETRIES}) reached. Returning best answer.")
            return "done"

        print(f"[ROUTER] Critic rejected (retry {state.retry_count}/{MAX_RETRIES}). Retrying answer only.")
        return "retry_answer"

    # ── Build graph ───────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("rewriter",   rewriter_node)
    graph.add_node("planner",    planner_node)
    graph.add_node("retriever",  retriever_node)
    graph.add_node("join",       join_node)
    graph.add_node("grader",     grader_node)
    graph.add_node("tool_agent", tool_node)
    graph.add_node("answer",     answer_node)
    graph.add_node("critic",     critic_node)

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
            "tool_agent": "tool_agent",
            "answer":     "answer",
        },
    )

    graph.add_edge("tool_agent", "answer")
    graph.add_edge("answer",     "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "done":         END,
            "retry_answer": "answer",
        },
    )

    return graph.compile()
"""
LangGraph orchestration for the RAG agent pipeline — Option B.

Flow:

    START → rewriter ──┬──→ planner ───┐
                        └──→ retriever ─┴──→ join → grader ──(route_after_planning)──→ tool_agent ──→ answer ──→ critic
                                                                  ├──────────────────────────────────────→ answer ──↑    │
                                                                  ├──→ no_answer ──→ END                                │
                                                                  └──→ metadata_answer ──→ END                          │
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

7. 2026-07-04: Answer generation uses the DEEP-REASONING model, not fast_llm
   - Eval against a real arXiv paper showed AnswerAgent (previously wired
     to fast_llm, i.e. qwen2.5:0.5b) fabricating facts wholesale instead
     of grounding in retrieved context: a nonexistent framework name
     ("ConversaSynth"), a nonexistent ASR tool name ("Xuetongyan (TTS)"),
     invented BERT hyperparameters that don't appear anywhere in the
     source document, and even leaking the ANSWER_PROMPT's own internal
     formatting instruction ("Rule: If the question asks for ONE fact...")
     into the visible answer text. A 0.5B model does not reliably
     distinguish "stay grounded in this context" from "free-associate to
     something topically similar from pretraining" — that's a capability
     gap, not a prompt-wording problem, and no amount of prompt tuning
     fixes it. fast_llm remains appropriate for rewriter/planner/critic/
     tool_agent, since routing and judging are comparatively low-stakes
     and only need to be right often enough to trigger the deterministic
     backstops already in planner.py — but the agent actually producing
     the user-facing answer needs the strongest available model.

8. 2026-07-04: Grader hard-stop on absolute-floor rejection
   - Previously, a batch of chunks all scoring below GraderAgent's
     ABSOLUTE_FLOOR still went through AnswerAgent + CriticAgent (~15-20s
     of wasted LLM calls) before eventually surfacing a low-confidence or
     hallucinated answer. Now GraderAgent sets state.retrieval_rejected,
     and route_after_planning checks it FIRST — routing straight to
     no_answer_node, which returns an honest "not found" response with
     zero additional LLM calls and zero hallucination risk.

9. 2026-07-04: Metadata short-circuit for title/author questions
   - Title/author questions have near-zero lexical/semantic overlap with
     the metadata text itself (the title never contains the word
     "title") — no amount of embedding tuning reliably surfaces it via
     retrieval. PlannerAgent now detects these questions and, if metadata
     was extracted at ingestion (see MetadataExtractor), sets
     sources_needed=["metadata"] and populates state.metadata_answer.
     route_after_planning checks this before anything else and routes to
     metadata_answer_node, which answers directly from stored metadata —
     skipping tool_agent/answer/critic entirely (grader still runs
     harmlessly on the parallel-fetched retrieval, since retriever runs
     concurrently with planner via the rewriter fan-out and can't be
     skipped without restructuring that fan-out — a future optimization,
     not a correctness issue, since the unused retrieved_docs are simply
     ignored on this path).

Design notes:
─────────────
- join is a no-op sync node. LangGraph waits for ALL predecessors before
  running it, giving us the planner+retriever fan-in we need.
- grader runs unconditionally after join (regardless of which path
  route_after_planning picks next), since both tool_agent→answer and
  the direct→answer path end up at AnswerAgent, which reads
  state.retrieved_docs either way. no_answer and metadata_answer paths
  simply ignore grader's output.
- Retry is capped at MAX_RETRIES. critic_node increments retry_count.
- planner_node and retriever_node return PARTIAL dicts (only the keys they
  own) to avoid InvalidUpdateError when both write in the same superstep.
- error field uses an Annotated reducer (_merge_error) in AgentState so
  two parallel error writes in the same step merge instead of crashing.
- retriever needs a Mongo db handle (for document_resolver's filename
  lookup) so build_agent_graph() now takes an optional db param, passed
  down from AgentOrchestrator._ensure_graph(). planner now also needs db
  (for metadata lookups), same pattern.
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
from langsmith import traceable

MAX_RETRIES = 2


def build_agent_graph(db=None):
    """
    Construct and compile the agent StateGraph.

    Returns a compiled graph with an `.ainvoke(state)` method that runs
    the full pipeline and returns the final AgentState.
    """

    llm      = LLMProvider(num_ctx=4096)                                      # qwen2.5:7b  — deep reasoning
    fast_llm = LLMProvider(model=settings.OLLAMA_FAST_MODEL)      # qwen2.5:1.5b — routing / judging

    rewriter   = RewriterAgent(fast_llm)
    planner    = PlannerAgent(fast_llm, db=db)     # db needed for metadata lookup (2026-07-04)
    retriever  = RetrieverAgent(fast_llm, db=db)   # db needed for document_resolver
    grader     = GraderAgent(fast_llm)             # no LLM call made, but BaseAgent needs an llm arg
    tool_agent = ToolAgent(fast_llm)
    critic     = CriticAgent(fast_llm)
    answer     = AnswerAgent(llm)   # 2026-07-04: deep-reasoning model — see note above

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
        }
        if result.error:
            update["error"] = result.error
        return update

    @traceable(name="retriever_node", run_type="retriever")
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

    @traceable(name="grader_node", run_type="chain")
    async def grader_node(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        result = await grader.run(state.copy())
        _print_timing("grader", time.perf_counter() - t0)

        # ---- High-confidence retrieval override ----
        # See route_after_planning's old comment for full rationale. This
        # MUST live in a node (not the route_after_planning conditional-edge
        # function) because conditional-edge functions only return a string;
        # any state mutation inside them is local and never gets committed
        # back into the graph's real state. grader_node returns a full
        # AgentState, which DOES get merged, so the override has to happen
        # here to actually reach AnswerAgent downstream.
        if "documents" not in result.sources_needed and result.retrieved_docs:
            top_score = result.retrieved_docs[0].get("rerank_score", 0.0)
            if top_score >= HIGH_CONFIDENCE_RETRIEVAL_THRESHOLD:
                print(
                    f"[GRADER] Override: planner said sources_needed={result.sources_needed} "
                    f"but found a high-confidence document match "
                    f"(top_score={top_score:.4f} >= {HIGH_CONFIDENCE_RETRIEVAL_THRESHOLD}) "
                    f"— adding 'documents' to sources"
                )
                result.sources_needed = result.sources_needed + ["documents"]

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

    async def no_answer_node(state: AgentState) -> dict:
        print("[ROUTER] Retrieval rejected (below absolute floor) — "
              "returning direct not-found response, skipping answer/critic")
        return {
            "answer": "I couldn't find this information in the document.",
            "confidence_final": 0.0,
            "sources": [],
            "is_valid": True,   # nothing to retry — this IS the final answer
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
            "confidence_final": 0.95,   # extracted directly from source text, not inferred
            "sources": [],
            "is_valid": True,
        }

        # ── Routing functions ─────────────────────────────────────────────────

        # High-confidence retrieval override threshold. If the Planner said
    # "web" (or omitted "documents" entirely) but the Grader independently
    # found a document chunk this strong, trust the retrieval signal over
    # the Planner's text-classification guess. This is the mirror image of
    # the Grader's existing ABSOLUTE_FLOOR hard-stop (skip generation when
    # evidence is too weak to trust) — this handles the opposite failure:
    # don't discard generation-worthy evidence just because a small LLM's
    # routing guess didn't ask for it.
    #
    # 2026-07-04: added after eval showed "Which model generates the
    # synthetic dialogue text?" scoring 0.957/0.15 on real document chunks
    # (including the correct GPT-2 answer) but Planner classified it as
    # sources=["web"] with no personal/doc-referential wording for the
    # existing regex backstops to catch. AnswerAgent then received "0 top
    # documents + 5 web results" and answered from an unrelated but
    # plausible-sounding web result (a different tool with a similar name)
    # instead of the paper's actual, correctly-retrieved answer.
    HIGH_CONFIDENCE_RETRIEVAL_THRESHOLD = 0.5


    def route_after_planning(state: AgentState) -> str:
        if state.sources_needed == ["metadata"]:
            print("[ROUTER] Metadata answer available → metadata_answer (skipping retrieval path)")
            return "metadata_answer"

        if state.retrieval_rejected:
            print("[ROUTER] Retrieval rejected by grader → no_answer (skipping tool_agent/answer/critic)")
            return "no_answer"

        if state.error:
            print(f"[ROUTER] Error detected, skipping tool_agent: {state.error}")
            return "answer"

        

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

    graph.add_node("rewriter",         rewriter_node)
    graph.add_node("planner",          planner_node)
    graph.add_node("retriever",        retriever_node)
    graph.add_node("join",             join_node)
    graph.add_node("grader",           grader_node)
    graph.add_node("tool_agent",       tool_node)
    graph.add_node("answer",           answer_node)
    graph.add_node("critic",           critic_node)
    graph.add_node("no_answer",        no_answer_node)
    graph.add_node("metadata_answer",  metadata_answer_node)

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
        },
    )

    graph.add_edge("tool_agent", "answer")
    graph.add_edge("answer",     "critic")
    graph.add_edge("no_answer",       END)
    graph.add_edge("metadata_answer", END)

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "done":         END,
            "retry_answer": "answer",
        },
    )

    return graph.compile()
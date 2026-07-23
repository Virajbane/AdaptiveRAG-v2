"""
Unit tests for the agent system (Phase 5 / Phase 10 testing).

Agents tested:
  - AgentState          (dataclass correctness)
  - BaseAgent           (run() error-handling wrapper, parse_json_response)
  - PlannerAgent        (_execute: happy path, JSON parse failure)
  - RetrieverAgent      (_execute: sources_needed gate, search called, error path)
  - AnswerAgent         (_execute: no-docs fast path, normal path, confidence formula)
  - CriticAgent         (_execute: happy path, JSON parse failure)
  - ToolAgent           (_execute: sources_needed gate, execute_tool called, error path)
  - AgentOrchestrator   (process: full happy path, planner error short-circuit,
                         retriever error short-circuit, memory save called,
                         memory_manager=None safe)

Mock strategy
─────────────
All external I/O is patched at the module where it is *used*, not where
it is defined — same convention as the retrieval test suite.

  LLMProvider.generate  → patched on the *instance* via BaseAgent.__init__
  HybridSearchEngine    → patched at app.agents.retriever.HybridSearchEngine
  tool_registry         → patched at app.services.tools.registry.tool_registry
  memory_manager        → patched at app.services.memory.manager.memory_manager

No real Ollama / Qdrant / MongoDB calls are made.

Known gaps (documented, not hidden)
────────────────────────────────────
1. CriticAgent is instantiated in AgentOrchestrator.__init__ but never
   called in process().  Critic tests cover the agent in isolation only.
   Gap: orchestrator integration with Critic is unimplemented.

2. ToolAgent is skipped inside process() (TODO comment in source).
   Tool tests cover the agent in isolation only.

3. CriticAgent._execute() still fires an LLM call even when
   retrieved_docs is empty (context becomes ""), wasting a round-trip.
   No fix applied; documented here.

4. RetrieverAgent constructs HybridSearchEngine() inside _execute() on
   every call (no dependency injection).  Tests patch the class so each
   instantiation returns the same mock.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.state import AgentState
from app.agents.base import BaseAgent
from app.agents.planner import PlannerAgent
from app.agents.retriever import RetrieverAgent
from app.agents.answer import AnswerAgent
from app.agents.critic import CriticAgent
from app.agents.tool_agent import ToolAgent
from app.agents.orchestrator import AgentOrchestrator
from contextlib import ExitStack


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_state(**kwargs) -> AgentState:
    """Return an AgentState with sensible defaults, overridable via kwargs."""
    defaults = dict(question="What is RAG?", user_id="user-1")
    defaults.update(kwargs)
    return AgentState(**defaults)


def make_doc(text="Some relevant text.", score=0.8, doc_id="doc-1", chunk_index=0):
    return {
        "text": text,
        "combined_score": score,
        "doc_id": doc_id,
        "chunk_index": chunk_index,
    }


def mock_llm(return_value: str = "") -> MagicMock:
    """Return a MagicMock LLMProvider whose generate() is an AsyncMock."""
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=return_value)
    return llm


# ─────────────────────────────────────────────
# AgentState
# ─────────────────────────────────────────────

class TestAgentState:
    def test_required_fields(self):
        s = AgentState(question="q", user_id="u")
        assert s.question == "q"
        assert s.user_id == "u"

    def test_defaults(self):
        s = AgentState(question="q", user_id="u")
        assert s.plan == ""
        assert s.sources_needed == []
        assert s.confidence == 0.0
        assert s.retrieved_docs == []
        assert s.web_results == []
        assert s.tool_results == {}
        assert s.is_valid is False
        assert s.validation_issues == []
        assert s.answer == ""
        assert s.sources == []
        assert s.confidence_final == 0.0
        assert s.search_time_ms == 0.0
        assert s.error == ""

    def test_mutable_defaults_are_independent(self):
        """Each AgentState instance must have its own list/dict instances."""
        s1 = AgentState(question="q", user_id="u")
        s2 = AgentState(question="q", user_id="u")
        s1.sources_needed.append("web")
        assert s2.sources_needed == [], (
            "Mutable default shared between instances — field_factory missing"
        )


# ─────────────────────────────────────────────
# BaseAgent — error wrapper + parse_json_response
# ─────────────────────────────────────────────

class ConcreteAgent(BaseAgent):
    """Minimal concrete subclass for testing BaseAgent behaviour."""

    def __init__(self, llm, *, raise_error=False, error_msg="boom"):
        super().__init__(llm)
        self._raise_error = raise_error
        self._error_msg = error_msg

    async def _execute(self, state: AgentState) -> AgentState:
        if self._raise_error:
            raise ValueError(self._error_msg)
        state.plan = "executed"
        return state


class TestBaseAgentRun:
    @pytest.mark.asyncio
    async def test_run_returns_updated_state_on_success(self):
        agent = ConcreteAgent(mock_llm())
        state = make_state()
        result = await agent.run(state)
        assert result.plan == "executed"
        assert result.error == ""

    @pytest.mark.asyncio
    async def test_run_catches_exception_and_sets_error(self):
        agent = ConcreteAgent(mock_llm(), raise_error=True, error_msg="oops")
        state = make_state()
        result = await agent.run(state)
        assert "ConcreteAgent error: oops" in result.error

    @pytest.mark.asyncio
    async def test_run_does_not_reraise(self):
        """run() must never propagate — orchestrator relies on error field."""
        agent = ConcreteAgent(mock_llm(), raise_error=True)
        state = make_state()
        # Should not raise
        await agent.run(state)


class TestParseJsonResponse:
    def setup_method(self):
        self.agent = ConcreteAgent(mock_llm())

    def test_plain_json(self):
        assert self.agent.parse_json_response('{"a": 1}') == {"a": 1}

    def test_markdown_fenced_json(self):
        raw = "```json\n{\"a\": 1}\n```"
        assert self.agent.parse_json_response(raw) == {"a": 1}

    def test_markdown_fenced_no_language_tag(self):
        raw = "```\n{\"a\": 1}\n```"
        assert self.agent.parse_json_response(raw) == {"a": 1}

    def test_json_embedded_in_prose(self):
        raw = 'Here is the plan: {"a": 1} — hope that helps.'
        assert self.agent.parse_json_response(raw) == {"a": 1}

    def test_raises_on_unparseable(self):
        with pytest.raises(json.JSONDecodeError):
            self.agent.parse_json_response("not json at all")

    def test_raises_on_empty_string(self):
        with pytest.raises(json.JSONDecodeError):
            self.agent.parse_json_response("")


# ─────────────────────────────────────────────
# PlannerAgent
# ─────────────────────────────────────────────

VALID_PLAN_JSON = json.dumps({
    "intent": "lookup",
    "sources": ["documents", "web"],
    "confidence": 0.9,
    "strategy": "Search docs first",
})


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_happy_path_populates_state(self):
        llm = mock_llm(VALID_PLAN_JSON)
        agent = PlannerAgent(llm)
        state = make_state()
        result = await agent._execute(state)

        assert result.plan == "Search docs first"
        assert result.sources_needed == ["documents", "web"]
        assert result.confidence == 0.9
        

    @pytest.mark.asyncio
    async def test_llm_called_with_question_in_prompt(self):
        llm = mock_llm(VALID_PLAN_JSON)
        agent = PlannerAgent(llm)
        state = make_state(question="Tell me about embeddings")
        await agent._execute(state)

        prompt_used = llm.generate.call_args[0][0]
        assert "Tell me about embeddings" in prompt_used

    @pytest.mark.asyncio
    async def test_json_parse_failure_sets_error(self):
        llm = mock_llm("definitely not json")
        agent = PlannerAgent(llm)
        state = make_state()
        result = await agent._execute(state)

        assert result.error == "Failed to parse planner response as JSON"

    @pytest.mark.asyncio
    async def test_json_parse_failure_sets_default_sources(self):
        """Even on parse failure, sources_needed gets a safe default."""
        llm = mock_llm("not json")
        agent = PlannerAgent(llm)
        state = make_state()
        result = await agent._execute(state)

        assert result.sources_needed == ["documents"]

    @pytest.mark.asyncio
    async def test_missing_json_keys_use_defaults(self):
        """parse_json_response succeeds but keys are absent — .get() defaults kick in."""
        llm = mock_llm('{"intent": "lookup"}')
        agent = PlannerAgent(llm)
        state = make_state()
        result = await agent._execute(state)

        assert result.plan == ""
        assert result.sources_needed == ["documents"]
        assert result.confidence == 0.5


# ─────────────────────────────────────────────
# RetrieverAgent
# ─────────────────────────────────────────────

class TestRetrieverAgent:
    @pytest.mark.asyncio
    async def test_always_searches_regardless_of_sources_needed(self):
        """
        2026-07-xx architecture change: RetrieverAgent runs in PARALLEL
        with PlannerAgent (see class docstring), so it always searches —
        the orchestrator/router decides whether to USE retrieved_docs
        based on sources_needed AFTER both planner and retriever finish.
        This replaces the old 'skips search when documents not in
        sources' test, which asserted behavior that no longer exists.
        """
        llm = mock_llm()
        agent = RetrieverAgent(llm)
        state = make_state()
        state.sources_needed = ["web"]  # no "documents" — must NOT matter

        mock_engine = MagicMock()
        mock_engine.search = AsyncMock(return_value=[])

        with patch("app.agents.retriever.HybridSearchEngine", return_value=mock_engine), \
             patch("app.agents.retriever.resolve_document_filter", AsyncMock(return_value=None)):
            result = await agent._execute(state)

        mock_engine.search.assert_called_once()
        assert result.error == ""

    @pytest.mark.asyncio
    async def test_calls_hybrid_search_when_documents_needed(self):
        llm = mock_llm()
        agent = RetrieverAgent(llm)
        state = make_state()
        state.sources_needed = ["documents"]

        docs = [make_doc("doc text", 0.75)]

        mock_engine = MagicMock()
        mock_engine.search = AsyncMock(return_value=docs)

        with patch("app.agents.retriever.HybridSearchEngine", return_value=mock_engine), \
             patch("app.agents.retriever.resolve_document_filter", AsyncMock(return_value=None)):
            result = await agent._execute(state)

        assert result.retrieved_docs == docs

    @pytest.mark.asyncio
    async def test_search_called_with_correct_args(self):
        """Ordinary (non-metric-style) question -> DEFAULT_CANDIDATE_K=5,
        document_id resolved to None (no specific file named)."""
        llm = mock_llm()
        agent = RetrieverAgent(llm)
        state = make_state(question="What is FAISS?", user_id="u-42")
        state.sources_needed = ["documents"]

        mock_engine = MagicMock()
        mock_engine.search = AsyncMock(return_value=[])

        with patch("app.agents.retriever.HybridSearchEngine", return_value=mock_engine), \
             patch("app.agents.retriever.resolve_document_filter", AsyncMock(return_value=None)):
            await agent._execute(state)

        mock_engine.search.assert_called_once_with(
            query="What is FAISS?",
            user_id="u-42",
            top_k=5,
            document_id=None,
        )

    @pytest.mark.asyncio
    async def test_metric_style_question_widens_candidate_pool(self):
        """2026-07-14 fix: questions asking for a specific numeric metric
        (accuracy/score/rate/Table N/etc) search a wider candidate pool
        (METRIC_CANDIDATE_K=12) but only forward FINAL_CONTEXT_SIZE=5 to
        the LLM."""
        llm = mock_llm()
        agent = RetrieverAgent(llm)
        state = make_state(question="What is the accuracy in Table 3?", user_id="u-1")
        state.sources_needed = ["documents"]

        # 12 candidates, all plain prose chunks (no table-row format),
        # scores descending so we can assert the narrowing keeps the top 5.
        candidates = [
            make_doc(text=f"chunk {i}", score=0.0, doc_id=f"doc-{i}")
            for i in range(12)
        ]
        for i, c in enumerate(candidates):
            c["rerank_score"] = 1.0 - (i * 0.05)  # descending: 1.0, 0.95, ...

        mock_engine = MagicMock()
        mock_engine.search = AsyncMock(return_value=candidates)

        with patch("app.agents.retriever.HybridSearchEngine", return_value=mock_engine), \
             patch("app.agents.retriever.resolve_document_filter", AsyncMock(return_value=None)):
            result = await agent._execute(state)

        mock_engine.search.assert_called_once_with(
            query="What is the accuracy in Table 3?",
            user_id="u-1",
            top_k=12,
            document_id=None,
        )
        assert len(result.retrieved_docs) == 5
        assert [d["doc_id"] for d in result.retrieved_docs] == \
            ["doc-0", "doc-1", "doc-2", "doc-3", "doc-4"]

    @pytest.mark.asyncio
    async def test_document_id_filter_passed_through_when_resolved(self):
        """When resolve_document_filter confidently names a specific
        uploaded document, that document_id must be forwarded to
        search() to scope retrieval to just that file."""
        llm = mock_llm()
        agent = RetrieverAgent(llm)
        state = make_state(question="What does report.pdf say about X?")
        state.sources_needed = ["documents"]

        mock_engine = MagicMock()
        mock_engine.search = AsyncMock(return_value=[])

        with patch("app.agents.retriever.HybridSearchEngine", return_value=mock_engine), \
             patch("app.agents.retriever.resolve_document_filter",
                   AsyncMock(return_value="doc-abc123")):
            await agent._execute(state)

        mock_engine.search.assert_called_once_with(
            query="What does report.pdf say about X?",
            user_id="user-1",
            top_k=5,
            document_id="doc-abc123",
        )

    @pytest.mark.asyncio
    async def test_search_time_ms_is_set(self):
        llm = mock_llm()
        agent = RetrieverAgent(llm)
        state = make_state()
        state.sources_needed = ["documents"]

        mock_engine = MagicMock()
        mock_engine.search = AsyncMock(return_value=[make_doc()])

        with patch("app.agents.retriever.HybridSearchEngine", return_value=mock_engine), \
             patch("app.agents.retriever.resolve_document_filter", AsyncMock(return_value=None)):
            result = await agent._execute(state)

        assert result.search_time_ms >= 0

    @pytest.mark.asyncio
    async def test_search_exception_sets_error(self):
        llm = mock_llm()
        agent = RetrieverAgent(llm)
        state = make_state()
        state.sources_needed = ["documents"]

        mock_engine = MagicMock()
        mock_engine.search = AsyncMock(side_effect=RuntimeError("qdrant down"))

        with patch("app.agents.retriever.HybridSearchEngine", return_value=mock_engine), \
             patch("app.agents.retriever.resolve_document_filter", AsyncMock(return_value=None)):
            result = await agent._execute(state)

        assert "Retriever error" in result.error
        assert "qdrant down" in result.error

# ─────────────────────────────────────────────
# AnswerAgent
# ─────────────────────────────────────────────

class TestAnswerAgent:
    @pytest.mark.asyncio
    async def test_no_docs_returns_upload_message(self):
        llm = mock_llm("This answer should never appear")
        agent = AnswerAgent(llm)
        state = make_state()
        state.retrieved_docs = []
        state.confidence = 0.8
        # no sources_needed set — irrelevant here since retrieved_docs is
        # empty either way, so this test is unaffected by the gating change

        result = await agent._execute(state)

        assert "upload" in result.answer.lower()
        assert result.sources == []
        assert result.confidence_final == pytest.approx(0.8 * 0.3, abs=1e-9)
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_happy_path_sets_answer_and_sources(self):
        llm = mock_llm("RAG stands for Retrieval-Augmented Generation.")
        agent = AnswerAgent(llm)
        state = make_state()
        state.sources_needed = ["documents"]  # 2026-07-03: doc context now gated on this
        state.retrieved_docs = [make_doc("RAG context", 0.9)]
        state.confidence = 0.8

        result = await agent._execute(state)

        assert result.answer == "RAG stands for Retrieval-Augmented Generation."
        assert len(result.sources) == 1
        assert result.sources[0]["doc_id"] == "doc-1"
        assert result.error == ""

    @pytest.mark.asyncio
    async def test_sources_come_from_retrieved_docs_not_llm(self):
        """FIX #1 guard: sources must reflect retrieved_docs, never LLM output."""
        docs = [make_doc("text A", 0.9, "doc-A"), make_doc("text B", 0.7, "doc-B")]
        llm = mock_llm("Some answer")
        agent = AnswerAgent(llm)
        state = make_state()
        state.sources_needed = ["documents"]
        state.retrieved_docs = docs
        state.confidence = 0.5

        result = await agent._execute(state)

        source_ids = [s["doc_id"] for s in result.sources]
        assert source_ids == ["doc-A", "doc-B"]

    @pytest.mark.asyncio
    async def test_confidence_formula(self):
        """FIX #2 guard: confidence_final = planner*0.5 + avg_doc_score*0.5."""
        docs = [make_doc(score=0.6), make_doc(score=0.8)]
        avg = (0.6 + 0.8) / 2  # 0.7
        planner_conf = 0.4
        expected = planner_conf * 0.5 + avg * 0.5  # 0.55

        llm = mock_llm("answer")
        agent = AnswerAgent(llm)
        state = make_state()
        state.sources_needed = ["documents"]
        state.retrieved_docs = docs
        state.confidence = planner_conf

        result = await agent._execute(state)

        assert result.confidence_final == pytest.approx(expected, abs=1e-9)

    @pytest.mark.asyncio
    async def test_confidence_capped_at_1(self):
        docs = [make_doc(score=1.0)]
        llm = mock_llm("answer")
        agent = AnswerAgent(llm)
        state = make_state()
        state.sources_needed = ["documents"]
        state.retrieved_docs = docs
        state.confidence = 1.0

        result = await agent._execute(state)

        assert result.confidence_final <= 1.0

    @pytest.mark.asyncio
    async def test_source_text_truncated_to_200_chars(self):
        long_text = "x" * 500
        docs = [make_doc(text=long_text)]
        llm = mock_llm("answer")
        agent = AnswerAgent(llm)
        state = make_state()
        state.sources_needed = ["documents"]
        state.retrieved_docs = docs
        state.confidence = 0.5

        result = await agent._execute(state)

        assert len(result.sources[0]["text"]) <= 200

    @pytest.mark.asyncio
    async def test_llm_exception_sets_error_and_empty_sources(self):
        llm = mock_llm()
        llm.generate = AsyncMock(side_effect=RuntimeError("ollama down"))
        agent = AnswerAgent(llm)
        state = make_state()
        state.sources_needed = ["documents"]
        state.retrieved_docs = [make_doc()]
        state.confidence = 0.5

        result = await agent._execute(state)

        assert "Answer agent error" in result.error
        assert result.sources == []
        assert result.confidence_final == 0.0
# ─────────────────────────────────────────────
# CriticAgent
# ─────────────────────────────────────────────

VALID_CRITICISM_JSON = json.dumps({
    "valid": True,
    "confidence": 90,
    "issues": [],
    "needs_more_info": False,
})

CRITICISM_WITH_ISSUES = json.dumps({
    "valid": False,
    "confidence": 40,
    "issues": ["Claim X is not supported by context"],
    "needs_more_info": True,
})


class TestCriticAgent:
    @pytest.mark.asyncio
    async def test_happy_path_sets_is_valid_true(self):
        """
        Context must actually contain the answer's checkable facts, or
        the 2026-07-10 overconfident_acceptance grounding backstop will
        (correctly) override the judge's approval — that's the intended
        behavior, not a bug. Using "RAG" as the checkable proper noun,
        so it must appear in the retrieved context too.
        """
        llm = mock_llm(VALID_CRITICISM_JSON)
        agent = CriticAgent(llm)
        state = make_state()
        state.retrieved_docs = [make_doc("RAG combines retrieval and generation.")]
        state.answer = "RAG is retrieval-augmented generation."

        result = await agent._execute(state)

        assert result.is_valid is True
        assert result.validation_issues == []
        assert result.error == ""

    @pytest.mark.asyncio
    async def test_critic_flags_issues(self):
        llm = mock_llm(CRITICISM_WITH_ISSUES)
        agent = CriticAgent(llm)
        state = make_state()
        state.retrieved_docs = [make_doc()]
        state.answer = "Some answer"

        result = await agent._execute(state)

        assert result.is_valid is False
        assert "Claim X is not supported by context" in result.validation_issues

    @pytest.mark.asyncio
    async def test_json_parse_failure_surfaces_as_validation_issue(self):
        """
        Unparseable judge output no longer sets state.error — it's
        surfaced through validation_issues instead, marking the answer
        invalid without treating it as a hard pipeline failure.
        """
        llm = mock_llm("not json")
        agent = CriticAgent(llm)
        state = make_state()
        state.retrieved_docs = [make_doc()]
        state.answer = "Some answer"

        result = await agent._execute(state)

        assert result.is_valid is False
        assert "Could not parse critic response" in result.validation_issues

    @pytest.mark.asyncio
    async def test_empty_answer_sets_error_and_stays_empty(self):
        """
        No more placeholder text — CriticAgent now short-circuits with
        state.error set and leaves state.answer untouched (empty) when
        called before AnswerAgent has produced anything.
        """
        llm = mock_llm(VALID_CRITICISM_JSON)
        agent = CriticAgent(llm)
        state = make_state()
        state.retrieved_docs = [make_doc()]
        state.answer = ""

        result = await agent._execute(state)

        assert result.answer == ""
        assert "CriticAgent called before AnswerAgent produced an answer" in result.error
        assert result.is_valid is False
        llm.generate.assert_not_called()
# ─────────────────────────────────────────────
# ToolAgent
# ─────────────────────────────────────────────

class TestToolAgent:
    @pytest.mark.asyncio
    async def test_skips_when_web_and_tools_not_in_sources(self):
        llm = mock_llm()
        agent = ToolAgent(llm)
        state = make_state()
        state.sources_needed = ["documents"]

        with patch("app.services.tools.registry.tool_registry") as mock_reg:
            result = await agent._execute(state)
            mock_reg.execute_tool.assert_not_called()

        assert result.tool_results == {}
        assert result.error == ""

    @pytest.mark.asyncio
    async def test_calls_web_search_when_web_in_sources(self):
        llm = mock_llm()
        agent = ToolAgent(llm)
        state = make_state(question="Latest news?")
        state.sources_needed = ["web"]

        web_result = {"results": ["news item 1"], "count": 1}

        with patch(
            "app.agents.tool_agent.tool_registry"
        ) as mock_reg:
            mock_reg.execute_tool = AsyncMock(return_value=web_result)
            result = await agent._execute(state)

        mock_reg.execute_tool.assert_called_once_with(
            "web_search",
            query="Latest news?",
            max_results=5,
        )
        assert result.tool_results["web_search"] == web_result

    @pytest.mark.asyncio
    async def test_calls_web_search_when_tools_in_sources(self):
        llm = mock_llm()
        agent = ToolAgent(llm)
        state = make_state()
        state.sources_needed = ["tools"]  # "tools" also triggers web_search branch

        with patch("app.agents.tool_agent.tool_registry") as mock_reg:
            mock_reg.execute_tool = AsyncMock(return_value={"count": 0})
            result = await agent._execute(state)

        mock_reg.execute_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_exception_sets_error(self):
        llm = mock_llm()
        agent = ToolAgent(llm)
        state = make_state()
        state.sources_needed = ["web"]

        with patch("app.agents.tool_agent.tool_registry") as mock_reg:
            mock_reg.execute_tool = AsyncMock(side_effect=RuntimeError("tavily down"))
            result = await agent._execute(state)

        assert "Tool agent error" in result.error
        assert "tavily down" in result.error


# ─────────────────────────────────────────────
# AgentOrchestrator
# ─────────────────────────────────────────────




def _make_final_state(
    *,
    question="What is RAG?",
    user_id="u-1",
    answer="Final answer.",
    docs=None,
    error="",
):
    """Build the AgentState the compiled graph's ainvoke() would return."""
    if docs is None:
        docs = [make_doc()]
    state = make_state(question=question, user_id=user_id)
    state.error = error
    state.answer = answer
    if answer and not error:
        state.sources_needed = ["documents"]
        state.sources = [
            {"doc_id": d["doc_id"], "score": d["combined_score"],
             "text": d["text"][:200], "chunk_index": d["chunk_index"]}
            for d in docs
        ]
        state.confidence_final = 0.85
        state.is_valid = True
    state.search_time_ms = 10.0
    return state


class TestAgentOrchestrator:
    def _build_orchestrator(self, stack: ExitStack, ainvoke_result=None,
                             cache_hit=None):
        """
        Build an AgentOrchestrator with the graph-construction seam mocked.
        Patches (all via the given ExitStack, so they auto-unwind at the
        end of the `with` block in each test):
          - get_db                -> avoids a real Mongo connection
          - build_agent_graph     -> returns a fake compiled graph whose
                                      .ainvoke() returns `ainvoke_result`
          - query_cache           -> .get() returns `cache_hit` (None = miss
                                      by default), .set() is a no-op AsyncMock
        Returns (orch, mock_graph) — tests can still reach into mock_graph
        if they need to assert on how ainvoke was called.
        """
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=ainvoke_result)

        stack.enter_context(patch(
            "app.agents.orchestrator.get_db",
            AsyncMock(return_value=MagicMock()),
        ))
        stack.enter_context(patch(
            "app.agents.orchestrator.build_agent_graph",
            return_value=mock_graph,
        ))

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=cache_hit)
        mock_cache.set = AsyncMock()
        stack.enter_context(patch(
            "app.agents.orchestrator.query_cache", mock_cache,
        ))

        orch = AgentOrchestrator()
        return orch, mock_graph

    @pytest.mark.asyncio
    async def test_full_happy_path_returns_expected_keys(self):
        final_state = _make_final_state()

        with ExitStack() as stack:
            orch, _ = self._build_orchestrator(stack, ainvoke_result=final_state)
            stack.enter_context(patch("app.services.memory.manager.memory_manager", None))
            result = await orch.process("What is RAG?", "u-1")

        assert set(result.keys()) == {
            "answer", "sources", "sources_needed", "confidence",
            "search_time_ms", "is_valid",
        }

    @pytest.mark.asyncio
    async def test_full_happy_path_answer_and_confidence(self):
        final_state = _make_final_state(answer="RAG uses retrieval.")

        with ExitStack() as stack:
            orch, _ = self._build_orchestrator(stack, ainvoke_result=final_state)
            stack.enter_context(patch("app.services.memory.manager.memory_manager", None))
            result = await orch.process("What is RAG?", "u-1")

        assert result["answer"] == "RAG uses retrieval."
        assert result["confidence"] == pytest.approx(0.85, abs=1e-3)

    @pytest.mark.asyncio
    async def test_graph_error_with_no_answer_returns_error_response(self):
        """
        Covers what were previously separate 'planner_error' /
        'retriever_error' short-circuit tests. Both agents now live
        inside the compiled graph, invisible to the orchestrator — the
        orchestrator only ever sees the graph's *final* state. Whether
        the error originated in the planner node or the retriever node,
        the orchestrator's contract is identical: error set + no answer
        -> return _error_response().
        """
        final_state = _make_final_state(answer="", error="Retriever error: qdrant down")

        with ExitStack() as stack:
            orch, _ = self._build_orchestrator(stack, ainvoke_result=final_state)
            stack.enter_context(patch("app.services.memory.manager.memory_manager", None))
            result = await orch.process("q", "u-1")

        assert "Error" in result["answer"]
        assert result["confidence"] == 0.0
        assert result["is_valid"] is False

    @pytest.mark.asyncio
    async def test_error_set_but_answer_present_is_not_treated_as_fatal(self):
        """
        A node upstream (e.g. Planner) can set `error` on a JSON-parse
        failure that the graph already recovered from gracefully. If the
        pipeline still produced a real answer, process() must return it
        normally instead of routing to _error_response().
        """
        final_state = _make_final_state(answer="RAG uses retrieval.")
        final_state.error = "Failed to parse planner response as JSON"

        with ExitStack() as stack:
            orch, _ = self._build_orchestrator(stack, ainvoke_result=final_state)
            stack.enter_context(patch("app.services.memory.manager.memory_manager", None))
            result = await orch.process("q", "u-1")

        assert result["answer"] == "RAG uses retrieval."
        assert not result["answer"].startswith("Error:")

    @pytest.mark.asyncio
    async def test_memory_save_called_on_success(self):
        final_state = _make_final_state()

        mock_mm = MagicMock()
        mock_mm.save_interaction = AsyncMock(return_value=True)

        with ExitStack() as stack:
            orch, _ = self._build_orchestrator(stack, ainvoke_result=final_state)
            stack.enter_context(patch("app.services.memory.manager.memory_manager", mock_mm))
            # orchestrator.py references mm_module.memory_manager at call
            # time, so patch it where it's imported too — belt and braces.
            import app.agents.orchestrator as orch_module
            stack.enter_context(patch.object(orch_module.mm_module, "memory_manager", mock_mm))

            await orch.process("What is RAG?", "u-1")

        mock_mm.save_interaction.assert_called_once_with(
            user_id="u-1",
            session_id="default_session",
            user_message="What is RAG?",
            assistant_response="Final answer.",
        )

    @pytest.mark.asyncio
    async def test_memory_manager_none_does_not_crash(self):
        final_state = _make_final_state()

        with ExitStack() as stack:
            orch, _ = self._build_orchestrator(stack, ainvoke_result=final_state)
            stack.enter_context(patch("app.services.memory.manager.memory_manager", None))
            import app.agents.orchestrator as orch_module
            stack.enter_context(patch.object(orch_module.mm_module, "memory_manager", None))

            result = await orch.process("q", "u-1")

        assert "answer" in result  # no exception raised

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result_without_invoking_graph(self):
        """New behavior (Option B): a cache hit must short-circuit before
        the graph ever runs."""
        cached = {"answer": "cached answer", "sources": [], "sources_needed": [],
                   "confidence": 0.9, "search_time_ms": 0.0, "is_valid": True}

        with ExitStack() as stack:
            orch, mock_graph = self._build_orchestrator(stack, cache_hit=cached)
            result = await orch.process("What is RAG?", "u-1")

        assert result == cached
        mock_graph.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_answer_is_not_cached(self):
        """2026-07-04 regression guard: a fallback failure string must
        never be written to query_cache, or it gets served forever."""
        final_state = _make_final_state(answer="Sorry, I couldn't generate an answer.")
        final_state.confidence_final = 0.0  # matches real crash scenario

        with ExitStack() as stack:
            orch, _ = self._build_orchestrator(stack, ainvoke_result=final_state)
            stack.enter_context(patch("app.services.memory.manager.memory_manager", None))
            await orch.process("q", "u-1")
            from app.agents.orchestrator import query_cache
            query_cache.set.assert_not_called()

    def test_error_response_structure(self):
        state = make_state()
        state.error = "Something went wrong"
        orch = AgentOrchestrator()  # no graph needed for this helper method
        resp = orch._error_response(state)

        assert resp["answer"].startswith("Error:")
        assert resp["sources"] == []
        assert resp["confidence"] == 0.0
        assert resp["is_valid"] is False
        assert "search_time_ms" in resp
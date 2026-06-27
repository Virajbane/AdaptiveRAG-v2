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
        assert result.error == ""

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

        assert result.error == "Failed to parse planner response"

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
    async def test_skips_search_when_documents_not_in_sources(self):
        llm = mock_llm()
        agent = RetrieverAgent(llm)
        state = make_state()
        state.sources_needed = ["web"]  # no "documents"

        with patch("app.agents.retriever.HybridSearchEngine") as MockHS:
            result = await agent._execute(state)
            MockHS.assert_not_called()

        assert result.retrieved_docs == []
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

        with patch("app.agents.retriever.HybridSearchEngine", return_value=mock_engine):
            result = await agent._execute(state)

        assert result.retrieved_docs == docs

    @pytest.mark.asyncio
    async def test_search_called_with_correct_args(self):
        llm = mock_llm()
        agent = RetrieverAgent(llm)
        state = make_state(question="What is FAISS?", user_id="u-42")
        state.sources_needed = ["documents"]

        mock_engine = MagicMock()
        mock_engine.search = AsyncMock(return_value=[])

        with patch("app.agents.retriever.HybridSearchEngine", return_value=mock_engine):
            await agent._execute(state)

        mock_engine.search.assert_called_once_with(
            query="What is FAISS?",
            user_id="u-42",
            top_k=5,
        )

    @pytest.mark.asyncio
    async def test_search_time_ms_is_set(self):
        llm = mock_llm()
        agent = RetrieverAgent(llm)
        state = make_state()
        state.sources_needed = ["documents"]

        mock_engine = MagicMock()
        mock_engine.search = AsyncMock(return_value=[make_doc()])

        with patch("app.agents.retriever.HybridSearchEngine", return_value=mock_engine):
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

        with patch("app.agents.retriever.HybridSearchEngine", return_value=mock_engine):
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

        result = await agent._execute(state)

        assert "upload" in result.answer.lower()
        assert result.sources == []
        # Confidence is capped at 0.3 * planner confidence
        assert result.confidence_final == pytest.approx(0.8 * 0.3, abs=1e-9)
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_happy_path_sets_answer_and_sources(self):
        llm = mock_llm("RAG stands for Retrieval-Augmented Generation.")
        agent = AnswerAgent(llm)
        state = make_state()
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
        llm = mock_llm(VALID_CRITICISM_JSON)
        agent = CriticAgent(llm)
        state = make_state()
        state.retrieved_docs = [make_doc()]
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
    async def test_json_parse_failure_sets_error(self):
        llm = mock_llm("not json")
        agent = CriticAgent(llm)
        state = make_state()
        state.retrieved_docs = [make_doc()]
        state.answer = "Some answer"

        result = await agent._execute(state)

        assert "Critic error" in result.error
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_placeholder_answer_set_when_answer_empty(self):
        """Source documents the placeholder behaviour when answer is empty."""
        llm = mock_llm(VALID_CRITICISM_JSON)
        agent = CriticAgent(llm)
        state = make_state()
        state.retrieved_docs = [make_doc()]
        state.answer = ""  # empty — critic sets placeholder before LLM call

        result = await agent._execute(state)

        # After _execute, answer should have been set to the placeholder
        assert result.answer != ""

    # Known gap: CriticAgent fires an LLM call even when retrieved_docs is
    # empty (context="" but prompt still sent). No fix; documented only.


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

def _make_orchestrator_mocks(
    *,
    plan_json=None,
    docs=None,
    answer_text="Final answer.",
):
    """
    Return a dict of AsyncMock patches for the orchestrator's four agents.
    Each mock's run() updates state as the real agent would.
    """
    if plan_json is None:
        plan_json = VALID_PLAN_JSON
    if docs is None:
        docs = [make_doc()]

    async def planner_run(state):
        plan = json.loads(plan_json)
        state.plan = plan.get("strategy", "")
        state.sources_needed = plan.get("sources", ["documents"])
        state.confidence = plan.get("confidence", 0.5)
        return state

    async def retriever_run(state):
        state.retrieved_docs = docs
        state.search_time_ms = 10.0
        return state

    async def answer_run(state):
        state.answer = answer_text
        state.sources = [{"doc_id": d["doc_id"], "score": d["combined_score"],
                          "text": d["text"][:200], "chunk_index": d["chunk_index"]}
                         for d in state.retrieved_docs]
        state.confidence_final = 0.85
        return state

    return {
        "planner_run": planner_run,
        "retriever_run": retriever_run,
        "answer_run": answer_run,
    }


class TestAgentOrchestrator:
    def _build_orchestrator(self):
        """Build orchestrator with all LLM/external calls mocked."""
        with patch("app.agents.orchestrator.LLMProvider"):
            orch = AgentOrchestrator()
        return orch

    @pytest.mark.asyncio
    async def test_full_happy_path_returns_expected_keys(self):
        orch = self._build_orchestrator()
        mocks = _make_orchestrator_mocks()

        orch.planner.run = AsyncMock(side_effect=mocks["planner_run"])
        orch.retriever.run = AsyncMock(side_effect=mocks["retriever_run"])
        orch.answer.run = AsyncMock(side_effect=mocks["answer_run"])

        with patch("app.services.memory.manager.memory_manager", None):
            result = await orch.process("What is RAG?", "u-1")

        assert set(result.keys()) == {
            "answer", "sources", "confidence", "search_time_ms", "is_valid"
        }

    @pytest.mark.asyncio
    async def test_full_happy_path_answer_and_confidence(self):
        orch = self._build_orchestrator()
        mocks = _make_orchestrator_mocks(answer_text="RAG uses retrieval.")

        orch.planner.run = AsyncMock(side_effect=mocks["planner_run"])
        orch.retriever.run = AsyncMock(side_effect=mocks["retriever_run"])
        orch.answer.run = AsyncMock(side_effect=mocks["answer_run"])

        with patch("app.services.memory.manager.memory_manager", None):
            result = await orch.process("What is RAG?", "u-1")

        assert result["answer"] == "RAG uses retrieval."
        assert result["confidence"] == pytest.approx(0.85, abs=1e-3)

    @pytest.mark.asyncio
    async def test_planner_error_short_circuits(self):
        orch = self._build_orchestrator()

        async def planner_error(state):
            state.error = "PlannerAgent error: parse failed"
            return state

        orch.planner.run = AsyncMock(side_effect=planner_error)
        orch.retriever.run = AsyncMock(side_effect=lambda state: state)

        with patch("app.services.memory.manager.memory_manager", None):
            result = await orch.process("q", "u-1")

        # Planner + Retriever run in parallel (Phase 11 optimization),
        # so retriever DOES start even when planner errors.
        # The orchestrator still correctly surfaces the planner error
        # in the final result rather than proceeding with a normal answer.
        orch.retriever.run.assert_called_once()
        assert "Error" in result["answer"]
        assert result["confidence"] == 0.0
        assert result["is_valid"] is False

    @pytest.mark.asyncio
    async def test_retriever_error_short_circuits(self):
        orch = self._build_orchestrator()
        mocks = _make_orchestrator_mocks()

        async def retriever_error(state):
            state.error = "Retriever error: qdrant down"
            return state

        orch.planner.run = AsyncMock(side_effect=mocks["planner_run"])
        orch.retriever.run = AsyncMock(side_effect=retriever_error)
        orch.answer.run = AsyncMock()  # must NOT be called

        with patch("app.services.memory.manager.memory_manager", None):
            result = await orch.process("q", "u-1")

        orch.answer.run.assert_not_called()
        assert "Error" in result["answer"]

    @pytest.mark.asyncio
    async def test_memory_save_called_on_success(self):
        orch = self._build_orchestrator()
        mocks = _make_orchestrator_mocks()

        orch.planner.run = AsyncMock(side_effect=mocks["planner_run"])
        orch.retriever.run = AsyncMock(side_effect=mocks["retriever_run"])
        orch.answer.run = AsyncMock(side_effect=mocks["answer_run"])

        mock_mm = MagicMock()
        mock_mm.save_interaction = AsyncMock(return_value=True)

        import app.services.memory.manager as mm_module
        original = mm_module.memory_manager
        mm_module.memory_manager = mock_mm
        try:
            await orch.process("What is RAG?", "u-1")
        finally:
            mm_module.memory_manager = original

        mock_mm.save_interaction.assert_called_once_with(
            user_id="u-1",
            session_id="default_session",
            user_message="What is RAG?",
            assistant_response="Final answer.",
        )

    @pytest.mark.asyncio
    async def test_memory_manager_none_does_not_crash(self):
        """memory_manager starts as None in main.py; process() must tolerate it."""
        orch = self._build_orchestrator()
        mocks = _make_orchestrator_mocks()

        orch.planner.run = AsyncMock(side_effect=mocks["planner_run"])
        orch.retriever.run = AsyncMock(side_effect=mocks["retriever_run"])
        orch.answer.run = AsyncMock(side_effect=mocks["answer_run"])

        with patch("app.services.memory.manager.memory_manager", None):
            result = await orch.process("q", "u-1")

        assert "answer" in result  # no exception raised

    @pytest.mark.asyncio
    async def test_error_response_structure(self):
        orch = self._build_orchestrator()
        state = make_state()
        state.error = "Something went wrong"
        resp = orch._error_response(state)

        assert resp["answer"].startswith("Error:")
        assert resp["sources"] == []
        assert resp["confidence"] == 0.0
        assert resp["is_valid"] is False
        assert "search_time_ms" in resp
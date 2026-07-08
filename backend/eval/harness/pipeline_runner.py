# eval/harness/pipeline_runner.py
"""
Single source of truth for "run one question through the real pipeline
and get back everything an evaluator could need."

Every framework-specific runner (ragas_eval, deepeval_suite, router_eval,
retrieval_eval) imports run_single() instead of touching AgentOrchestrator
directly. This means if AgentState grows a field tomorrow, only this file
changes.

NOTE: In production, tools (e.g. web search) are initialized by main.py's
FastAPI startup event. This harness never boots FastAPI, so that startup
event never fires — we have to initialize tools explicitly here, once,
at import time, or every tool call in every eval framework silently
resolves to "tool not found" (masked as "0 results" by ToolAgent's logging).
"""
import time
from dataclasses import dataclass, field

from app.services.tools.web_search import init_web_search
from app.config.settings import settings

# Must run before AgentOrchestrator() is constructed / used, so that
# ToolRegistry.get_tools() sees a populated web_search_tool on every
# run_single() call across the whole eval suite.
init_web_search(settings.TAVILY_API_KEY)

from app.agents.orchestrator import AgentOrchestrator
from app.agents.state import AgentState

_orchestrator = AgentOrchestrator()


@dataclass
class PipelineResult:
    question: str
    answer: str
    contexts: list[str]              # text of retrieved_docs, Ragas/DeepEval expect list[str]
    sources_needed: list[str]        # for router evaluation
    confidence: float
    is_valid: bool
    latency_s: float
    raw_state: dict = field(default_factory=dict)  # full AgentState for debugging


async def run_single(question: str, user_id: str, session_id: str = "eval_session") -> PipelineResult:
    start = time.perf_counter()
    result = await _orchestrator.process(question=question, user_id=user_id, session_id=session_id)
    elapsed = time.perf_counter() - start

    return PipelineResult(
        question=question,
        answer=result.get("answer", ""),
        contexts=[d.get("text", "") for d in result.get("retrieved_docs", [])] if "retrieved_docs" in result else [],
        sources_needed=result.get("sources_needed", []),
        confidence=result.get("confidence", 0.0),
        is_valid=result.get("is_valid", False),
        latency_s=elapsed,
        raw_state=result,
    )
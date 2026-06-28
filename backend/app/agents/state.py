from typing import Any, Optional, Annotated
from dataclasses import dataclass, field


def _merge_error(a: str, b: str) -> str:
    """
    Reducer for the `error` field.

    When planner_node and retriever_node both run in the same superstep
    (parallel fan-out from START or retry_dispatch), LangGraph may receive
    two writes to `error` in one step. This reducer tells LangGraph how to
    merge them instead of raising InvalidUpdateError.

    Rule: keep the most recent non-empty error; if both are non-empty,
    concatenate with a separator so neither is silently dropped.
    If both are empty/None, return empty string.
    """
    a = a or ""
    b = b or ""
    if a and b:
        return f"{a} | {b}"
    return a or b


@dataclass
class AgentState:
    """
    Shared state passed between agents.

    IMPORTANT: This state is TEMPORARY
    - Created when user sends a message
    - Destroyed after response is sent
    - NOT persisted to database
    """

    def copy(self):
        """Create a shallow copy of the state."""
        from dataclasses import replace
        return replace(self)

    # ── Input ────────────────────────────────────────────────────────────
    question: str
    user_id: str

    # ── Planner output ───────────────────────────────────────────────────
    plan: str = ""
    sources_needed: list[str] = field(default_factory=list)  # ["documents", "web"]
    confidence: float = 0.0

    # ── Retrieved data ───────────────────────────────────────────────────
    retrieved_docs: list[dict] = field(default_factory=list)
    web_results: list[dict] = field(default_factory=list)

    # ── Tool results ─────────────────────────────────────────────────────
    tool_results: dict[str, Any] = field(default_factory=dict)

    # ── Critic validation ────────────────────────────────────────────────
    is_valid: bool = False
    validation_issues: list[str] = field(default_factory=list)
    critic_confidence: float = 0.0   # 0-1; set by CriticAgent each run

    # ── Retry control ────────────────────────────────────────────────────
    # Guards the Critic → Answer retry loop in the LangGraph StateGraph.
    # Without this, a persistently low-confidence query would retry forever
    # instead of degrading gracefully.
    retry_count: int = 0

    # ── Final answer ─────────────────────────────────────────────────────
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    confidence_final: float = 0.0   # blended: rerank score + critic score

    # ── Metadata ─────────────────────────────────────────────────────────
    search_time_ms: float = 0.0

    # Annotated so LangGraph uses _merge_error when two parallel nodes
    # (planner + retriever) both write to this field in the same superstep.
    error: Annotated[str, _merge_error] = ""
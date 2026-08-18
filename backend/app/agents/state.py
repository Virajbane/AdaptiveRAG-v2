import hashlib
from typing import Any, Optional, Annotated
from dataclasses import dataclass, field


def _merge_error(a: str, b: str) -> str:
    """
    Reducer for the `error` field.

    When planner_node and retriever_node both run in the same superstep
    (parallel fan-out from START or retry_dispatch), LangGraph may receive
    two writes to `error` in one step. This reducer tells LangGraph how to
    merge them instead of raising InvalidUpdateError.

    Rule: keep the most recent non-empty error; if both are non-empty AND
    different, concatenate with a separator so neither is silently dropped.
    If both are empty/None, return empty string.

    IMPORTANT: tool_node/answer_node/critic_node return the FULL AgentState
    (not a partial dict), so they re-submit the unchanged `error` value on
    every step even when nothing new went wrong. Without the a == b check
    below, the same error string gets concatenated with itself repeatedly
    (e.g. "X | X | X | X") as it passes through each subsequent node.
    """
    a = a or ""
    b = b or ""
    if not a:
        return b
    if not b:
        return a
    if a == b:
        return a
    return f"{a} | {b}"


def _compute_hash(obj: Any) -> str:
    """
    Deterministic hash of an object for change detection.
    
    Used to detect if retriever/planner/answer produced identical output
    across retries. If hashes match, stop retrying (early stop logic).
    Converts to JSON string to ensure determinism.
    """
    import json
    try:
        s = json.dumps(obj, sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:12]
    except (TypeError, ValueError):
        return hashlib.md5(str(obj).encode()).hexdigest()[:12]


@dataclass
class AgentState:
    """
    Shared state passed between agents.

    IMPORTANT: This state is TEMPORARY
    - Created when user sends a message
    - Destroyed after response is sent
    - NOT persisted to database
    
    2026-07-XX: Added failure_type classification, independent retry
    counters per component, and hash-based change detection to enable
    smart routing and early stopping.
    
    2026-08-06: Added low_confidence flag. Set by GraderAgent when 
    retrieval quality is weak (top_score < ABSOLUTE_FLOOR). Downstream 
    agents (AnswerAgent, CriticAgent) use this to adjust behavior.
    """

    def copy(self):
        """Create a shallow copy of the state."""
        from dataclasses import replace
        return replace(self)

    # ── Input ────────────────────────────────────────────────────────────
    question: str
    user_id: str
    session_id: str = "default_session"   # used to look up Redis short-term history
    rewritten_question: str = ""          # set by RewriterAgent; "" = not rewritten,
                                           # downstream nodes fall back to `question`

    # ── Planner output ───────────────────────────────────────────────────
    plan: str = ""
    sources_needed: list[str] = field(default_factory=list)  # ["documents", "web"]
    # Concrete tool name. Set by tool_mapping.resolve_concrete_tool() to map
    # source→tool according to contract: documents→document_retrieval,
    # database→sql_query, calculator→calculator, web→web_search,
    # tool+weather→weather, direct_llm→None. Never None for implemented sources.
    tool: Optional[str] = None
    confidence: float = 0.0

    # ── Retrieved data ───────────────────────────────────────────────────
    retrieved_docs: list[dict] = field(default_factory=list)
    web_results: list[dict] = field(default_factory=list)
    retrieval_rejected: bool = False   
    
    # ── Retrieval quality assessment ──────────────────────────────────────
    # Set by GraderAgent. Signals whether top-score retrieval result was 
    # weak (below ABSOLUTE_FLOOR threshold). Downstream agents use this 
    # to decide whether to proceed cautiously or decline.
    low_confidence: bool = False
    
    # ── Metadata short-circuit ───────────────────────────────────────────
    metadata_answer: dict = field(default_factory=dict)
    
    # ── Tool results ─────────────────────────────────────────────────────
    tool_results: dict[str, Any] = field(default_factory=dict)

    # ── Critic validation ────────────────────────────────────────────────
    is_valid: bool = False
    validation_issues: list[str] = field(default_factory=list)
    critic_confidence: float = 0.0   # 0-1; set by CriticAgent each run
    
    # ── Failure classification ────────────────────────────────────────────
    # Set by CriticAgent. Determines which component should be retried.
    # Values: "generation", "retrieval", "planning", "tool", "unknown"
    # If "unknown", the answer is returned as-is (no further retries).
    failure_type: str = ""

    # ── Independent retry counters ────────────────────────────────────────
    # Each component tracks its own retry budget. Prevents retrying a
    # component that's already exhausted its attempts.
    planner_retry_count: int = 0
    retriever_retry_count: int = 0
    answer_retry_count: int = 0
    tool_retry_count: int = 0
    
    # Legacy counter kept for backward compatibility, but not actively used
    # in the new routing logic.
    retry_count: int = 0

    # ── Change detection hashes ───────────────────────────────────────────
    # Stored after each component runs. If the next retry produces an
    # identical hash, we stop retrying (early stop) instead of burning
    # retry budget on unchanged output.
    last_planning_hash: str = ""    # hash of (plan, sources_needed)
    last_retrieval_hash: str = ""   # hash of retrieved_docs scores/IDs
    last_answer_hash: str = ""      # hash of answer text
    last_tool_result_hash: str = "" # hash of tool_results dict

    # ── Final answer ─────────────────────────────────────────────────────
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    confidence_final: float = 0.0   # blended: rerank score + critic score

    # ── Metadata ─────────────────────────────────────────────────────────
    search_time_ms: float = 0.0

    # Annotated so LangGraph uses _merge_error when two parallel nodes
    # (planner + retriever) both write to this field in the same superstep.
    error: Annotated[str, _merge_error] = ""
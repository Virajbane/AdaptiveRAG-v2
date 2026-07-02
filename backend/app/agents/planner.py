import json
import re

from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.agents.prompts import PLANNER_PROMPT

# --------------------------------------------------------------------------
# 2026-06-30 routing bug backstop
#
# The Planner LLM has repeatedly classified personal-reference questions
# ("my"/"I"/"me") as public-fact/web lookups, even though PLANNER_PROMPT
# explicitly rules this out ('Contains "my"/"I"/"me" about content ->
# sources: ["documents"]'). Root cause appears to be words like "the docs"
# reading ambiguously as generic documentation rather than "my uploaded
# documents" — this pulls the model toward "public fact" despite the
# explicit rule. Since prompt wording alone hasn't reliably fixed this
# (same failure mode seen twice), we backstop it deterministically here.
#
# Deliberately narrow: only overrides when there's a personal reference
# AND no comparison/external-info signal, so legitimate
# ["documents", "web"] compare-my-data-against-industry cases are
# untouched.
#
# 2026-07-02: added a symmetrical backstop below (_needs_both) for the
# opposite failure mode — personal reference + comparison signal, where
# the LLM sometimes drops "documents" from sources_needed entirely
# (routing_005_both: expected ["documents", "web"], got ["web"]).
# --------------------------------------------------------------------------
_PERSONAL_REF = re.compile(r"\b(my|i|me)\b", re.IGNORECASE)
_COMPARISON_HINT = re.compile(
    r"\b(compare|comparison|vs\.?|versus|industry|market|benchmark|external|competitors?)\b",
    re.IGNORECASE,
)

_VALID_SOURCES = {"documents", "web", "tools"}


def _looks_personal_only(question: str) -> bool:
    """
    True if the question references the user's own content ("my"/"I"/"me")
    with no signal it also needs external/public info.
    """
    return bool(_PERSONAL_REF.search(question)) and not _COMPARISON_HINT.search(question)


def _needs_both(question: str) -> bool:
    """
    True if the question references the user's own content AND has a
    comparison/external-info signal — the "both" case where the LLM has
    been observed to drop 'documents' from sources_needed even though
    the personal reference means the user's own data must be pulled in
    (e.g. "how does my resume compare to industry benchmarks").

    Mutually exclusive with _looks_personal_only by construction: one
    requires the absence of a comparison hint, the other requires its
    presence. Safe to check as an elif in _execute.
    """
    return bool(_PERSONAL_REF.search(question)) and bool(_COMPARISON_HINT.search(question))


def _normalize_sources(raw_sources) -> list[str]:
    """
    2026-07-02 fix: the planner LLM occasionally emits a nested structure
    for "sources" — e.g. [["documents"], ["web"]] instead of the expected
    flat ["documents", "web"] — even though the prompt asks for a flat
    list. Nothing downstream expects nesting (ToolAgent does plain
    `"web" in state.sources_needed` checks), so an unflattened list
    silently disables web/tool routing without raising any error.

    This flattens one level of nesting, coerces every leaf to a string,
    drops anything not in the allowed source set, dedupes while
    preserving order, and falls back to ["documents"] if nothing valid
    survives — same safe default used elsewhere in this file when the
    LLM response can't be trusted.
    """
    if not isinstance(raw_sources, list):
        return ["documents"]

    flat: list = []
    for item in raw_sources:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)

    cleaned: list[str] = []
    for item in flat:
        if not isinstance(item, str):
            continue
        item = item.strip().lower()
        if item in _VALID_SOURCES and item not in cleaned:
            cleaned.append(item)

    return cleaned if cleaned else ["documents"]


class PlannerAgent(BaseAgent):
    """
    Planner Agent: Decides what sources to use

    - Analyzes user intent
    - Chooses retrieval strategy
    - Sets confidence level

    Note: model selection (fast vs. main LLM) is handled by
    AgentOrchestrator, which injects the right LLMProvider instance.
    This class does not need its own __init__.
    """

    async def _execute(self, state: AgentState) -> AgentState:
        """Plan the search strategy"""

        question_for_routing = state.rewritten_question or state.question

        prompt = PLANNER_PROMPT.format(question=question_for_routing)

        try:
            response = await self.call_llm(prompt)
        except Exception as e:
            # Surface the REAL failure (e.g. model not found, connection
            # refused, timeout) instead of letting it fall through to a
            # JSON parse error that hides the actual cause.
            state.error = f"Planner LLM call failed: {str(e)}"
            state.sources_needed = ["documents"]
            print(f"[PLANNER] LLM call failed: {e}")
            return state

        try:
            plan = self.parse_json_response(response)

            state.plan = plan.get("strategy", "")

            raw_sources = plan.get("sources", ["documents"])
            state.sources_needed = _normalize_sources(raw_sources)
            if raw_sources != state.sources_needed:
                print(
                    f"[PLANNER] Normalized malformed sources: "
                    f"{raw_sources!r} -> {state.sources_needed!r}"
                )

            state.confidence = plan.get("confidence", 0.5)

            print(f"[PLANNER] Plan: {state.plan}")
            print(f"[PLANNER] Sources: {state.sources_needed}")
            print(f"[PLANNER] Confidence: {state.confidence}")

        except json.JSONDecodeError:
            state.error = "Failed to parse planner response as JSON"
            state.sources_needed = ["documents"]
            print(f"[PLANNER] Could not parse response as JSON: {response[:200]!r}")
            return state

        # ---- Deterministic override: see module-level note above ----
        if _looks_personal_only(question_for_routing) and state.sources_needed != ["documents"]:
            print(
                f"[PLANNER] Override: personal-reference question with no comparison "
                f"signal, forcing sources=['documents'] (LLM said {state.sources_needed})"
            )
            state.sources_needed = ["documents"]

        # ---- Deterministic override: personal + comparison needs both ----
        elif _needs_both(question_for_routing) and "documents" not in state.sources_needed:
            print(
                f"[PLANNER] Override: personal-reference question with comparison "
                f"signal, adding 'documents' to sources (LLM said {state.sources_needed})"
            )
            state.sources_needed = ["documents"] + state.sources_needed

        return state
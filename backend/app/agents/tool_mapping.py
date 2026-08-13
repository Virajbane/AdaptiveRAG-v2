"""
source -> concrete tool name mapping.

Root cause this module fixes (AMB_09 / "tool selection accuracy = 0%"):
  AgentState never had a field for the *concrete* tool (e.g. "weather",
  "web_search"), only `sources_needed` (the abstract source, e.g. "tool",
  "web"). ToolAgent worked out a concrete tool internally at execution
  time, but that decision was never written back onto the state, so
  nothing upstream of execution (e.g. an eval harness, or the Answer
  agent) could ever observe which concrete tool was actually chosen --
  it was always missing/None, no matter how good the routing was.

This module is the single place that turns:
    source ("calculator" / "tool" / "web" / "database" / "documents" / "direct_llm")
into:
    concrete tool name ("calculator" / "weather" | "slack" | "email" / "web_search" / ...)

so the planner can set state.tool immediately after routing, and
tool_agent's `_handle_tool` sub-dispatch (weather vs slack vs email) uses
the exact same keyword detection instead of a second, potentially
divergent, copy of it.
"""

import re
from typing import Optional

# Weather takes priority over messaging keywords regardless of what else
# is in the question (mirrors tool_agent.py's existing behavior).
WEATHER_KEYWORDS = re.compile(
    r"\b(weather|temperature|forecast|climate|temp|rain|snow|sunny|cloudy|"
    r"humid|windy|precipitation)\b|will\s+it\s+(?:rain|snow)",
    re.IGNORECASE,
)
SLACK_EXPLICIT = re.compile(r"\bslack\b", re.IGNORECASE)
EMAIL_EXPLICIT = re.compile(r"\b(email|e-mail|mail)\b", re.IGNORECASE)
SLACK_KEYWORDS = re.compile(r"\b(slack|post|alert|channel)\b", re.IGNORECASE)
EMAIL_KEYWORDS = re.compile(r"\b(email|e-mail|mail|send)\b", re.IGNORECASE)

# Sources with one unambiguous concrete tool.
_DIRECT_TOOL_FOR_SOURCE = {
    "calculator": "calculator",
    "web": "web_search",
}

# Sources with no concrete "tool" in the tool_registry sense -- they're
# either pure LLM generation or a retrieval path, not a callable tool.
_NO_CONCRETE_TOOL_SOURCES = {"direct_llm", "documents", "database"}


def resolve_sub_tool(question: str) -> str:
    """
    For the generic "tool" source, work out which concrete integration
    (weather / slack / email) the question actually wants.

    Same precedence tool_agent.ToolAgent._handle_tool uses: weather
    first (it never collides with messaging keywords), then whichever
    product is named explicitly, then generic keyword fallback, then
    "tool" itself if nothing more specific is found.
    """
    q = question.lower()

    if WEATHER_KEYWORDS.search(q):
        return "weather"

    slack_named = bool(SLACK_EXPLICIT.search(q))
    email_named = bool(EMAIL_EXPLICIT.search(q))

    if slack_named and not email_named:
        return "slack"
    if email_named and not slack_named:
        return "email"
    if slack_named and email_named:
        return "email"  # ambiguous -- defaults to the safer, more targeted channel

    if SLACK_KEYWORDS.search(q):
        return "slack"
    if EMAIL_KEYWORDS.search(q):
        return "email"

    return "tool"


def resolve_concrete_tool(sources: list[str], question: str) -> Optional[str]:
    """
    Map a planner routing decision to a single concrete tool name.

    Returns None when:
      - the source has no concrete tool (direct_llm / documents / database), or
      - more than one source was selected (multi-source routing has no
        single concrete tool -- each source is handled independently
        downstream), or
      - sources is empty.
    """
    if not sources or len(sources) != 1:
        return None

    source = sources[0]

    if source in _DIRECT_TOOL_FOR_SOURCE:
        return _DIRECT_TOOL_FOR_SOURCE[source]

    if source == "tool":
        return resolve_sub_tool(question)

    if source in _NO_CONCRETE_TOOL_SOURCES:
        return None

    return None
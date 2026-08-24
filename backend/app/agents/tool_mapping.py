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
    concrete tool name ("calculator" / "weather" | "slack_search" | "slack" | "email" / "web_search" / ...)

so the planner can set state.tool immediately after routing, and
tool_agent's `_handle_tool` sub-dispatch (weather vs slack vs email) uses
the exact same keyword detection instead of a second, potentially
divergent, copy of it.

STAGE 15 FIX (2026-08-22, item 6.6):
  Added "humidity" to WEATHER_KEYWORDS so "What is the humidity in
  Visakhapatnam right now?" correctly resolves to "weather" sub-tool
  instead of generic "tool".

STAGE 16+ FIX (2026-08-22, root cause 6.5):
  Slack routing now distinguishes search intent ("find/search Slack
  messages") from post intent ("send/post a Slack message"):
  - Search intent -> "slack_search"  (backed by SlackTool.search_messages)
  - Post intent   -> "slack"         (backed by SlackTool.post_message)
  Previously every Slack question resolved to "slack" regardless of
  intent, so the eval's expected concrete tool "slack_search" was never
  reached.
"""

import re
from typing import Optional

# Weather takes priority over messaging keywords regardless of what else
# is in the question (mirrors tool_agent.py's existing behavior).
WEATHER_KEYWORDS = re.compile(
    r"\b(weather|temperature|forecast|climate|temp|humidity|rain|snow|sunny|cloudy|"
    r"humid|wind|windy|precipitation)\b|will\s+it\s+(?:rain|snow)",
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
    "documents": "document_retrieval",
    "database": "sql_query",
}

# Sources with no concrete "tool" in the tool_registry sense -- they're
# either pure LLM generation or a retrieval path, not a callable tool.
_NO_CONCRETE_TOOL_SOURCES = {"direct_llm"}


# Slack SEARCH signals: find/search/what did/what was/discussions/mention/
# conversation/messages -- the user wants to READ Slack history, not post.
# Checked BEFORE the generic SLACK_KEYWORDS so that questions like
# "Find Slack messages about X" go to slack_search, not slack (post).
SLACK_SEARCH_KEYWORDS = re.compile(
    r"\b(?:"
    r"find|search|look\s+for|what\s+did|what\s+was|who\s+said|did\s+anyone|"
    r"discuss(?:ion|ions|ed)?|mention(?:ed|s)?|conversation|messages?|thread|threads|"
    r"said|talked?|brought\s+up"
    r")\b",
    re.IGNORECASE,
)
# Slack POST signals: send/post/write/notify/alert -- the user wants to WRITE.
SLACK_POST_KEYWORDS = re.compile(
    r"\b(?:send|post|write|compose|notify|alert|dm|message|ping)\b",
    re.IGNORECASE,
)


def resolve_sub_tool(question: str) -> str:
    """
    For the generic "tool" source, work out which concrete integration
    (weather / slack_search / slack / email) the question actually wants.

    Precedence:
      1. Weather (never collides with messaging keywords)
      2. Slack -- search vs post discrimination:
         - search keywords AND \bslack\b  -> "slack_search"
         - post  keywords AND \bslack\b   -> "slack"
         - \bslack\b alone (no clear verb) -> "slack_search" (read > write
           as the safer default for ambiguous Slack questions)
      3. Email
      4. Generic keyword fallback
      5. "tool" if nothing more specific
    """
    q = question.lower()

    if WEATHER_KEYWORDS.search(q):
        return "weather"

    slack_named = bool(SLACK_EXPLICIT.search(q))
    email_named = bool(EMAIL_EXPLICIT.search(q))

    if slack_named and not email_named:
        # Distinguish search vs posting for Slack.
        # 2026-08-22 STAGE 16+: previously always returned "slack", which
        # was correct for posting but wrong for searching.  Now:
        #   search-intent -> "slack_search"
        #   post-intent   -> "slack"
        #   ambiguous     -> "slack_search" (reading is the safer default)
        has_search = bool(SLACK_SEARCH_KEYWORDS.search(q))
        has_post = bool(SLACK_POST_KEYWORDS.search(q))
        if has_search and not has_post:
            return "slack_search"
        if has_post and not has_search:
            return "slack"
        # Both or neither: default to search (read-only is safer)
        return "slack_search"

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
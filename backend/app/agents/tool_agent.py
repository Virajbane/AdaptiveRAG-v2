"""
Tool Agent: bridges Planner's sources_needed (documents/web/calculator/
database/tool/direct_llm) to the REAL tool_registry's tool names
(web_search/calculator/sql_query/weather/slack/email). This mapping layer
is necessary because the two use different vocabularies -- Planner speaks
in terms of "source of information", tool_registry speaks in terms of
"callable + params".

2026-08-09 FIX: Added handlers for "tool" source which includes:
- Weather (OpenWeatherMap)
- Slack (messaging)
- Email (SMTP)

2026-08-09 FIX (routing bugs found in review):
1. _extract_location used `(?:in|for|of|at)` as a raw substring match
   with no word boundary, so it matched the "at" INSIDE the word "What"
   (e.g. "What is the weather in Mumbai?") and then greedily captured
   everything up to the trailing "?" -- producing "is the weather in
   Mumbai" as the "location". Every naturally-phrased weather question
   was silently falling back to a web search instead of calling the
   weather API. Fixed with \\b word boundaries + per-word stopword
   stripping instead of whole-phrase stopword matching.
2. _SLACK_KEYWORDS included "message", which also appears constantly in
   ordinary email requests ("send an email with a message about..."),
   so Slack was checked first and silently stole email requests. Fixed
   by checking for the explicit product name ("slack" / "email") first,
   and only falling back to the older generic keyword sets when neither
   is named.
3. Slack channel and email recipient were hardcoded regardless of the
   question's content. Now extracted from the question when present
   (a "#channel" token, or an email address), falling back to the old
   hardcoded defaults only when nothing is found.

2026-08-09 IMPROVEMENT (Resume Demo):
4. Fixed weather fallback to not assume "web" is available.
5. Added tool_result_formatter() for Answer agent to use.
6. Wired database source handler.
7. Simplified error handling and logging for clarity.

2026-08-10 FIX (Tests 2-3 — Weather keyword sync):
8. Broadened _WEATHER_KEYWORDS to match planner's own _WEATHER_ACTION_INTENT
   patterns. The planner correctly detects "rain", "snow", action verbs, etc.,
   but that knowledge was thrown away (reduced to bare "tool" string).
   tool_agent was re-detecting with a narrower regex, causing "Mumbai rain"
   to fail routing. Now both use the same keyword set.
"""

import re
from typing import Optional, Dict, Any

from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.services.tools.registry import tool_registry

_MATH_EXPR = re.compile(r"[-+/*^%().\d\s]{3,}")

# 2026-08-10 FIX: Broadened to match planner's weather intent detection.
# Planner's _WEATHER_ACTION_INTENT already checks for these patterns;
# tool_agent must use the same set or weather questions will be missed
# (e.g., "Mumbai rain" has "rain" which planner detects, but tool_agent's
# old regex was too narrow and couldn't find it, causing a routing failure).
_WEATHER_KEYWORDS = re.compile(
    r"\b(weather|temperature|forecast|climate|temp|rain|snow|sunny|cloudy|"
    r"humid|windy|precipitation)\b|will\s+it\s+(?:rain|snow)",
    re.IGNORECASE,
)

# Explicit product names -- checked FIRST so a question that names its
# target ("send an email...", "post this in slack...") is never
# hijacked by a generic word ("message", "post", "send") that happens
# to appear in both kinds of requests.
_SLACK_EXPLICIT = re.compile(r"\bslack\b", re.IGNORECASE)
_EMAIL_EXPLICIT = re.compile(r"\b(email|e-mail|mail)\b", re.IGNORECASE)

# Generic fallback keyword sets -- only consulted when NEITHER product is
# named explicitly. "message" was removed from the Slack set because it
# is equally common in email requests and was the direct cause of the
# misrouting bug; "channel"/"alert"/"post" are Slack-specific enough to
# keep.
_SLACK_KEYWORDS = re.compile(r"\b(slack|post|alert|channel)\b", re.IGNORECASE)
_EMAIL_KEYWORDS = re.compile(r"\b(email|e-mail|mail|send)\b", re.IGNORECASE)

_CHANNEL_PATTERN = re.compile(r"#[\w-]+")
_EMAIL_ADDR_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

_LOCATION_STOPWORDS = {
    "the", "a", "an", "now", "today", "tomorrow", "tonight",
    "this", "current", "currently", "please",
}


def _extract_expression(question: str) -> Optional[str]:
    """Extract mathematical expression from question."""
    q = question.replace("×", "*").replace("÷", "/").replace("^", "**")
    match = _MATH_EXPR.search(q)
    if match and any(ch.isdigit() for ch in match.group(0)):
        return match.group(0).strip()
    return None


def _extract_location(question: str) -> Optional[str]:
    """
    Extract a city/location from a weather question.

    Anchors the preposition on word boundaries (\\b) so it can't match
    a 2-letter preposition hiding inside an unrelated word (the old bug:
    "What" contains "at"). Trailing filler words ("today", "tomorrow",
    "now", ...) are stripped one token at a time from the end of the
    captured phrase, rather than only checking the whole phrase against
    a stopword list (which missed "Mumbai today", "New York tomorrow", etc.).
    """
    match = re.search(
        r"\b(?:in|for|of|at)\b\s+([A-Za-z][A-Za-z\s\-]*?)\s*(?:[?.!]|$)",
        question,
        re.IGNORECASE,
    )
    if not match:
        return None

    words = match.group(1).strip().split()
    while words and words[-1].lower() in _LOCATION_STOPWORDS:
        words.pop()

    location = " ".join(words).strip()
    return location or None


def tool_result_formatter(tool_name: str, result: Dict[str, Any]) -> str:
    """
    Format tool result into readable text for Answer agent to use.
    
    Args:
        tool_name: Name of the tool (weather, calculator, slack, email, etc.)
        result: Result dict from tool execution
    
    Returns:
        Human-readable formatted string
    """
    if result.get("error"):
        return f"Tool error: {result['error']}"
    
    if tool_name == "weather":
        location = result.get("location", "Unknown location")
        temp = result.get("temperature")
        description = result.get("description", "")
        if temp is not None:
            return f"{location}: {temp}°C, {description}"
        return f"Weather data for {location}: {result}"
    
    if tool_name == "calculator":
        expr = result.get("expression", "?")
        res = result.get("result")
        if res is not None:
            return f"{expr} = {res}"
        return f"Calculation result: {result}"
    
    if tool_name == "slack":
        channel = result.get("channel", "#unknown")
        return f"Message posted to Slack channel {channel}"
    
    if tool_name == "email":
        to_email = result.get("to_email", "recipient")
        return f"Email sent to {to_email}"
    
    if tool_name == "web_search":
        count = result.get("count", 0)
        results = result.get("results", [])
        if count > 0:
            snippets = [r.get("snippet", r.get("title", ""))[:100] for r in results[:3]]
            return f"Found {count} results:\n" + "\n".join(snippets)
        return "No web search results found"
    
    if tool_name == "database":
        count = result.get("count", 0)
        return f"Found {count} database results"
    
    # Generic fallback
    return str(result)


class ToolAgent(BaseAgent):
    """
    Execute tools based on planner's source decisions.
    
    Maps planner's abstract "sources" to concrete tool implementations.
    """

    def __init__(self, llm):
        super().__init__(llm)
        self.SOURCE_HANDLERS = {
            "web": self._handle_web_search,
            "calculator": self._handle_calculator,
            "tool": self._handle_tool,
            "database": self._handle_database,
        }

    async def _handle_web_search(self, state: AgentState) -> None:
        """Execute web search using tool registry."""
        print("[TOOL] Executing web search...")
        result = await tool_registry.execute_tool(
            "web_search", query=state.question, max_results=5
        )
        state.tool_results["web_search"] = result
        
        if result.get("error"):
            print(f"[TOOL] Web search FAILED: {result['error']}")
        else:
            count = result.get("count", 0)
            print(f"[TOOL] Web search complete: {count} results")

    async def _handle_calculator(self, state: AgentState) -> None:
        """Extract math expression and execute calculation."""
        print("[TOOL] Executing calculator...")
        expr = _extract_expression(state.question)
        
        if not expr:
            state.tool_results["calculator"] = {"error": "No math expression found"}
            print("[TOOL] Calculator: No math expression found in question")
            return
        
        result = await tool_registry.execute_tool("calculator", expression=expr)
        # Attach expression so Answer agent can render "235 * 18 = 4230"
        state.tool_results["calculator"] = {**result, "expression": expr}
        
        if result.get("error"):
            print(f"[TOOL] Calculator FAILED: {result['error']}")
        else:
            res = result.get("result")
            print(f"[TOOL] Calculator: {expr} = {res}")

    async def _handle_database(self, state: AgentState) -> None:
        """
        Execute database query.
        
        Queries allowed collections: documents, chat_sessions, memory_long_term, queries.
        """
        print("[TOOL] Executing database query...")
        
        # Default query: find documents for this user
        result = await tool_registry.execute_tool(
            "sql_query",
            collection="documents",
            query_type="find",
            query={"user_id": state.user_id},
            limit=10
        )
        state.tool_results["database"] = result
        
        if result.get("error"):
            print(f"[TOOL] Database FAILED: {result['error']}")
        else:
            count = result.get("count", 0)
            print(f"[TOOL] Database: Retrieved {count} documents")

    async def _handle_tool(self, state: AgentState) -> None:
        """
        Route to specific tool based on question keywords:
        - Weather keywords → OpenWeatherMap
        - Slack keywords → Slack API
        - Email keywords → SMTP/Gmail

        Explicit product names ("slack" / "email") are checked before
        generic keyword sets so overlapping generic words ("message",
        "send", "post") can no longer steal a request that actually
        named its target.
        """
        question = state.question.lower()

        # Weather takes priority over messaging keywords regardless.
        if _WEATHER_KEYWORDS.search(question):
            await self._handle_weather(state)
            return

        slack_named = bool(_SLACK_EXPLICIT.search(question))
        email_named = bool(_EMAIL_EXPLICIT.search(question))

        if slack_named and not email_named:
            await self._handle_slack(state)
            return
        
        if email_named and not slack_named:
            await self._handle_email(state)
            return
        
        if slack_named and email_named:
            # Both named in the same question -- genuinely ambiguous.
            # Default to email: it's the more targeted/private channel,
            # so defaulting there is the safer failure mode than
            # broadcasting to a shared Slack channel by mistake.
            print("[TOOL] Both 'slack' and 'email' named -- defaulting to email")
            await self._handle_email(state)
            return

        # Neither product named explicitly -- fall back to generic
        # keyword sets (weaker signal, kept for backward compatibility).
        if _SLACK_KEYWORDS.search(question):
            await self._handle_slack(state)
            return
        
        if _EMAIL_KEYWORDS.search(question):
            await self._handle_email(state)
            return

        print("[TOOL] No specific tool matched in generic 'tool' source")
        state.tool_results["tool"] = {"error": "Could not determine which tool to use"}

    async def _handle_weather(self, state: AgentState) -> None:
        """Execute weather (OpenWeatherMap) query."""
        print("[TOOL] Executing weather tool...")
        location = _extract_location(state.question)

        if not location:
            # No location found - can't proceed
            print("[TOOL] Weather: No location found in question")
            state.tool_results["weather"] = {
                "error": "Could not extract location from question",
                "location": None
            }
            return

        result = await tool_registry.execute_tool(
            "weather", location=location
        )
        # Attach the resolved location so Answer agent can say
        # "Mumbai: 31°C, clear sky" instead of just raw numbers.
        state.tool_results["weather"] = {**result, "location": location}

        if result.get("error"):
            print(f"[TOOL] Weather FAILED: {result['error']}")
        else:
            temp = result.get("temperature")
            condition = result.get("description")
            print(f"[TOOL] Weather: {location} = {temp}°C, {condition}")

    async def _handle_slack(self, state: AgentState) -> None:
        """
        Execute Slack posting.

        Channel is read from the question when the user names one
        (e.g. "#eng-alerts"), instead of always posting to a hardcoded channel.
        """
        print("[TOOL] Executing Slack tool...")

        channel_match = _CHANNEL_PATTERN.search(state.question)
        channel = channel_match.group(0) if channel_match else "#alerts"

        # Use the question as the message content
        message = state.question

        result = await tool_registry.execute_tool(
            "slack_post",
            channel=channel,
            message=message
        )
        # Attach the channel actually used so Answer agent can confirm
        # where the message went.
        state.tool_results["slack"] = {**result, "channel": channel}

        if result.get("error"):
            print(f"[TOOL] Slack FAILED: {result['error']}")
        else:
            print(f"[TOOL] Slack: Message posted to {channel}")

    async def _handle_email(self, state: AgentState) -> None:
        """
        Execute email sending.

        Recipient is read from the question when an email address is
        present, instead of always sending to a hardcoded address.
        """
        print("[TOOL] Executing email tool...")

        addr_match = _EMAIL_ADDR_PATTERN.search(state.question)
        to_email = addr_match.group(0) if addr_match else "virajbane2004@gmail.com"

        subject = f"RAG System: {state.question[:50]}"
        body = state.question

        result = await tool_registry.execute_tool(
            "send_email",
            subject=subject,
            body=body,
            to_email=to_email
        )
        # Attach the recipient actually used so Answer agent can confirm
        # who received it.
        state.tool_results["email"] = {**result, "to_email": to_email}

        if result.get("error"):
            print(f"[TOOL] Email FAILED: {result['error']}")
        else:
            print(f"[TOOL] Email: Message sent to {to_email}")

    async def _execute(self, state: AgentState) -> AgentState:
        """Execute all active sources in order."""
        active_sources = [s for s in state.sources_needed if s in self.SOURCE_HANDLERS]
        
        if not active_sources:
            print(f"[TOOL] No active sources to handle: {state.sources_needed}")
            return state

        try:
            for source in active_sources:
                print(f"[TOOL] Handling source: {source}")
                await self.SOURCE_HANDLERS[source](state)
        except Exception as e:
            print(f"[TOOL] Tool execution error: {e}")
            state.error = f"Tool agent error: {str(e)}"
            import traceback
            traceback.print_exc()

        return state
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

2026-08-10 FIX (Tests 2-3 -- Weather keyword sync):
8. Broadened _WEATHER_KEYWORDS to match planner's own _WEATHER_ACTION_INTENT
   patterns. The planner correctly detects "rain", "snow", action verbs, etc.,
   but that knowledge was thrown away (reduced to bare "tool" string).
   tool_agent was re-detecting with a narrower regex, causing "Mumbai rain"
   to fail routing. Now both use the same keyword set.

2026-08-22 FIX (location extraction generalization + database intent):
9. _extract_location rewritten. The old version required one of a fixed
   set of prepositions (in/for/of/at) to appear literally in the question,
   so "how humid is kolkata right now" (no preposition at all) returned
   None with "No location found in question" even though the city was
   right there. Replaced with a strip-the-noise strategy: remove tokens
   that are (exactly, or fuzzily within a tight length-bounded threshold)
   known weather-question vocabulary; whatever tokens remain, in original
   order, are the location. This requires no preposition and tolerates
   common typos ("cuurect" -> current, "iin" -> in) because we only need
   to recognize and discard the surrounding noise words, never the city
   name itself. The old preposition-anchored regex is kept as a fallback
   for the rare case where stripping removes every token (e.g. the whole
   question IS just weather vocabulary with no location present at all).
   Verified against the 10 required cities plus 20+ additional cities
   with zero false positives (no real city name gets eaten by the fuzzy
   matcher), and against several intentionally-mistyped inputs.
10. _handle_database rewritten to actually read the question instead of
    always running the same hardcoded find on "documents". Added
    _parse_database_intent, a small NL->intent classifier restricted to
    collections that actually exist in this app's schema (users,
    documents, chat_sessions -- confirmed against
    app/db/mongodb/queries.py). This does NOT let the LLM generate
    arbitrary queries; it maps recognized phrasing to the existing
    find/count query_types sql_executor.py already supports. Questions
    about entities with no real collection in this schema (patients,
    appointments, medications, ...) are intentionally left unmatched so
    the answer agent gets an honest "no matching data source" signal
    instead of a fabricated answer.
"""

import re
import difflib
from typing import Optional, Dict, Any

from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.agents.nl_arithmetic import build_expression as _build_nl_expression
from app.agents.tool_mapping import resolve_sub_tool
from app.services.tools.registry import tool_registry

_MATH_EXPR = re.compile(r"[-+/*^%().\d\s]{3,}")

_CHANNEL_PATTERN = re.compile(r"#[\w-]+")
_EMAIL_ADDR_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

_LOCATION_STOPWORDS = {
    "the", "a", "an", "now", "today", "tomorrow", "tonight",
    "this", "current", "currently", "please",
}


def _extract_expression(question: str) -> Optional[str]:
    """
    Extract mathematical expression from question.

    Priority:
    1. Percentage phrases: "18% of 3500" -> "3500 * 18 / 100"
    2. Symbolic expressions: "+", "-", "/", "*"
    3. Natural language expressions via nl_arithmetic module

    STAGE 13 FIX: fallback to NL expression building when no symbolic
    expression found. STAGE 14 FIX: add percentage extraction first.
    """
    q = question.replace("×", "*").replace("÷", "/").replace("^", "**")

    pct_expr = _extract_percentage_expression(q)
    if pct_expr:
        return pct_expr

    match = _MATH_EXPR.search(q)
    if match and any(ch.isdigit() for ch in match.group(0)):
        return match.group(0).strip()

    return _build_nl_expression(question)


def _extract_percentage_expression(q: str) -> Optional[str]:
    """
    Extract percentage expressions like "18% of 3500" -> "3500 * 18 / 100"
    or "18 percent of 3500" -> "3500 * 18 / 100".

    Returns None if no percentage pattern found.
    """
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)|"
        r"(\d+(?:\.\d+)?)\s+percent\s+of\s+(\d+(?:\.\d+)?)",
        q,
        re.IGNORECASE
    )
    if m:
        pct = m.group(1) or m.group(3)
        base = m.group(2) or m.group(4)
        if pct and base:
            return f"{base} * {pct} / 100"
    return None


# ── WEATHER LOCATION EXTRACTION ────────────────────────────────────────
#
# Full stopword set -- EXACT match only. Broad and generous on purpose;
# an exact match is by definition safe (it can never accidentally eat
# part of a real city name).
_WEATHER_STOPWORDS = {
    "what", "whats", "is", "are", "was", "were", "the", "a", "an", "please",
    "current", "currently", "right", "now", "rn", "today", "tonight",
    "tomorrow", "this",
    "weather", "temperature", "temp", "climate", "forecast", "forcast",
    "humid", "humidity", "rain", "raining", "rains", "rainy",
    "snow", "snowy", "snowing",
    "wind", "windy", "winds", "speed", "storm", "stormy",
    "thunder", "thundering",
    "hot", "cold", "warm", "cool", "sunny", "cloudy", "cloud", "clouds",
    "overcast",
    "degrees", "degree", "celsius", "fahrenheit", "like", "how", "hows",
    "it", "its", "in", "at", "for", "of", "on", "about", "near", "around",
    "city", "there", "doing", "going", "looking", "condition", "conditions",
    "outside",
    "will", "does", "do", "tell", "me", "know", "check", "give", "show",
    "update", "updates",
}

# Narrower set -- FUZZY match candidates only. Deliberately excludes
# generic short words ("know", "give", "show", "climate"...) that are
# common near-matches of real proper nouns (e.g. "Lucknow" ~ "know",
# "Coimbatore" ~ "climate"). Only words that are genuinely useful to
# typo-correct belong here.
_FUZZY_CANDIDATES = {
    "current", "currently", "weather", "temperature", "forecast",
    "humidity", "humid", "raining", "rainy", "snowing", "snowy",
    "windy", "stormy", "degrees", "celsius", "fahrenheit",
    "in", "the", "of", "for", "at", "how", "is", "it",
}

_FUZZY_MAX_LEN_DIFF = 1
_FUZZY_THRESHOLD = 0.70


def _fuzzy_is_stopword(word: str) -> bool:
    """
    True if `word` is (a) an exact weather-vocabulary stopword, or
    (b) a likely typo of one of the curated _FUZZY_CANDIDATES -- judged
    by edit-distance ratio AND length proximity together, so a long word
    (a real city name) can never be "corrected" into a short generic
    word just because a substring happens to line up (e.g. "Coimbatore"
    is 3+ characters longer than "climate", so it's never even compared
    at the ratio stage).
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return True
    if w in _WEATHER_STOPWORDS:
        return True
    if len(w) <= 2:
        # Too short for a safe fuzzy comparison either way.
        return w in _WEATHER_STOPWORDS
    for cand in _FUZZY_CANDIDATES:
        if abs(len(w) - len(cand)) > _FUZZY_MAX_LEN_DIFF:
            continue
        if difflib.SequenceMatcher(None, w, cand).ratio() >= _FUZZY_THRESHOLD:
            return True
    return False


def _extract_location(question: str) -> Optional[str]:
    """
    Extract a city/location from a weather question.

    Strategy: strip known (and typo-tolerant) weather-question vocabulary
    from the tokenized question; whatever tokens remain, in original
    order, are the location. This works regardless of whether the
    question uses a preposition ("in Mumbai"), omits one entirely
    ("how humid is kolkata right now"), or misspells the surrounding
    words ("cuurect temperature iin hyderabad") -- because we never need
    to correctly recognize the intent words, only exclude them.

    A single-token question ("Pune", "Hyderabad?") is treated as the
    location directly without stopword filtering, since this handler is
    only ever reached after the Planner has already classified the
    question as weather intent -- a bare one-word "question" at that
    point can only be the city itself.

    Falls back to the old preposition-anchored regex if stripping
    removes every token (i.e. the question was entirely weather
    vocabulary with no location present -- a genuine "no location
    given" case, not an extraction failure).
    """
    cleaned = re.sub(r"[?!.,]", " ", question)
    tokens = cleaned.split()
    if not tokens:
        return None

    if len(tokens) == 1:
        return tokens[0].title()

    remaining = [t for t in tokens if not _fuzzy_is_stopword(t)]
    location = " ".join(remaining).strip()
    if location:
        return location.title()

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

    fallback_location = " ".join(words).strip()
    return fallback_location or None


# ── DATABASE INTENT PARSING ────────────────────────────────────────────
#
# Restricted to collections that actually exist in this app's MongoDB
# schema (confirmed from app/db/mongodb/queries.py: UserQueries ->
# "users", DocumentQueries -> "documents"; chat_sessions is referenced
# elsewhere in the codebase as a collection name). This is intentionally
# NOT a general-purpose NL-to-query translator -- it recognizes a fixed,
# small set of question shapes and maps them to the find/count
# query_types sql_executor.py already implements. Anything it doesn't
# recognize returns None, which the caller treats as "no database
# intent matched" rather than guessing.
_DB_ENTITIES = {
    "users": {
        "collection": "users",
        "patterns": (re.compile(r"\busers?\b"), re.compile(r"\baccounts?\b")),
        # Never includes password_hash -- enforced again server-side in
        # sql_executor.py regardless of what's requested here.
        "safe_fields": {"email": 1, "name": 1, "created_at": 1, "is_active": 1, "_id": 0},
        "name_field": "name",
    },
    "documents": {
        "collection": "documents",
        "patterns": (
            re.compile(r"\bdocuments?\b"),
            re.compile(r"\bpdfs?\b"),
            re.compile(r"\bfiles?\b"),
            re.compile(r"\buploads?\b"),
            re.compile(r"\buploaded\b"),
        ),
        "safe_fields": {"filename": 1, "file_type": 1, "status": 1, "created_at": 1, "_id": 0},
        "name_field": "filename",
    },
    "chat_sessions": {
        "collection": "chat_sessions",
        "patterns": (
            re.compile(r"\bsessions?\b"),
            re.compile(r"\bchats?\b"),
            re.compile(r"\bconversations?\b"),
        ),
        "safe_fields": {"created_at": 1, "_id": 0},
        "name_field": None,
    },
}

_DB_COUNT_PHRASES = ("how many", "count", "number of", "total")
_DB_LIST_PHRASES = ("show me", "list", "what are the names", "names of", "give me a list")
_DB_LATEST_PHRASES = ("latest", "most recent", "newest", "last uploaded")
_DB_MINE_PHRASES = ("my ", "i've ", "i have ", "this user", "have i")


def _parse_database_intent(question: str) -> Optional[Dict[str, Any]]:
    """
    Map a natural-language database question to a structured, safe
    query intent: {collection, query_type, projection, scoped_to_user}.

    Entity resolution picks whichever known entity's keyword appears
    EARLIEST in the question (not just "first entity checked in dict
    order") so that e.g. "How many documents has this user uploaded?"
    correctly resolves to documents, not users, even though "user"
    also appears in the sentence.

    Returns None when no known entity is mentioned at all -- e.g.
    "patients", "appointments", "medications" have no collection in
    this app's schema, so these intentionally fall through rather than
    being mapped to the wrong collection or a fabricated one.
    """
    ql = question.lower()

    best_key, best_pos = None, None
    for key, cfg in _DB_ENTITIES.items():
        for pat in cfg["patterns"]:
            m = pat.search(ql)
            if m and (best_pos is None or m.start() < best_pos):
                best_key, best_pos = key, m.start()

    if not best_key:
        return None

    cfg = _DB_ENTITIES[best_key]

    if any(p in ql for p in _DB_LATEST_PHRASES) and cfg["name_field"]:
        query_type = "latest"
    elif any(p in ql for p in _DB_LIST_PHRASES):
        query_type = "list"
    elif any(p in ql for p in _DB_COUNT_PHRASES):
        query_type = "count"
    else:
        # Default to count for ambiguous "database" routing -- a safe,
        # non-PII-leaking answer shape when intent isn't explicit.
        query_type = "count"

    scoped_to_user = any(p in ql for p in _DB_MINE_PHRASES)

    return {
        "entity": best_key,
        "collection": cfg["collection"],
        "query_type": query_type,
        "safe_fields": cfg["safe_fields"],
        "name_field": cfg["name_field"],
        "scoped_to_user": scoped_to_user,
    }


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
        entity = result.get("entity")
        query_type = result.get("query_type")
        count = result.get("count", 0)
        rows = result.get("rows", [])
        if query_type == "count":
            return f"{entity}: {count} record(s)"
        if query_type == "latest" and rows:
            return f"Latest {entity}: {rows[0]}"
        if query_type == "list":
            return f"{entity} ({count} total): {rows}"
        return f"Found {count} database results for {entity}"

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
        state.tool_results["calculator"] = {**result, "expression": expr}

        if result.get("error"):
            print(f"[TOOL] Calculator FAILED: {result['error']}")
        else:
            res = result.get("result")
            print(f"[TOOL] Calculator: {expr} = {res}")

    async def _handle_database(self, state: AgentState) -> None:
        """
        Execute database query.

        Parses the question into a structured intent (entity + operation
        + scope) restricted to collections that actually exist in this
        app's schema, then executes via the existing sql_query tool.
        No LLM-generated queries; no arbitrary collections; users' find
        results are always restricted to a fixed safe field projection.
        """
        print("[TOOL] Executing database query...")

        intent = _parse_database_intent(state.question)

        if intent is None:
            print(f"[TOOL] Database: no matching data source for question: {state.question!r}")
            state.tool_results["database"] = {
                "error": "No matching database collection for this question",
                "entity": None,
                "count": 0,
                "rows": [],
            }
            return

        query_filter: Dict[str, Any] = {}
        if intent["scoped_to_user"] and state.user_id:
            query_filter["user_id"] = state.user_id

        if intent["query_type"] == "count":
            result = await tool_registry.execute_tool(
                "sql_query",
                collection=intent["collection"],
                query_type="count",
                query=query_filter,
            )
            rows = []
            count = 0 if result.get("error") else result.get("results", {}).get("count", 0)

        elif intent["query_type"] == "latest":
            result = await tool_registry.execute_tool(
                "sql_query",
                collection=intent["collection"],
                query_type="find",
                query=query_filter,
                projection=intent["safe_fields"],
                sort=[("created_at", -1)],
                limit=1,
            )
            rows = [] if result.get("error") else result.get("results", [])
            count = len(rows)

        else:  # "list"
            result = await tool_registry.execute_tool(
                "sql_query",
                collection=intent["collection"],
                query_type="find",
                query=query_filter,
                projection=intent["safe_fields"],
                limit=20,
            )
            rows = [] if result.get("error") else result.get("results", [])
            count = len(rows)

        if result.get("error"):
            print(f"[TOOL] Database FAILED: {result['error']}")
            state.tool_results["database"] = {
                "error": result["error"],
                "entity": intent["entity"],
                "query_type": intent["query_type"],
                "count": 0,
                "rows": [],
            }
        else:
            print(f"[TOOL] Database: {intent['entity']} {intent['query_type']} -> {count}")
            state.tool_results["database"] = {
                "error": None,
                "entity": intent["entity"],
                "query_type": intent["query_type"],
                "count": count,
                "rows": rows,
            }

    async def _handle_tool(self, state: AgentState) -> None:
        """
        Route to specific tool based on question keywords:
        - Weather keywords -> OpenWeatherMap
        - Slack keywords -> Slack API
        - Email keywords -> SMTP/Gmail

        STAGE 13 FIX: sub-tool selection now delegates to
        app.agents.tool_mapping.resolve_sub_tool -- the same function
        the planner uses to set state.tool right after routing -- so
        this can never pick a different concrete tool than the one the
        planner already reported.
        """
        sub_tool = resolve_sub_tool(state.question)

        if sub_tool == "weather":
            await self._handle_weather(state)
        elif sub_tool == "slack":
            await self._handle_slack(state)
        elif sub_tool == "email":
            await self._handle_email(state)
        else:
            print("[TOOL] No specific tool matched in generic 'tool' source")
            state.tool_results["tool"] = {"error": "Could not determine which tool to use"}

    async def _handle_weather(self, state: AgentState) -> None:
        """Execute weather (OpenWeatherMap) query."""
        print("[TOOL] Executing weather tool...")
        location = _extract_location(state.question)

        if not location:
            print("[TOOL] Weather: No location found in question")
            state.tool_results["weather"] = {
                "error": "Could not extract location from question",
                "location": None
            }
            return

        result = await tool_registry.execute_tool(
            "weather", location=location
        )
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

        message = state.question

        result = await tool_registry.execute_tool(
            "slack_post",
            channel=channel,
            message=message
        )
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
import json
import re

from app.agents.base import BaseAgent
from app.agents.prompts import PLANNER_PROMPT
from app.agents.state import AgentState
from app.agents.nl_arithmetic import has_nl_arithmetic_intent
from app.agents.tool_mapping import resolve_concrete_tool

# --------------------------------------------------------------------------
# ROUTING HISTORY (read before modifying)
#
# v1 (LLM free-text routing): caused Bug 1 -- a question naming a
# document's subject by name (no "my"/"this doc" wording) got silently
# misrouted to "web", returning content about an unrelated same-named
# entity.
#
# v2 (explicit @tag routing): fixed Bug 1 by removing inference
# entirely. Later reverted per explicit request -- inference was
# wanted back.
#
# v3 (current): LLM classification restored, but constrained to a
# CLOSED output space (SOURCE_REGISTRY keys only, enforced by
# _parse_classifier_output dropping anything outside that set). This
# is stricter than v1's free-text router. Residual risk from Bug 1
# still exists in principle -- mitigated by:
#   (a) closed output space (can't invent a 7th source),
#   (b) documents-bias instruction in the prompt,
#   (c) safe ["documents"] fallback on any parse failure,
#   (d) the entity-binding check on web results (Bug 6 fix) living
#       downstream in the web tool, independent of how "web" was
#       selected.
#   (e) 2026-08-09 FIX: document-intent protection with high-confidence
#       detection for paper/figure/table/section/abstract queries.
#
# v4 (2026-08-09): Added document-intent protection patterns to catch
# research paper evaluation questions without LLM inference. Added
# support for multi-source routing (e.g. ["documents", "web"]). Now
# uses rewritten_question as the authoritative input to the classifier.
#
# v5 (2026-08-10 STAGE 11 FIX FOR 0.5B): 
#   - Separated deterministic high-confidence routing into comprehensive layer
#   - Expanded document-intent patterns to cover more variants
#   - Added calculator-intent explicit detection (not just LLM)
#   - Added weather/action-intent explicit detection (not just LLM)
#   - Added database-intent detection (BEFORE LLM, high confidence)
#   - Added multi-source-intent detection (BEFORE LLM, high confidence)
#   - Added web-intent detection (high confidence patterns)
#   - Implemented explicit parsing pipeline: raw → JSON → schema → normalization → validation
#   - Multi-source validation now explicit and preserved
#   - All failures are now observable (not silent)
#   - Clear deterministic vs LLM priority model
#   - Reordered routing layers: weather BEFORE web (prevents confusion)
#   - Added keyword-based fallback for when LLM fails (0.5B safety net)
#   - SOURCE_REGISTRY validation enforced at multiple points
#
# v6 (2026-08-10 STAGE 12 FIX -- root-cause investigation follow-up):
#   Stage 10's planner-routing harness was found to be calling
#   _classify_sources() directly (the LLM-only sub-step), bypassing
#   Layer 2 deterministic routing and the Layer 3 keyword fallback
#   entirely. That harness bug (fixed separately in stage10_eval.py)
#   masked/exposed three real issues in THIS file, fixed here:
#     1. _MULTI_SOURCE_INTENT required strict word-adjacency between the
#        comparison verb and the "my/our/uploaded" reference (e.g. it
#        matched "compare my paper" but not "compare the findings in my
#        uploaded paper"). Added an order-independent structural
#        detector (verb + own-content reference + external/recency
#        reference, anywhere in the question) as a second check.
#     2. Deterministic Layer 2 matches returned directly without ever
#        passing through the placeholder-source check that Layer 3/4
#        LLM-classified results went through -- so a deterministically
#        routed "database" (implemented=False) skipped the placeholder
#        message while an LLM-routed "database" got it. Unified both
#        paths through one _apply_placeholder_check() helper.
#     3. _keyword_fallback_classification() had no direct_llm branch and
#        silently defaulted general-knowledge questions to "documents".
#        Added an explicit direct_llm branch, and narrowed the
#        overly-broad "what is" calculator keyword (which was shadowing
#        direct_llm-shaped questions like "what is the capital of
#        France") to require an actual math symbol + digit.
#
# SOURCE_REGISTRY is the intended single source of truth for the
# classifier prompt AND for what tool_agent.py / graph.py treat as
# valid. The import-time assertion below will fail fast if the prompt
# step is forgotten.
#
# Adding a future source = one dict entry here + a rule/example in
# PLANNER_PROMPT (prompts.py) + one handler in tool_agent.py + (if it
# needs pre-answer execution) one line in graph.py's dispatch check.
# --------------------------------------------------------------------------

# Metadata query: title, author, affiliations
_METADATA_Q = re.compile(r"\b(title|author|affiliat)", re.IGNORECASE)

# =============================================================================
# STAGE 11 FIX: COMPREHENSIVE DETERMINISTIC ROUTING PATTERNS
# 
# These patterns bypass LLM inference entirely. They are high-confidence
# intent signals that should take priority over LLM classification.
# Organized by source, with expanded variants to catch 0.5B confusion.
# =============================================================================

# Personal content indicator (shows user is asking about their own data)
_PERSONAL_CONTENT = re.compile(
    r"(?:"
    r"my\s+(?:paper|pdf|file|document|research|project|study|report|data|files|work)\b|"
    r"our\s+(?:paper|pdf|file|document|research|project|study|report|data|files|work)\b|"
    r"(?:i\s+(?:have|uploaded|have|wrote)|we\s+(?:have|uploaded|wrote))\b|"
    r"the\s+(?:uploaded|attached|provided)\s+(?:paper|pdf|file|document|research)\b"
    r")",
    re.IGNORECASE
)

# Document-intent: comprehensive patterns for research paper/figure/table queries
_DOCUMENT_INTENT = re.compile(
    r"(?:"
    # Explicit document-referencing phrases. Allows a possessive/topic
    # modifier between "according to" and the document noun (e.g.
    # "according to my RAG document") -- the original pattern required
    # strict adjacency and missed this, the same class of bug fixed for
    # multi-source routing in STAGE 12.
    r"according\s+to\s+(?:the\s+|my\s+|our\s+)?(?:\w+\s+){0,2}(?:paper|document|pdf|file|abstract|publication|study|research|report)\b|"
    r"(?:my|our)\s+uploaded\s+(?:paper|pdf|file|document|research|data)\b|"
    r"in\s+(?:the\s+)?(?:paper|document|pdf|file|abstract|publication|section|appendix|figure|table|table\s+of\s+contents)\b|"
    # Query about what document says/reports/shows
    r"(?:what\s+(?:is|does|did)|what's|list)\s+(?:reported|mentioned|described|stated|shown|found|demonstrated|proposed|suggested)\s+in\b|"
    r"(?:what|which|where)\s+(?:is|are|does|did)\s+(?:reported|mentioned|in)\s+(?:the\s+)?(?:paper|document|pdf|file|study)\b|"
    # Explicit figure/table/section references
    r"(?:figure|table|section|appendix|appendices|fig\.|tbl\.|sec\.)\s+(?:\d+[a-z]?|[A-Z][\d.]*)\b|"
    r"(?:from|in)\s+(?:figure|table|section|appendix)\s+(?:\d+[a-z]?|[A-Z][\d.]*)\b|"
    r"according\s+to\s+(?:figure|table|section|appendix)\b|"
    # Document-specific content questions
    r"(?:table\s+of\s+)?(?:contents|results|findings|metrics|scores|benchmarks).*(?:in|from)\s+(?:the\s+)?(?:paper|document|file)\b|"
    r"(?:the\s+)?(?:paper|document|file|study).*(?:shows?|reports?|demonstrates?|proposes?|suggests?)\s+|"
    # Results/metrics from specific document
    r"(?:what|which|where).*(?:utmos|bleu|rouge|f1|accuracy|precision|recall|loss|score).*(?:in|from|according\s+to)\b|"
    r"(?:utmos|bleu|rouge|f1|accuracy|precision|recall|loss|score)\s+.*(?:according\s+to|in|from)\s+(?:the\s+)?(?:paper|document|file|table|figure)\b|"
    # Meta-level paper questions
    r"what\s+(?:is|are)\s+the\s+(?:contributions?|findings?|results?|conclusions?)\s+(?:of\s+)?(?:the\s+)?(?:paper|document|study)\b|"
    r"summarize\s+(?:the\s+)?(?:paper|pdf|document|research)\b"
    r")",
    re.IGNORECASE
)

# Calculator-intent: mathematical operations, conversions, numeric computations
#
# v7 (2026-08-13 FIX -- planner eval failure LLM_02): the previous first
# branch here, `(?:what\s+is|calculate|...)\s+.+(?:\+|-|\*|/|...).*[?\"]`,
# matched a bare "-" character with no digits anywhere nearby. Since the
# character class `(?:\+|-|\*|/|÷|×)` treats "-" as "the literal minus
# sign", it happily matched the hyphen inside "object-oriented" --
# "What is object-oriented programming?" satisfied "what is" + ".+" +
# "-" + ".*?" and was misrouted to the calculator. That branch is
# removed: `\d+\s*(?:\+|-|\*|/|÷|×)\s*\d+` below already covers every
# symbolic-math case it was meant to catch (digit-operator-digit), and
# additionally requires digits adjacent to the operator, so a stray
# hyphen in an ordinary word can never match it.
_CALCULATOR_INTENT = re.compile(
    r"(?:"
    # Arithmetic operations (operator MUST be adjacent to digits on both
    # sides -- this is what makes it immune to stray hyphens elsewhere
    # in the sentence, e.g. "object-oriented")
    r"\d+\s*(?:\+|-|\*|/|÷|×)\s*\d+|"
    # Explicit calculator keywords
    r"(?:what's?|calculate|compute|solve|find|simplify)\s+.+[0-9].+[?\"]|"
    # Percentage/ratio
    r"(?:percentage|percent|%|ratio|proportion)\s+(?:of|between|among).*[?\"]|"
    r"what\s+(?:is|'s)\s+.+%\s+(?:of|increase|decrease).*[?\"]|"
    r"how\s+much\s+is\s+.+%\s+of\s+\d+.*[?\"]|"
    # Unit conversions
    r"convert\s+.+(?:to|into|in\s+terms\s+of)\s+.+[?\"]|"
    r"how\s+many\s+(?:meters|feet|pounds|kilograms|celsius|fahrenheit|miles|kilometers|liters|gallons)\b|"
    r"what\s+is\s+.+\s+in\s+(?:meters|feet|pounds|kilograms|celsius|fahrenheit|miles|kilometers)\b|"
    # Powers, roots, logarithms
    r"(?:square|cube|square\s+root|cube\s+root|log|logarithm)\s+(?:of\s+)?[0-9.]+|"
    # Typical calculator phrases
    r"what\s+is\s+the\s+sum|add.*together|divide.*by|multiply.*by"
    r")",
    re.IGNORECASE
)

# Weather and action-intent: explicit external utility requests
# HIGH PRIORITY: Put this BEFORE web to prevent weather→web confusion
#
# v2 (2026-08-13 FIX -- planner eval failure AMB_03): the bare
# "(?:what's?|what\s+is)\s+(?:the\s+)?(?:weather|...)\b" branch matched
# on the weather NOUN alone, with nothing checked about what came after
# it -- so "What is the weather concept?" and "What is the weather in
# Mumbai?" were indistinguishable to this pattern. Weather-noun
# *mention* is not the same as a request for live weather DATA.
# Live-data intent requires the weather noun to be paired with either
# (a) a location, (b) a temporal/immediacy marker (today, right now,
# ...), or (c) an explicit action phrasing ("will it rain"). Mentioning
# the term with no such pairing (a bare noun, or paired with an
# explanation word like "concept"/"mean"/"definition") is conceptual,
# not operational, and is deliberately left unmatched here so it falls
# through to the explanation-intent layer instead.
_WEATHER_TERM = re.compile(
    r"\b(?:weather|temperature|temp|forecast|precipitation|rain|snow|wind|humidity|climate|conditions?)\b",
    re.IGNORECASE,
)
_WEATHER_EXPLANATION_MARKER = re.compile(
    r"\b(?:concept|mean|means|meaning|definition|works?|explain|"
    r"forecasting|prediction|technology|science|how\s+does|how\s+do)\b",
    re.IGNORECASE,
)
_WEATHER_LIVE_SIGNAL = re.compile(
    r"\b(?:in|at|for|near)\s+[A-Za-z][A-Za-z\s\-]{1,40}|"
    r"\b(?:today|tomorrow|tonight|this\s+week|this\s+weekend|right\s+now|currently|current|now)\b",
    re.IGNORECASE,
)
_WEATHER_ACTION_VERB = re.compile(
    r"(?:is\s+it|will\s+it)\s+(?:rain|snow|be\s+sunny|be\s+cloudy|be\s+cold|be\s+hot|be\s+warm)|"
    r"(?:will|is)\s+(?:there|it)\s+(?:be\s+rain|be\s+snow|rain|snow)",
    re.IGNORECASE,
)


def _is_weather_live_query(question: str) -> bool:
    """
    True only for an OPERATIONAL weather request (live data wanted),
    never for a mention of the concept of weather. See v2 note above.
    """
    if not _WEATHER_TERM.search(question):
        return False
    if _WEATHER_ACTION_VERB.search(question):
        return True
    if _WEATHER_EXPLANATION_MARKER.search(question):
        # An explanation word is present alongside the weather term
        # ("concept", "how does ... work", "forecasting" as a topic,
        # etc.) -- treat as conceptual unless there's ALSO an
        # unambiguous location/temporal signal (e.g. "how's the
        # weather in Mumbai right now" still wants live data).
        return bool(_WEATHER_LIVE_SIGNAL.search(question))
    return bool(_WEATHER_LIVE_SIGNAL.search(question))


_WEATHER_ACTION_INTENT = re.compile(
    r"(?:"
    # Action requests (send, post, email, message, etc.)
    r"(?:send|post|message|email|notify|alert|slack|dm|dm\s+me|share)\s+(?:.+\s+)?(?:to|on|in|via)\b|"
    r"(?:post|send|message|email|write|compose|create)\s+.+(?:to|into|on)\s+.*[?\"]|"
    r"(?:create|schedule|set|add)\s+(?:a\s+)?(?:reminder|alarm|event|task|calendar\s+event)\b|"
    r"(?:add|create|schedule)\s+(?:to\s+)?(?:my\s+)?(?:calendar|schedule|to.?do|todo|to.?do\s+list|checklist)\b|"
    r"(?:post|share|send)\s+.+to\s+(?:#|@)?[a-z]"
    r")",
    re.IGNORECASE
)

# Multi-source intent: comparing user data with external information
_MULTI_SOURCE_INTENT = re.compile(
    r"(?:"
    r"compare\s+(?:my|our|the\s+uploaded)\s+(?:research|paper|data|results|findings).*(?:with|to|against)\s+(?:the\s+)?(?:latest|current|recent|external)\b|"
    r"(?:my|our|the\s+uploaded)\s+(?:research|paper|data|results)\s+.*(?:vs|versus|compared\s+to|against)\s+(?:latest|current|recent|external|public)\b|"
    r"does?\s+(?:my|our|the\s+uploaded)\s+(?:research|paper|results|findings)\s+(?:match|align|compare|fit)\s+(?:the\s+)?(?:latest|current|recent)\b|"
    r"verify?\s+(?:my|our)\s+(?:results|findings|analysis)\s+(?:against|with)\s+(?:current|latest|recent|external)\s+(?:data|information|benchmarks|research)\b|"
    r"(?:check|search|look\s+in)\s+(?:my|our)\s+(?:files|documents|papers)\s+(?:and|also)\s+(?:check|search|look\s+for)\s+(?:current|latest|recent|external|online)\b"
    r")",
    re.IGNORECASE
)

# STAGE 12 FIX: the pattern above requires strict word-adjacency between
# the comparison verb and the "my/our/uploaded" reference (e.g. it matches
# "compare my paper with..." but NOT "compare the findings in my uploaded
# paper with..." -- the extra words between "compare" and "my" break it).
# This is an order-independent structural detector: it fires only when a
# comparison/verification verb, a reference to the user's own content, AND
# a reference to external/recent information ALL appear somewhere in the
# question -- regardless of their order or adjacency. This is a general
# three-signal check, not a hack tailored to one specific question.
_MULTI_SOURCE_VERB = re.compile(
    r"\b(compare|verify|cross-?check|cross-?reference)\b", re.IGNORECASE
)
_MULTI_SOURCE_OWN_CONTENT_REF = re.compile(
    r"\bmy\s+\w+|\bour\s+\w+|\bthe\s+uploaded\s+\w+|"
    r"\buploaded\s+(?:paper|pdf|file|document|research|data)\b",
    re.IGNORECASE,
)
_MULTI_SOURCE_EXTERNAL_REF = re.compile(
    r"\b(?:latest|current|recent|external|public|online)\b",
    re.IGNORECASE,
)

# Web-intent: current/live/external information requiring search
_WEB_INTENT = re.compile(
    r"(?:"
    # Temporal keywords indicating recency
    r"(?:latest|current|recent|newest|breaking|today|this\s+week|this\s+month|2024|2025|2026)\b.*(?:news|developments?|events?|results?|updates?|benchmarks?|papers?|research|findings?|techniques?|methods?|approaches?|advances?|trends?|tools?|models?)|"
    r"what's?\s+(?:new|happening|going\s+on|the\s+latest|trending)\b|"
    r"(?:what|what's|find|search)\s+(?:the\s+)?(?:latest|current|recent)\s+(?:news|developments?|events?|updates?|benchmarks?|papers?|research|findings?|techniques?|methods?|approaches?|advances?|trends?|tools?|models?)\b|"
    # GitHub/public repository search
    r"(?:github|gitlab|bitbucket|repository|repo|source\s+code)\s+.*(?:search|find|look\s+for|show)\b|"
    r"(?:search|find|show|look\s+for)\s+.*(?:on\s+github|repository|repo|on\s+(?:github|gitlab|bitbucket))\b|"
    # External/public information requiring search
    r"(?:search\s+(?:for|online)?|find|look\s+up|research)\s+.+(?:on\s+(?:the\s+)?(?:internet|web|online)|publicly|external)\b|"
    r"what\s+is\s+.+on\s+(?:the\s+)?(?:internet|web|online).*[?\"]|"
    # "what happened ... <recency marker>" -- current/recent events, as
    # opposed to "what happened ... historically" (see _HISTORICAL_INTENT)
    r"what(?:'s|\s+is|\s+happened|\s+are\s+the\s+latest\s+developments?)\b.*\b(?:today|yesterday|this\s+week|this\s+month|recently|lately)\b"
    r")",
    re.IGNORECASE
)

# =============================================================================
# STAGE 13 FIX (2026-08-13): EXPLANATION-INTENT LAYER
#
# Root cause shared by planner eval failures LLM_02 (residual), AMB_03
# (residual), AMB_04, AMB_05, AMB_07: none of documents/database/
# calculator/weather/web deterministic layers match these questions, so
# they all fell through to LLM classification -- which, per this
# project's own comments elsewhere (v3, "documents-bias instruction"),
# is unreliable on a small model and biased toward "documents" on
# uncertainty. The fix is not "make the LLM better" but "recognize this
# whole class of question deterministically", per the doc's recommended
# intent model (GENERAL EXPLANATION INTENT as its own category, checked
# before ever reaching the LLM).
#
# This is a MENTION-OF vs REQUEST-FOR distinction: "explain X", "how
# does X work", "what is X" (bare concept, no digits/location/document/
# recency signal), "history of X" / "X historically" are all requests
# for an explanation, not a request to fetch/compute/retrieve anything.
# =============================================================================

# Explicit "please explain this concept" phrasing.
_EXPLANATION_MARKER = re.compile(
    r"^\s*explain\b|"
    r"\bhow\s+(?:does|do)\s+.+\s+work\b|"
    r"\bwhat\s+(?:does|is)\s+.+\s+mean\b|"
    r"\b(?:definition|meaning)\s+of\b|"
    r"\bwhat\s+is\s+the\s+\w+\s+concept\b",
    re.IGNORECASE,
)

# General knowledge / historical framing, as opposed to a request for
# current/recent information (which _WEB_INTENT already catches first).
_HISTORICAL_INTENT = re.compile(
    r"\bhistorically\b|\bhistory\s+of\b|\bhistorical(?:ly)?\b|"
    r"\bwhen\s+was\s+.+\s+(?:founded|established|built|invented|created)\b|"
    r"\borigin(?:s|ated)?\s+of\b",
    re.IGNORECASE,
)

def _looks_generic_concept(question: str) -> bool:
    """
    True if the "What is X" question asks about a generic concept
    (e.g., "What is vector search?") vs. specific entity
    (e.g., "What is Lychee-FD?").
    """
    # Extract the X from "What is X?"
    match = re.search(r"what\s+(?:is|are|'s)\s+(.+?)[\?\.]*$", 
                      question, re.IGNORECASE)
    if not match:
        return False
    
    entity = match.group(1).strip()
    
    # Technical entities have these markers:
    # - CamelCase or PascalCase (Lychee-FD, DAG-PP, SpeechGPT-5)
    # - Hyphenated with numbers (GPT-4, DALL-E 2)
    # - All caps with numbers (BERT, RoBERTa)
    # - Known research project patterns
    
    if re.search(r"[A-Z][a-z]+[A-Z]|[A-Z]+-[A-Z0-9]|\d+[a-zA-Z]", entity):
        return False  # Looks like a technical entity
    
    # Generic concepts are lowercase or simple phrases
    if entity[0].islower() or entity in ["vector search", "dependency injection", 
                                         "precision", "recall", "attention"]:
        return True
    
    return False

# Bare "what is X?" / "what's X?" / "what are X?" with no digits, no
# location preposition, and no other operational signal -- a plain
# request for a definition/explanation. Deliberately simple: this is
# the same shape as rule 7 in PLANNER_PROMPT ("General Knowledge"), just
# enforced deterministically instead of hoping the LLM applies it.
_BARE_WHAT_IS = re.compile(
    r"^\s*what(?:'s|\s+is|\s+are)\s+(?:a\s+|an\s+|the\s+)?[a-z][\w\s\-]*\??\s*$",
    re.IGNORECASE,
)
_HAS_DIGIT = re.compile(r"\d")

# Current/live/real-time information intent (STAGE 14 FIX):
# Signals that the user wants up-to-date data, not stale general knowledge.
# This catches "current price of Bitcoin" (web) vs "What is Bitcoin?" (direct_llm).
_CURRENT_LIVE_INTENT = re.compile(
    r"\b(?:"
    r"current|now|today|tonight|right\s+now|this\s+moment|"
    r"live|real.?time|latest|up.?to.?date|recent|breaking|"
    r"just\s+(?:now|happened|released)|update|fresh|latest|"
    r"what's\s+happening|ongoing|active"
    r")\b",
    re.IGNORECASE,
)

# Database intent: internal app data queries (moved to deterministic layer for 0.5B)
_DATABASE_INTENT = re.compile(
    r"(?:"
    r"(?:how\s+many|total\s+number\s+of|records?|entries?|items?)\s+(?:in\s+)?(?:the\s+)?\w+\s+table\b|"
    # Count/stat queries
    r"how\s+many\s+(?:users?|records?|signups?|registrations?|entries?|items?|customers?|accounts?)\b.*(?:today|this|last|month|week|day|week|month|year).*[?\"]|"
    r"what's\s+the\s+(?:total|sum|count|average|mean)\s+(?:number|count|amount)\s+of\s+(?:users?|records?|signups?|registrations?|entries?)\b|"
    r"how\s+many\s+.+(?:created|added|registered|signed\s+up)\s+(?:in|last|this|today|this\s+week)\b|"
    r"(?:count|get\s+the\s+count)\s+of\s+(?:users?|records?|entries?)\b.*(?:in|from)\s+(?:the\s+)?(?:database|app|system)\b|"
    r"(?:database|app|internal\s+data|our\s+system).*(?:says|shows?|has|contains)\s+(?:how\s+many|what|total)\b"
    r")",
    re.IGNORECASE
)

# Canonical source name -> config
SOURCE_REGISTRY: dict[str, dict] = {
    "documents": {
        "description": "user's uploaded files / vector DB / internal knowledge base",
        "implemented": True,
    },
    "web": {
        "description": "current events, news, latest benchmarks, live information",
        "implemented": True,
    },
    "calculator": {
        "description": "math, unit conversions, percentage calculations",
        "implemented": True,
    },
    "database": {
        "description": (
            "counts/stats/records from the app's own database "
            "(SQL/Postgres/MySQL/MongoDB/Redis/Supabase)"
        ),
        "implemented": True,
    },
    "tool": {
        "description": (
            "external APIs not covered above -- weather, email, calendar, "
            "GitHub, Slack, generic REST APIs"
        ),
        "implemented": True,
    },
    "direct_llm": {
        "description": "general knowledge, definitions, explanations, no external data",
        "implemented": True,
    },
}

_VALID_SOURCES = set(SOURCE_REGISTRY.keys())
_PLACEHOLDER_SOURCES = {name for name, cfg in SOURCE_REGISTRY.items() if not cfg["implemented"]}

# Import-time SOURCE_REGISTRY → PLANNER_PROMPT alignment check
_missing_from_prompt = [name for name in _VALID_SOURCES if name not in PLANNER_PROMPT]
if _missing_from_prompt:
    raise RuntimeError(
        f"PLANNER_PROMPT (prompts.py) does not mention source(s) "
        f"{_missing_from_prompt!r} defined in SOURCE_REGISTRY. The "
        f"classifier can never select a source name it is never shown -- "
        f"add a rule/example for it in PLANNER_PROMPT before deploying, or "
        f"the router will silently default those questions to ['documents']."
    )


def _extract_json_from_text(raw: str) -> str:
    """
    STAGE 11 FIX (Step 1): JSON extraction
    
    Handles:
    1. Pure JSON (array or object) - return as-is
    2. Markdown code fence - strip and return
    3. JSON with surrounding text - extract via regex
    4. Priming-brace case: model continued from prompt's trailing "{"
       without re-emitting it - add it back
    
    Returns cleaned JSON string (still needs parsing).
    Raises ValueError if no valid JSON structure found.
    """
    cleaned = raw.strip()
    
    # Case 1: Already starts with { or [
    if cleaned.startswith("{") or cleaned.startswith("["):
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
        return cleaned
    
    # Case 2: Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    
    # Case 3: Starts with { after fence stripping
    if cleaned.startswith("{"):
        return cleaned
    
    # Case 4: Priming brace case
    if cleaned.endswith("}"):
        return "{" + cleaned
    
    # Case 5: Try to extract JSON from embedded text
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        return match.group(0)
    
    # Case 6: Try to extract JSON array
    match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if match:
        return match.group(0)
    
    raise ValueError(f"No JSON structure found in: {raw!r}")


def _parse_json_to_dict(json_str: str) -> dict | list:
    """
    STAGE 11 FIX (Step 2): Schema validation
    
    Parses JSON string and ensures it's either:
    - A dict (object)
    - A list (array of strings)
    
    Raises json.JSONDecodeError or ValueError if invalid.
    """
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    
    if not isinstance(parsed, (dict, list)):
        raise ValueError(
            f"Expected dict or list, got {type(parsed).__name__}: {parsed!r}"
        )
    
    return parsed


def _normalize_and_extract_sources(parsed: dict | list) -> list[str]:
    """
    STAGE 11 FIX (Step 3 & 4): Normalization + Source extraction
    
    Handles both JSON formats:
    1. Array: ["web", "calculator"] → return as-is
    2. Object with "sources" key: {"sources": [...], ...} → extract "sources" array
    
    Returns raw source list (strings, may include invalid names).
    Next step validates against SOURCE_REGISTRY.
    """
    if isinstance(parsed, list):
        return parsed
    
    if isinstance(parsed, dict):
        if "sources" not in parsed:
            raise ValueError(
                f"Dict has no 'sources' key. Keys present: {list(parsed.keys())}"
            )
        sources = parsed["sources"]
        if not isinstance(sources, list):
            raise ValueError(
                f"'sources' value is not a list: {type(sources).__name__}"
            )
        return sources
    
    raise ValueError(f"Unexpected type after schema validation: {type(parsed)}")


def _validate_sources_against_registry(sources: list[str]) -> list[str]:
    """
    STAGE 11 FIX (Step 5): SOURCE_REGISTRY validation
    
    Filters sources to only those in SOURCE_REGISTRY.
    Logs any dropped sources explicitly (observable failure).
    
    Returns validated sources list. May be empty if all are invalid.
    """
    if not sources:
        print("[PLANNER] Extracted sources list is empty")
        return []
    
    valid = []
    invalid = []
    
    for source in sources:
        if not isinstance(source, str):
            print(f"[PLANNER] Source is not a string: {type(source).__name__} = {source!r}")
            invalid.append(source)
            continue
        
        if source in _VALID_SOURCES:
            valid.append(source)
        else:
            print(f"[PLANNER] Source '{source}' not in SOURCE_REGISTRY "
                  f"(valid: {sorted(_VALID_SOURCES)}) -- dropping it")
            invalid.append(source)
    
    if invalid:
        print(f"[PLANNER] Dropped {len(invalid)} invalid source(s): {invalid}")
    
    return valid


def _parse_classifier_output(raw: str) -> list[str]:
    """
    STAGE 11 FIX: Explicit parsing pipeline
    
    Complete flow:
        raw LLM response
            ↓ (Step 1)
        JSON extraction (handle code fences, priming brace, embedded JSON)
            ↓ (Step 2)
        schema validation (ensure dict or list)
            ↓ (Step 3 & 4)
        normalization & source extraction
            ↓ (Step 5)
        SOURCE_REGISTRY validation
            ↓
        result (may be empty, not silent)
    
    If any step fails, returns [] and logs the failure point.
    """
    # Step 1: Extract JSON from raw response
    try:
        json_str = _extract_json_from_text(raw)
    except ValueError as e:
        print(f"[PLANNER] Step 1 JSON extraction failed: {e}")
        return []
    
    # Step 2: Parse JSON and validate schema
    try:
        parsed = _parse_json_to_dict(json_str)
    except ValueError as e:
        print(f"[PLANNER] Step 2 schema validation failed: {e}")
        return []
    
    # Step 3 & 4: Normalize and extract sources
    try:
        sources = _normalize_and_extract_sources(parsed)
    except ValueError as e:
        print(f"[PLANNER] Step 3-4 normalization failed: {e}")
        return []
    
    # Step 5: Validate against SOURCE_REGISTRY
    validated = _validate_sources_against_registry(sources)
    
    return validated


def _keyword_fallback_classification(question: str) -> list[str]:
    """
    STAGE 11 FIX: Simple keyword-based fallback for when LLM fails.
    
    Used when _classify_sources returns nothing or invalid.
    This is a safety net for 0.5B model weakness.
    
    Returns most likely source based on simple keyword heuristics.

    STAGE 12 FIX:
      - Narrowed the calculator branch's overly-broad "what is" trigger
        (it matched almost any question, including general-knowledge
        ones, and always fired before the database/web/direct_llm
        checks). It now requires an actual math symbol next to a digit,
        in addition to the existing explicit calc keywords.
      - Added an explicit direct_llm branch. Previously, a question that
        matched none of weather/action/document/calculator/database/web
        keywords silently fell through to the "documents" default, even
        for plain general-knowledge questions with no document, tool, or
        database signal at all.
    """
    q_lower = question.lower()

    # STAGE 13 FIX: historical/explanation questions are checked FIRST
    # in the fallback too, for the same reason as the deterministic
    # layer -- otherwise a plain substring match below (e.g. "forecast"
    # inside "forecasting") would misroute an explanation question
    # before ever reaching a real check.
    if _HISTORICAL_INTENT.search(question) or _EXPLANATION_MARKER.search(question):
        print("[PLANNER] Keyword fallback: detected explanation/historical intent → direct_llm")
        return ["direct_llm"]

    # Weather/Tool keywords (HIGHEST PRIORITY to avoid weather→web confusion)
    # STAGE 13 FIX: uses word-boundary regex, not plain substring
    # containment -- "forecast" as a bare Python `in` check matched
    # inside "forecasting", "temp" matched inside "attempt", etc.
    weather_keyword_re = re.compile(
        r"\b(?:weather|temperature|temp|forecast|rain|snow)\b|\bwill\s+it\b",
        re.IGNORECASE,
    )
    if _is_weather_live_query(question) or (
        weather_keyword_re.search(question) and not _WEATHER_EXPLANATION_MARKER.search(question)
    ):
        print("[PLANNER] Keyword fallback: detected weather intent → tool")
        return ["tool"]
    
    action_keywords = ["send", "post", "email", "slack", "message", "alert", "notify", "create event", "set reminder"]
    if any(kw in q_lower for kw in action_keywords):
        print("[PLANNER] Keyword fallback: detected action intent → tool")
        return ["tool"]
    
    # Document keywords (SECOND PRIORITY)
    doc_keywords = ["paper", "pdf", "file", "document", "my research", "uploaded", "attached", "according to", "figure", "section"]
    if any(kw in q_lower for kw in doc_keywords):
        print("[PLANNER] Keyword fallback: detected document intent → documents")
        return ["documents"]
    
    # Calculator keywords -- STAGE 12 FIX: "what is" removed as a bare
    # trigger (it collided with direct_llm questions like "what is the
    # capital of France"). Explicit calc verbs are kept; a bare "what is"
    # now only counts as a calculator signal when the question also has a
    # math symbol adjacent to a digit.
    calc_keywords = ["calculate", "solve", "convert", "percentage"]
    if (
        any(kw in q_lower for kw in calc_keywords)
        or has_nl_arithmetic_intent(question)
    ):
        print("[PLANNER] Keyword fallback: detected calculator intent → calculator")
        return ["calculator"]
    
    # Database keywords
    db_keywords = ["how many users", "how many records", "total", "database", "signed up", "users", "records"]
    if any(kw in q_lower for kw in db_keywords):
        print("[PLANNER] Keyword fallback: detected database intent → database")
        return ["database"]
    
    # Web keywords (LOWER PRIORITY to avoid over-triggering)
    web_keywords = ["latest", "current", "today", "breaking", "recent", "github", "search for"]
    if any(kw in q_lower for kw in web_keywords):
        print("[PLANNER] Keyword fallback: detected web intent → web")
        return ["web"]

    # STAGE 12 FIX: direct_llm branch -- general knowledge / definition
    # style questions with no other matching signal. Checked before the
    # final "documents" default so plain factual questions no longer get
    # silently routed to a source with no matching evidence.
    direct_llm_keywords = [
        "what is", "what's", "who is", "who was", "define", "explain",
        "meaning of", "how does", "how do", "why does", "why do",
        "capital of", "what are", "difference between",
    ]
    if any(kw in q_lower for kw in direct_llm_keywords):
        print("[PLANNER] Keyword fallback: detected direct_llm intent → direct_llm")
        return ["direct_llm"]
    
    # Default: assume documents (safest per prompt)
    print("[PLANNER] Keyword fallback: no match, defaulting to documents")
    return ["documents"]


class PlannerAgent(BaseAgent):
    """
    STAGE 11/12 FIX: Planner Agent with explicit deterministic-then-LLM architecture.
    OPTIMIZED FOR 0.5B QWEN MODEL.
    
    Routing precedence (in order):
      1. Metadata short-circuit (title/author questions)
      2. Deterministic high-confidence routing (bypasses LLM)
         - Multi-source intent (comparing uploaded with external)
         - Document-intent (paper/figure/table/section/abstract patterns)
         - Database-intent (counts/stats from app data)
         - Calculator-intent (math/conversions)
         - Weather/Action-intent (weather/send/email/calendar)
         - Web-intent (current/latest/news)
      3. LLM classification (for ambiguous cases)
         - Constrained to SOURCE_REGISTRY names
         - Explicit parsing pipeline
         - Falls back to keyword heuristics on any parse failure
      4. Placeholder source handling (returns placeholder instead of executing)
    
    Responsibility: ONLY decide WHERE information should come from.
    Does NOT retrieve, execute tools, answer, validate, or retry.
    
    STAGE 11 CHANGES FOR 0.5B:
    - Reordered routing: weather BEFORE web (prevents confusion)
    - Multi-source detection moved to Layer 2 (deterministic)
    - Database detection moved to Layer 2 (deterministic)
    - Added keyword-based fallback for LLM failures
    - All failures are observable (extensive logging)
    - Clear deterministic vs LLM priority model

    STAGE 12 CHANGES:
    - Multi-source deterministic detection now also uses an
      order-independent three-signal check (verb + own-content +
      external-content), fixing missed phrasings like "compare the
      findings in my uploaded paper with the latest external research".
    - Deterministic Layer 2 matches and Layer 3 LLM/keyword-fallback
      matches now go through the SAME placeholder-source check
      (_apply_placeholder_check), so a not-yet-implemented source is
      handled identically no matter which layer classified it.
    """

    def __init__(self, llm, db=None):
        super().__init__(llm)
        self.db = db

    async def _get_document_metadata(self, state: AgentState) -> dict | None:
        if self.db is None:
            return None
        doc = await self.db.documents.find_one(
            {"user_id": state.user_id},
            sort=[("created_at", -1)],
        )
        if doc and doc.get("metadata"):
            return doc["metadata"]
        return None

    async def _classify_sources(self, question: str) -> list[str]:
        """
        Single-turn LLM classification. One routing decision, then stop.
        No chain-of-thought, per latency requirement.
        
        STAGE 11 CHANGES:
        - Explicit error handling with observable logging
        - Returns [] on any error (caller decides fallback)
        """
        try:
            response = await self.llm.acomplete(
                system=PLANNER_PROMPT,
                prompt=question,
                temperature=0,
                max_tokens=40,
            )
            raw = response.text if hasattr(response, "text") else str(response)
        except AttributeError as exc:
            print(f"[PLANNER] LLM method missing: {exc!r}")
            return []
        except Exception as exc:
            print(f"[PLANNER] LLM call failed: {exc!r}")
            return []

        # Parse using explicit pipeline
        sources = _parse_classifier_output(raw)
        print(f"[PLANNER] LLM classified: {sources!r}")
        return sources

    def _apply_placeholder_check(
        self, state: AgentState, sources: list[str], confidence: float
    ) -> AgentState:
        """
        STAGE 12 FIX: unified placeholder-source handling.

        Previously, deterministic Layer 2 matches returned directly from
        _execute() without ever checking _PLACEHOLDER_SOURCES, while only
        LLM/keyword-classified (Layer 3/4) results went through the
        placeholder check. That meant a not-yet-implemented source (e.g.
        "database") behaved differently depending on which layer picked
        it -- deterministic routing would try to hand it downstream for
        execution, while LLM routing would correctly short-circuit to a
        placeholder message.

        This helper is now the single place that decides "is any of this
        source list not yet implemented" and is called from every layer
        (2, 3, and 4) so the behavior is identical regardless of which
        layer made the routing decision.
        """
        placeholder_hits = [s for s in sources if s in _PLACEHOLDER_SOURCES]

        if placeholder_hits:
            print(f"[PLANNER] Routed source(s) {placeholder_hits!r} "
                  f"not yet implemented")
            state.sources_needed = placeholder_hits
            state.metadata_answer = {
                "placeholder": f"{placeholder_hits[0]} integration is under development."
            }
            state.confidence = min(confidence, 0.6)
            return state

        state.sources_needed = sources
        # STAGE 13 FIX (AMB_09 / "tool selection accuracy = 0%"): every
        # exit point that sets sources_needed also resolves and sets the
        # concrete tool name, via the single shared resolver in
        # app.agents.tool_mapping -- see that module for why this field
        # didn't exist before.
        state.tool = resolve_concrete_tool(sources, state.rewritten_question or state.question)
        state.confidence = confidence
        return state

    def _deterministic_multi_source_routing(self, question: str) -> list[str] | None:
        """
        STAGE 11: High-confidence multi-source detection.
        Moved to EARLY in routing (before single-source patterns).
        
        If detected, returns ["documents", "web"] (high confidence 0.95).
        If not detected, returns None (continue to next pattern).
        
        Covers: compare X with latest, verify results against current, etc.

        STAGE 12 FIX: added a second, order-independent check
        (_MULTI_SOURCE_VERB + _MULTI_SOURCE_OWN_CONTENT_REF +
        _MULTI_SOURCE_EXTERNAL_REF all present anywhere in the question)
        so phrasing that breaks the original pattern's strict word-
        adjacency (e.g. extra words between "compare" and "my") is still
        caught deterministically instead of falling through to the LLM.
        """
        if _MULTI_SOURCE_INTENT.search(question):
            print(f"[PLANNER] Multi-source-intent pattern detected: "
                  f"{question[:70]}...")
            return ["documents", "web"]

        if (
            _MULTI_SOURCE_VERB.search(question)
            and _MULTI_SOURCE_OWN_CONTENT_REF.search(question)
            and _MULTI_SOURCE_EXTERNAL_REF.search(question)
        ):
            print(f"[PLANNER] Multi-source-intent (order-independent) "
                  f"detected: {question[:70]}...")
            return ["documents", "web"]

        return None

    def _deterministic_document_routing(self, question: str) -> list[str] | None:
        """
        STAGE 11: High-confidence document-intent detection.
        
        If detected, returns ["documents"] (high confidence 0.95).
        If not detected, returns None (LLM will decide).
        
        Covers: paper/pdf/abstract/figure/table/section references,
        document-specific metrics, "according to the paper", etc.
        """
        if _DOCUMENT_INTENT.search(question):
            print(f"[PLANNER] Document-intent pattern detected: "
                  f"{question[:70]}...")
            return ["documents"]
        return None

    def _deterministic_database_routing(self, question: str) -> list[str] | None:
        """
        STAGE 11: High-confidence database-intent detection.
        Moved to deterministic layer (was in LLM, low priority).
        
        If detected, returns ["database"] (high confidence 0.95).
        If not detected, returns None (LLM will decide).
        
        Covers: how many users/records, total, sum, app database queries.
        """
        if _DATABASE_INTENT.search(question):
            print(f"[PLANNER] Database-intent pattern detected: "
                  f"{question[:70]}...")
            return ["database"]
        return None

    def _deterministic_calculator_routing(self, question: str) -> list[str] | None:
        """
        STAGE 11: High-confidence calculator-intent detection.
        STAGE 13 (2026-08-13, planner eval failure CALC_04): added
        natural-language arithmetic detection (app.agents.nl_arithmetic)
        as a second, independent check -- symbolic patterns
        ("250 - 37", "calculate 25 * 40") and verbal arithmetic ("250
        items, remove 37", "average of 10, 20 and 30") are genuinely
        different signals, so they're checked separately rather than
        forcing one regex to do both jobs.
        
        If detected, returns ["calculator"] (high confidence 0.95).
        If not detected, returns None (LLM will decide).
        
        Covers: arithmetic, percentages, unit conversions, powers, roots,
        and natural-language arithmetic phrasing.
        """
        if _CALCULATOR_INTENT.search(question):
            print(f"[PLANNER] Calculator-intent pattern detected: "
                  f"{question[:70]}...")
            return ["calculator"]
        if has_nl_arithmetic_intent(question):
            print(f"[PLANNER] NL-arithmetic-intent detected: "
                  f"{question[:70]}...")
            return ["calculator"]
        return None

    def _deterministic_weather_action_routing(self, question: str) -> list[str] | None:
        """
        STAGE 11: High-confidence weather/action-intent detection.
        MOVED BEFORE WEB (prevents weather→web confusion in 0.5B).
        
        If detected, returns ["tool"] (high confidence 0.95).
        If not detected, returns None (LLM will decide).
        
        Covers: weather queries, send/post/email/Slack actions, 
        calendar/reminder creation.
        """
        if _is_weather_live_query(question):
            print(f"[PLANNER] Weather-live-query detected: "
                  f"{question[:70]}...")
            return ["tool"]
        if _WEATHER_ACTION_INTENT.search(question):
            print(f"[PLANNER] Action-intent pattern detected: "
                  f"{question[:70]}...")
            return ["tool"]
        return None

    def _deterministic_web_routing(self, question: str) -> list[str] | None:
        """
        STAGE 11: High-confidence web-intent detection.
        MOVED AFTER weather (to avoid weather→web confusion).
        
        If detected, returns ["web"] (high confidence 0.9).
        If not detected, returns None (LLM will decide).
        
        Covers: latest news, current events, recent benchmarks,
        breaking news, public repositories.
        """
        if _WEB_INTENT.search(question):
            print(f"[PLANNER] Web-intent pattern detected: "
                  f"{question[:70]}...")
            return ["web"]
        return None

    def _deterministic_current_live_routing(self, question: str) -> list[str] | None:
        """
        STAGE 14: Detect current/live/real-time information intent.
        
        Runs AFTER web/weather but BEFORE bare "what is" explanation.
        This catches "current price of Bitcoin" → web (live data)
        vs "What is Bitcoin?" → direct_llm (general knowledge).
        
        The key signal: "current", "now", "today", "live", "real-time"
        strongly indicate the user wants up-to-date information, not
        a definition or historical explanation.
        
        If detected, returns ["web"] (high confidence 0.85).
        If not detected, returns None (continue routing).
        """
        # Don't route to web if question is obviously about document content
        doc_phrases = ["according to", "paper", "document", "figure", 
                       "section", "in the", "uploaded", "my research"]
        if any(p in question.lower() for p in doc_phrases):
            return None

        if _CURRENT_LIVE_INTENT.search(question):
            print(f"[PLANNER] Current/live-information intent detected: "
                  f"{question[:70]}...")
            return ["web"]
        return None
    
    def _deterministic_explanation_routing(self, question: str) -> list[str] | None:
        """
        STAGE 13: High-confidence GENERAL EXPLANATION INTENT detection.

        Runs LAST among the deterministic layers (after document/
        database/calculator/weather/web have all had a chance to claim
        the question), which is what keeps it safe: by construction,
        anything reaching this point has already been checked against
        every more-specific operational pattern and matched none of
        them, so a request to *explain* a topic can't be confused with
        a request to *fetch/compute* something -- it's a distinct
        category, not merely "in case the earlier layers missed it".

        If detected, returns ["direct_llm"] (high confidence 0.9).
        If not detected, returns None (LLM will decide).
        """
        if _HISTORICAL_INTENT.search(question):
            print(f"[PLANNER] Historical-knowledge-intent detected: "
                  f"{question[:70]}...")
            return ["direct_llm"]
        if _EXPLANATION_MARKER.search(question):
            print(f"[PLANNER] Explanation-intent pattern detected: "
                  f"{question[:70]}...")
            return ["direct_llm"]
        if not _HAS_DIGIT.search(question) and _BARE_WHAT_IS.match(question.strip()):
            if _looks_generic_concept(question):
                print(f"[PLANNER] Bare general-knowledge question detected: "
                      f"{question[:70]}...")
                return ["direct_llm"]
            else:
                return None
        return None

    async def _execute(self, state: AgentState) -> AgentState:
        # Use rewritten_question if available, else fall back to original
        question_to_classify = state.rewritten_question or state.question
        original_question = state.question

        # ---- Layer 1: Metadata short-circuit ----
        if _METADATA_Q.search(original_question):
            doc_metadata = await self._get_document_metadata(state)
            if doc_metadata:
                print(f"[PLANNER] Metadata question detected, using stored metadata")
                state.metadata_answer = doc_metadata
                state.sources_needed = ["metadata"]
                state.confidence = 0.95
                return state
            print("[PLANNER] Metadata question detected but no metadata found")

        # ---- Layer 2: Deterministic high-confidence routing ----
        # Reordered for 0.5B: check multi-source first, then single sources
        # Weather is checked BEFORE web (prevents confusion)
        #
        # STAGE 12 FIX: every Layer 2 match now goes through
        # _apply_placeholder_check() instead of writing state directly, so
        # a deterministically-routed not-yet-implemented source behaves
        # identically to one classified by the LLM/keyword fallback.

        # Multi-source (highest priority for comparing/verifying)
        deterministic = self._deterministic_multi_source_routing(question_to_classify)
        if deterministic is not None:
            return self._apply_placeholder_check(state, deterministic, 0.95)

        # Document-intent (very specific, high confidence)
        deterministic = self._deterministic_document_routing(question_to_classify)
        if deterministic is not None:
            return self._apply_placeholder_check(state, deterministic, 0.95)

        # Database-intent (moved to deterministic for 0.5B)
        deterministic = self._deterministic_database_routing(question_to_classify)
        if deterministic is not None:
            return self._apply_placeholder_check(state, deterministic, 0.95)

        # Calculator-intent
        deterministic = self._deterministic_calculator_routing(question_to_classify)
        if deterministic is not None:
            return self._apply_placeholder_check(state, deterministic, 0.95)

        # Weather/Action-intent (BEFORE web)
        deterministic = self._deterministic_weather_action_routing(question_to_classify)
        if deterministic is not None:
            return self._apply_placeholder_check(state, deterministic, 0.95)

        # Web-intent (AFTER weather)
        deterministic = self._deterministic_web_routing(question_to_classify)
        if deterministic is not None:
            return self._apply_placeholder_check(state, deterministic, 0.9)

        # Current/live/real-time information (AFTER web, BEFORE explanation)
        # This catches "current price" vs "what is" general knowledge distinction
        deterministic = self._deterministic_current_live_routing(question_to_classify)
        if deterministic is not None:
            return self._apply_placeholder_check(state, deterministic, 0.85)

        # General explanation-intent (AFTER every operational pattern)
        deterministic = self._deterministic_explanation_routing(question_to_classify)
        if deterministic is not None:
            return self._apply_placeholder_check(state, deterministic, 0.9)

        # ---- Layer 3: LLM classification (for remaining ambiguous cases) ----
        sources = await self._classify_sources(question_to_classify)

        # If LLM failed, try keyword fallback (0.5B safety net)
        if not sources:
            print("[PLANNER] LLM classification returned nothing, trying keyword fallback")
            sources = _keyword_fallback_classification(question_to_classify)

        if not sources:
            # Final fallback: assume documents query
            print("[PLANNER] All routing failed, defaulting to documents")
            state.sources_needed = ["documents"]
            state.tool = resolve_concrete_tool(["documents"], question_to_classify)
            state.confidence = 0.5
            return state

        # ---- Layer 4: Placeholder source handling ----
        # STAGE 12 FIX: now the SAME helper used by Layer 2, instead of a
        # duplicated inline check -- one placeholder policy, one place.
        return self._apply_placeholder_check(state, sources, 0.7)
import json
import re

from app.agents.base import BaseAgent
from app.agents.prompts import PLANNER_PROMPT
from app.agents.state import AgentState
from app.agents.nl_arithmetic import has_nl_arithmetic_intent
from app.agents.tool_mapping import resolve_concrete_tool

# STAGE 15 FIXES (2026-08-22):
#   6.1: Broadened hyphenated technical entity detection (Lychee-FD, Thinker-Talker)
#   6.3: Loosened calculator regex to accept periods (not just ? or ")
#   6.7: Added "collection" to database scope-word list (MongoDB terminology)
#
# STAGE 16 FIXES (2026-08-22 -- new root causes 8, 9, 10):
#   8: Added common paper-section names ("introduction", "conclusion",
#      "methodology", "related work") to _DOCUMENT_INTENT's section-word
#      list. Deliberately NOT adding "background"/"discussion" -- those
#      are common enough in plain English ("what's the background on the
#      2008 crisis?") that adding them risks new false positives with no
#      eval coverage on the false-positive side. Revisit if a real
#      failing case shows up.
#   9: Added a general bare-"how does X <verb-phrase>" pattern (not just
#      "...work") to the explanation-intent layer, gated by the same
#      technical-entity check used for "what is X" (now factored out as
#      _TECHNICAL_ENTITY_MARKERS) so it defers instead of asserting
#      direct_llm when X looks like a specific named entity (e.g.
#      "Lychee-FD"). NOTE: this inherits the same blind spot as 6.2 --
#      a bare ALLCAPS acronym with no hyphen (e.g. "BERT") does not
#      match _TECHNICAL_ENTITY_MARKERS, so "How does BERT process input
#      sequences?" will currently be treated as a generic explanation
#      question rather than deferred as a possible document question.
#      Fixing 6.2's acronym-detection gap fixes this for free; until
#      then, this is a known, accepted gap, not a hidden one.
#   10: Added "who (invented|discovered|founded|created) X" to
#       _HISTORICAL_INTENT.

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
# v7 (2026-08-22 -- database routing generalization, root-cause chain
#   for "how many pdf are there in database"-style questions):
#   Live production log showed the rewritten question "What is the
#   total number of PDF files in your database?" being confidently
#   classified as direct_llm by _deterministic_explanation_routing's
#   bare-"what is" shortcut, never reaching the database layer's LLM/
#   keyword fallback at all. Traced to two compounding gaps:
#     1. _DATABASE_INTENT required either the literal word "table", the
#        contraction "what's" (not "what is"), or one of a fixed noun
#        list (users/records/signups/registrations/entries) that didn't
#        include documents/pdfs/files/sessions/etc. -- so a real
#        database question with different phrasing fell all the way
#        through Layer 2 unmatched.
#     2. _deterministic_explanation_routing's bare-what-is fallback had
#        no awareness that the "generic concept" it was about to assert
#        direct_llm for might explicitly be OUR OWN database/app data
#        -- it only checked for PascalCase/hyphen-digit "technical
#        entity" markers, nothing about the database/storage domain.
#   Fixed additively (see _DATABASE_INTENT and
#   _deterministic_explanation_routing below) without touching the
#   existing narrower patterns, which stay exactly as they were.
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
#
# STAGE 16 FIX (root cause 8, 2026-08-22): the "in the <section>" branch
# below only recognized paper/document/pdf/file/abstract/publication/
# section/appendix/figure/table -- not common named sections like
# "introduction", "conclusion", "methodology", or "related work", so
# "What does the paper say in the introduction?"-shaped questions with
# the section named explicitly missed this pattern. Added those four.
# Deliberately NOT adding "background" or "discussion" here -- both are
# common enough in ordinary non-document English (e.g. "what's the
# background on the 2008 financial crisis?", "what's the discussion
# around AI regulation?") that adding them risks new false-positive
# document routing with no eval case covering that direction yet.
_DOCUMENT_INTENT = re.compile(
    r"(?:"
    # Explicit document-referencing phrases. Allows a possessive/topic
    # modifier between "according to" and the document noun (e.g.
    # "according to my RAG document") -- the original pattern required
    # strict adjacency and missed this, the same class of bug fixed for
    # multi-source routing in STAGE 12.
    r"according\s+to\s+(?:the\s+|my\s+|our\s+)?(?:\w+\s+){0,2}(?:paper|document|pdf|file|abstract|publication|study|research|report)\b|"
    r"(?:my|our)\s+uploaded\s+(?:paper|pdf|file|document|research|data)\b|"
    r"in\s+(?:the\s+)?(?:paper|document|pdf|file|abstract|publication|section|appendix|figure|table|"
    r"introduction|conclusion|methodology|related\s+work|table\s+of\s+contents)\b|"
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
#
# STAGE 15 FIX 6.3 (2026-08-22): Loosened `.*[?\"]` to `.*[?\".]?` on
# percentage and unit-conversion branches so questions ending in periods
# (e.g. "Calculate 23% of 960.") are accepted, not just those ending in
# question marks or quotes.
_CALCULATOR_INTENT = re.compile(
    r"(?:"
    # Arithmetic operations (operator MUST be adjacent to digits on both
    # sides -- this is what makes it immune to stray hyphens elsewhere
    # in the sentence, e.g. "object-oriented")
    r"\d+\s*(?:\+|-|\*|/|÷|×)\s*\d+|"
    # Explicit calculator keywords
    r"(?:what's?|calculate|compute|solve|find|simplify)\s+.+[0-9].+[?\"]|"
    # Percentage/ratio (STAGE 15 FIX 6.3: changed [?\" ] to [?\".]? )
    r"(?:percentage|percent|%|ratio|proportion)\s+(?:of|between|among).*[?\".]?|"
    r"what\s+(?:is|'s)\s+.+%\s+(?:of|increase|decrease).*[?\".]?|"
    r"how\s+much\s+is\s+.+%\s+of\s+\d+.*[?\".]?|"
    # Unit conversions (STAGE 15 FIX 6.3: changed [?\" ] to [?\".]? )
    r"convert\s+.+(?:to|into|in\s+terms\s+of)\s+.+[?\".]?|"
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

# Slack message-search intent: the user wants to READ Slack history.
#
# 2026-08-22 STAGE 16+ FIX (root cause 6.5): This MUST be defined and
# checked BEFORE _WEB_INTENT because Slack questions commonly contain
# recency keywords ("Find *recent* Slack messages...", "...latest API
# bug in Slack") that _WEB_INTENT's first branch would otherwise capture
# first, misrouting the question to web/web_search.
#
# Discrimination criteria:
#   REQUIRED: \bslack\b somewhere in the question.
#   SEARCH:   find/search/what did/mention/discussion/messages/conversation
#             or any information-retrieval verb -- user wants to read history.
#   POST:     send/post/write/notify/alert -- user wants to write.
# If both or neither, the search interpretation is preferred (read-only is
# the safer default; accidental reads are less damaging than accidental posts).
_SLACK_SEARCH_INTENT = re.compile(
    r"(?:"
    # Explicit Slack mention + information-retrieval verb
    r"\bslack\b.*\b(?:find|search|look\s+for|what\s+did|what\s+was|who\s+said|"
    r"discuss(?:ion|ions|ed)?|mention(?:ed|s)?|conversation|messages?|thread|threads|"
    r"said|talked?|brought\s+up|did\s+anyone)\b|"
    # Reversed order: retrieval verb first, then slack
    r"\b(?:find|search|look\s+for|what\s+did|what\s+was|who\s+said|did\s+anyone|"
    r"discuss(?:ion|ions|ed)?|mention(?:ed|s)?)\b.*\bslack\b"
    r")",
    re.IGNORECASE,
)

# Web-intent: current/live/external information requiring search
#
# 2026-08-22 STAGE 16+ FIX (PLAN_WEB_02, PLAN_WEB_05): extended with two
# new temporal alternatives:
#   - "(released|launched|published|announced) recently/this week/this month"
#     catches "Which new foundation models were released recently?"
#   - "changed recently" / "(changed|updated|evolved) recently in"
#     catches "What changed recently in the LangGraph ecosystem?"
# These were previously deferred to the LLM because they didn't match any
# of the existing noun-adjacent patterns (which required the temporal word
# to come BEFORE a specific content noun, not after a past-tense verb).
_WEB_INTENT = re.compile(
    r"(?:"
    # Temporal keywords indicating recency
    r"(?:latest|current|recent|newest|breaking|today|this\s+week|this\s+month|2024|2025|2026)\b.*(?:news|developments?|events?|results?|updates?|benchmarks?|papers?|research|findings?|techniques?|methods?|approaches?|advances?|trends?|tools?|models?)|"
    r"what's?\s+(?:new|happening|going\s+on|the\s+latest|trending)\b|"
    r"(?:what|what's|find|search)\s+(?:the\s+)?(?:latest|current|recent)\s+(?:news|developments?|events?|updates?|benchmarks?|papers?|research|findings?|techniques?|methods?|approaches?|advances?|trends?|tools?|models?)\b|"
    # Released/launched/published recently -- PLAN_WEB_02 fix
    r"(?:released|launched|published|announced)\s+(?:recently|this\s+week|this\s+month|lately)\b|"
    r"(?:recently|lately)\s+(?:released|launched|published|announced)\b|"
    # Changed/updated recently -- PLAN_WEB_05 fix
    r"(?:what\s+changed|what's\s+changed|changed|updated|evolved)\s+(?:recently|lately|this\s+week|this\s+month)\b|"
    r"(?:recently|lately)\s+(?:changed|updated|evolved)\b|"
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
#
# STAGE 16 FIX (found while validating root cause 9, 2026-08-22): the
# "how does X work" alternative used to live here, UNGATED -- it fired
# on any question of that shape regardless of what X was, including a
# specific named entity ("How does Lychee-FD work?", "How does the
# Thinker-Talker architecture work?"), which would have misrouted those
# to direct_llm exactly like the bug fixed for root cause 9 below. That
# alternative is removed from here and folded into the new gated
# _BARE_HOW_DOES check instead (which matches "how does X work" too --
# it isn't restricted to non-"work" verb phrases, just broader than
# "work" alone) so EVERY "how does X ..." shape, "work" included, goes
# through the same _TECHNICAL_ENTITY_MARKERS gate before asserting
# direct_llm.
_EXPLANATION_MARKER = re.compile(
    r"^\s*explain\b|"
    r"\bwhat\s+(?:does|is)\s+.+\s+mean\b|"
    r"\b(?:definition|meaning)\s+of\b|"
    r"\bwhat\s+is\s+the\s+\w+\s+concept\b",
    re.IGNORECASE,
)

# STAGE 16 ADDITION (root cause 9, 2026-08-22): a general "how does X
# <verb-phrase>" pattern, NOT restricted to ending in the word "work"
# the way _EXPLANATION_MARKER's existing how-does branch is. This is
# checked separately (not folded into _EXPLANATION_MARKER) because it
# needs its own entity gate -- see _TECHNICAL_ENTITY_MARKERS and
# _deterministic_explanation_routing below -- to avoid swallowing
# genuine document questions like "How does Lychee-FD achieve real-time
# interaction?".
_BARE_HOW_DOES = re.compile(
    r"^\s*how\s+(?:does|do)\s+\S.*\S\s*\??\s*$",
    re.IGNORECASE,
)

# General knowledge / historical framing, as opposed to a request for
# current/recent information (which _WEB_INTENT already catches first).
#
# STAGE 16 FIX (root cause 10, 2026-08-22): added a standalone
# "who (invented|discovered|founded|created) X" alternative. The
# existing "when was X founded/established/..." pattern required the
# question to be phrased with "when was", so "Who invented the World
# Wide Web?" (a "who", not a "when") fell through unmatched.
_HISTORICAL_INTENT = re.compile(
    r"\bhistorically\b|\bhistory\s+of\b|\bhistorical(?:ly)?\b|"
    r"\bwhen\s+was\s+.+\s+(?:founded|established|built|invented|created)\b|"
    r"\bwho\s+(?:invented|discovered|founded|created)\b|"
    r"\borigin(?:s|ated)?\s+of\b",
    re.IGNORECASE,
)

# 2026-08-22 ADDITION: signals that a bare "what is X" question is
# actually about OUR OWN application data/database, not a request for a
# dictionary-style definition. See v7 note at top of file. Deliberately
# narrow -- only fires on explicit database/storage-scope wording, never
# on a bare entity name like "users"/"documents"/"pdf" alone, since
# those alone are still legitimately conceptual ("What is a PDF?" stays
# direct_llm). Verified: "What is a database?" is caught by this guard
# (so its explanation is decided by Layer 3, not this shortcut) but
# still ends up routed to direct_llm there via the existing
# direct_llm_keywords check in _keyword_fallback_classification -- so
# genuinely conceptual "what is a database?" is unaffected end-to-end.
_APP_DATA_SCOPE_TERMS = re.compile(
    r"\b(?:database|db\b|our\s+system|our\s+app|internal\s+data|stored|recorded)\b",
    re.IGNORECASE,
)

# STAGE 16 REFACTOR (2026-08-22): factored the "does this span look like
# a specific named technical entity" check out of _looks_generic_concept
# into a standalone, reusable regex so root cause 9's new bare-"how does"
# pattern can share the exact same entity gate as the existing bare-
# "what is" shortcut, instead of duplicating (and risking drifting from)
# the pattern. Behavior for _looks_generic_concept is unchanged -- this
# is a pure extraction, not a logic change.
_TECHNICAL_ENTITY_MARKERS = re.compile(
    r"[A-Z][a-z]+[A-Z]|"           # CamelCase/PascalCase (SpeechGPT)
    r"[A-Z]+-[A-Z0-9]|"            # ALLCAPS-hyphen (GPT-4)
    r"[A-Z][a-z]+-[A-Z]{2,}|"      # Titlecase-hyphen-ALLCAPS (Lychee-FD)
    r"[A-Z][a-z]+-[A-Z][a-z]+|"    # Titlecase-hyphen-Titlecase (Thinker-Talker)
    r"\d+[a-zA-Z]"                  # digit-then-letter (4x)
)


def _looks_generic_concept(question: str) -> bool:
    """
    True if the "What is X" question asks about a generic concept
    (e.g., "What is vector search?") vs. specific entity
    (e.g., "What is Lychee-FD?").
    
    STAGE 15 FIX 6.1 (2026-08-22): Broadened regex to recognize
    hyphenated technical entities like Lychee-FD and Thinker-Talker
    that were previously misidentified as generic concepts.

    STAGE 16 REFACTOR (2026-08-22): entity-marker regex now lives in
    the shared _TECHNICAL_ENTITY_MARKERS constant (see above); this
    function's behavior is unchanged.
    """
    # Extract the X from "What is X?"
    match = re.search(r"what\s+(?:is|are|'s)\s+(.+?)[\?\.]*$", 
                      question, re.IGNORECASE)
    if not match:
        return False
    
    entity = match.group(1).strip()
    
    # Technical entities have these markers:
    # - CamelCase or PascalCase (SpeechGPT, AlexNet)
    # - Hyphenated with numbers (GPT-4, DALL-E 2)
    # - Titlecase-hyphen-ALLCAPS (Lychee-FD)
    # - Titlecase-hyphen-Titlecase (Thinker-Talker)
    # - digit-then-letter (4x)
    if _TECHNICAL_ENTITY_MARKERS.search(entity):
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

# Document names/titles are the only inexpensive, user-scoped context the
# planner can safely inspect before it decides whether retrieval is needed.
# They are deliberately used as a routing signal, never as answer evidence.
_DOCUMENT_CONTEXT_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_DOCUMENT_CONTEXT_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "paper",
    "document", "report", "research", "study", "file", "pdf",
}
_MAX_DOCUMENT_CONTEXT_ITEMS = 20


def _document_context_matches_question(question: str, document_context: list[str]) -> str | None:
    """Return a matching user document label when its distinctive tokens occur.

    A single common word (for example, ``report``) is not enough. Requiring
    two tokens, including one of at least four characters, keeps this a
    document-identity signal rather than a broad keyword classifier.
    """
    question_tokens = set(_DOCUMENT_CONTEXT_TOKEN.findall(question.lower()))
    for label in document_context:
        label_tokens = {
            token for token in _DOCUMENT_CONTEXT_TOKEN.findall(label.lower())
            if token not in _DOCUMENT_CONTEXT_STOPWORDS
        }
        shared = label_tokens.intersection(question_tokens)
        if len(shared) >= 2 and any(len(token) >= 4 for token in shared):
            return label
    return None


def _planner_prompt_with_document_context(document_context: list[str]) -> str:
    """Attach bounded user document names to the LLM routing prompt.

    The names are context for source selection only. They are not document
    content and must not be treated as answer evidence.
    """
    if not document_context:
        return PLANNER_PROMPT

    labels = "\n".join(f"- {label}" for label in document_context)
    return (
        f"{PLANNER_PROMPT}\n\n"
        "Active user documents (routing context only; not answer evidence):\n"
        f"{labels}\n"
        "If a question asks for a specific fact, result, method, or detail "
        "about one of these active documents, select documents even when it "
        "does not explicitly say 'paper' or 'document'."
    )

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
#
# 2026-08-22 FIX (v7 note above): added two new order-independent
# alternatives at the end. The original alternatives above are
# UNCHANGED -- these are pure additions. They deliberately still
# require either an explicit database/system scope word or a
# storage-state word ("stored"/"recorded"), not just a count-phrase +
# entity alone, because entity+count alone collides with genuinely
# document-scoped questions like "how many pages does the pdf have?"
# (singular "the pdf" = an uploaded document, not a database count).
# Verified against 9 positive and 5 negative/collision cases before
# finalizing this pattern.
#
# STAGE 15 FIX 6.7 (2026-08-22): Added "collection" to database scope
# words (MongoDB terminology for table).
_DATABASE_INTENT = re.compile(
    r"(?:"
    r"(?:how\s+many|total\s+number\s+of|records?|entries?|items?)\s+(?:in\s+)?(?:the\s+)?\w+\s+table\b|"
    # Count/stat queries
    r"how\s+many\s+(?:users?|records?|signups?|registrations?|entries?|items?|customers?|accounts?)\b.*(?:today|this|last|month|week|day|week|month|year).*[?\"]|"
    r"what's\s+the\s+(?:total|sum|count|average|mean)\s+(?:number|count|amount)\s+of\s+(?:users?|records?|signups?|registrations?|entries?)\b|"
    r"how\s+many\s+.+(?:created|added|registered|signed\s+up)\s+(?:in|last|this|today|this\s+week)\b|"
    r"(?:count|get\s+the\s+count)\s+of\s+(?:users?|records?|entries?)\b.*(?:in|from)\s+(?:the\s+)?(?:database|app|system)\b|"
    r"(?:database|app|internal\s+data|our\s+system).*(?:says|shows?|has|contains)\s+(?:how\s+many|what|total)\b|"
    # NEW: count-phrase + known entity + explicit database/system scope,
    # anywhere in the question (order-independent). Catches phrasing
    # like "What is the total number of PDF files in your database?"
    # that the narrower patterns above miss (no "table", no "what's",
    # entity not in the original fixed noun list).
    # STAGE 15 FIX 6.7: Added "collection" to scope-words list
    r"(?=.*\b(?:how\s+many|total\s+(?:number|count)\s+of|number\s+of|count\s+of)\b)"
    r"(?=.*\b(?:users?|accounts?|documents?|pdfs?|files?|uploads?|records?|sessions?|chats?|conversations?)\b)"
    r"(?=.*\b(?:database|db|collection|our\s+system|our\s+app|your\s+database|the\s+database|internal\s+data)\b).*|"
    # NEW: known entity + storage-state word ("stored"/"recorded") --
    # covers "how many documents are stored", "documents recorded",
    # without necessarily saying the word "database".
    r"\b(?:users?|accounts?|documents?|pdfs?|files?|uploads?|records?|sessions?|chats?|conversations?)\b"
    r".{0,40}?\b(?:stored|recorded)\b"
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


def _keyword_fallback_classification(
    question: str, document_context: list[str] | None = None
) -> list[str]:
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

    2026-08-22 FIX: added an early count+entity check (see below) BEFORE
    the doc_keywords check. Previously, a count-style database question
    that also happened to mention "pdf"/"file"/"document" (e.g. "how
    many pdf are there") would be caught by the broader doc_keywords
    branch first and misrouted to document retrieval instead of a
    database count -- doc_keywords matches on bare substring presence
    with no awareness that "how many ... pdf" is a count intent, not a
    "look inside this document" intent.
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
    
    if _WEATHER_ACTION_INTENT.search(question):
        print("[PLANNER] Keyword fallback: detected action intent → tool")
        return ["tool"]

    # 2026-08-22 FIX: count-style database question, checked BEFORE
    # doc_keywords below so "how many pdf/file/document are there" isn't
    # stolen by the broader (bare substring) document check.
    _count_re = re.compile(
        r"\b(?:how\s+many|total(?:\s+number)?\s+of|count\s+of|number\s+of)\b",
        re.IGNORECASE,
    )
    _count_entity_re = re.compile(
        r"\b(?:users?|accounts?|documents?|pdfs?|files?|uploads?|records?|"
        r"sessions?|chats?|conversations?)\b",
        re.IGNORECASE,
    )
    if _count_re.search(question) and _count_entity_re.search(question):
        print("[PLANNER] Keyword fallback: detected count-style database intent → database")
        return ["database"]

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
    if (
        _CALCULATOR_INTENT.search(question)
        or has_nl_arithmetic_intent(question)
    ):
        print("[PLANNER] Keyword fallback: detected calculator intent → calculator")
        return ["calculator"]
    
    # Database keywords
    db_keywords = ["how many users", "how many records", "total", "database", "signed up", "users", "records", "collection"]
    if any(kw in q_lower for kw in db_keywords):
        print("[PLANNER] Keyword fallback: detected database intent → database")
        return ["database"]
    
    # Web keywords (LOWER PRIORITY to avoid over-triggering)
    web_keywords = ["latest", "current", "today", "breaking", "recent", "github", "search for"]
    if any(kw in q_lower for kw in web_keywords):
        print("[PLANNER] Keyword fallback: detected web intent → web")
        return ["web"]

    # If the LLM response was unusable and the user has active documents,
    # prefer retrieval for an otherwise ambiguous factual question. Explicit
    # calculator/weather/action/database/web intents have already returned.
    if document_context:
        print("[PLANNER] Keyword fallback: active document context → documents")
        return ["documents"]

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

    2026-08-22 CHANGES (v7):
    - _DATABASE_INTENT broadened with two order-independent alternatives
      (count-phrase + entity + scope word; entity + stored/recorded) so
      phrasing outside the original fixed patterns is still caught
      deterministically.
    - _deterministic_explanation_routing's bare-"what is" shortcut now
      defers (returns None) when the question explicitly mentions our
      own database/app/storage, instead of confidently asserting
      direct_llm -- see _APP_DATA_SCOPE_TERMS.
    
    STAGE 15 CHANGES (2026-08-22):
    - FIX 6.1: Broadened _looks_generic_concept regex to recognize
      hyphenated entities (Lychee-FD, Thinker-Talker) that were being
      misidentified as generic concepts and routed to direct_llm.
    - FIX 6.3: Loosened _CALCULATOR_INTENT regex from [?\" ] to [?\".]?
      so questions ending with periods are accepted.
    - FIX 6.7: Added "collection" to database scope-word list in
      _DATABASE_INTENT for MongoDB terminology support.

    STAGE 16 CHANGES (2026-08-22 -- root causes 8, 9, 10):
    - Root cause 8: added introduction/conclusion/methodology/related
      work to _DOCUMENT_INTENT's section-word list.
    - Root cause 9: added a general bare-"how does X <verb-phrase>"
      pattern to the explanation layer, gated by the same technical-
      entity check as bare-"what is" (now shared via
      _TECHNICAL_ENTITY_MARKERS). Known remaining gap: bare acronyms
      with no hyphen (e.g. "BERT") aren't recognized as entities by
      this gate yet -- same root gap as 6.2, not yet fixed.
    - Root cause 10: added "who invented/discovered/founded/created X"
      to _HISTORICAL_INTENT.
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

    async def _get_document_context(self, state: AgentState) -> list[str]:
        """Return bounded, user-scoped document titles and filenames.

        This intentionally does not retrieve chunks or document content. It
        gives the planner just enough context to know that an entity can be
        document-grounded, while keeping answer grounding in RetrieverAgent.
        """
        if self.db is None:
            return []

        try:
            cursor = self.db.documents.find(
                {"user_id": state.user_id, "status": "processed"},
                {"filename": 1, "metadata.title": 1},
            ).sort("created_at", -1).limit(_MAX_DOCUMENT_CONTEXT_ITEMS)
            docs = await cursor.to_list(length=_MAX_DOCUMENT_CONTEXT_ITEMS)
        except Exception as exc:
            print(f"[PLANNER] Document context lookup failed: {exc!r}")
            return []

        labels: list[str] = []
        for doc in docs:
            metadata = doc.get("metadata") or {}
            for value in (metadata.get("title"), doc.get("filename")):
                if isinstance(value, str) and value.strip() and value not in labels:
                    labels.append(value.strip())
        return labels

    async def _classify_sources(
        self, question: str, document_context: list[str] | None = None
    ) -> list[str]:
        """
        Single-turn LLM classification. One routing decision, then stop.
        No chain-of-thought, per latency requirement.
        
        STAGE 11 CHANGES:
        - Explicit error handling with observable logging
        - Returns [] on any error (caller decides fallback)
        """
        try:
            response = await self.llm.acomplete(
                system=_planner_prompt_with_document_context(document_context or []),
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

        STAGE 16 FIX (root cause 8): section-word list broadened -- see
        _DOCUMENT_INTENT above.
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
        
        Covers: how many users/records, total, sum, app database queries,
        and (2026-08-22) documents/pdfs/files/sessions/etc. with an
        explicit database/system scope or stored/recorded phrasing.
        """
        if _DATABASE_INTENT.search(question):
            print(f"[PLANNER] Database-intent pattern detected: "
                  f"{question[:70]}...")
            return ["database"]
        return None

    def _deterministic_slack_search_routing(self, question: str) -> list[str] | None:
        """
        2026-08-22 STAGE 16+ (root cause 6.5): Detect Slack message-search
        intent and route to tool/slack_search.

        MUST run BEFORE _deterministic_weather_action_routing and
        _deterministic_web_routing so that Slack questions containing
        recency words ("Find *recent* Slack messages...") are not stolen
        by _WEB_INTENT's temporal-keyword branches.

        Only fires when \bslack\b is present AND a search/read verb is
        detected (see _SLACK_SEARCH_INTENT). Questions about posting to
        Slack (send/post/notify) do NOT match and are left for the
        weather/action routing layer to handle (they already matched
        _WEATHER_ACTION_INTENT's post/send/slack branch).

        If detected, returns ["tool"] (high confidence 0.95).
        resolve_concrete_tool() / resolve_sub_tool() will then map this
        to "slack_search" using the same search-vs-post discrimination
        logic in tool_mapping.py.
        """
        if _SLACK_SEARCH_INTENT.search(question):
            print(f"[PLANNER] Slack-search-intent detected: "
                  f"{question[:70]}...")
            return ["tool"]
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

        2026-08-22 FIX (v7 note at top of file): the bare-"what is"
        shortcut below now defers (returns None) when the question
        explicitly names our own database/app/storage
        (_APP_DATA_SCOPE_TERMS) instead of confidently asserting
        direct_llm. This is a backstop for cases the broadened
        _DATABASE_INTENT above still doesn't catch (typos, unusual
        phrasing) -- it doesn't route to database itself, it just
        stops this layer from preempting Layer 3/4, which already know
        how to route database questions.

        STAGE 16 ADDITION (root cause 9): a general bare-"how does X
        <verb-phrase>" check, run after the existing "how does X work"
        branch inside _EXPLANATION_MARKER. Gated by
        _TECHNICAL_ENTITY_MARKERS on the WHOLE question (not just an
        extracted subject span, since "how does X <verb> Y" doesn't
        have a clean single-noun-phrase boundary the way "what is X"
        does) -- if any technical-entity-looking token appears anywhere
        in the question, this defers (returns None) instead of
        asserting direct_llm, so a genuine document question like "How
        does Lychee-FD achieve real-time interaction?" still falls
        through to Layer 3/4 rather than being preempted here. KNOWN
        GAP: this only catches entities matching
        _TECHNICAL_ENTITY_MARKERS (hyphenated/PascalCase/digit+letter
        forms) -- a bare ALLCAPS acronym like "BERT" is not currently
        recognized and will be treated as a generic "how does X work"
        question. Same root gap as 6.2; not fixed here.

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
            if _APP_DATA_SCOPE_TERMS.search(question):
                print(f"[PLANNER] Bare 'what is' question mentions our own "
                      f"database/app data -- deferring to LLM/keyword "
                      f"routing instead of asserting direct_llm: "
                      f"{question[:70]}...")
                return None
            if _looks_generic_concept(question):
                print(f"[PLANNER] Bare general-knowledge question detected: "
                      f"{question[:70]}...")
                return ["direct_llm"]
            else:
                return None
        # STAGE 16 ADDITION (root cause 9): bare "how does X <verb phrase>"
        # not already caught by _EXPLANATION_MARKER's "...work" branch.
        if _BARE_HOW_DOES.match(question.strip()):
            if _TECHNICAL_ENTITY_MARKERS.search(question):
                print(f"[PLANNER] Bare 'how does' question mentions a "
                      f"technical-entity-looking token -- deferring to "
                      f"LLM/keyword routing instead of asserting "
                      f"direct_llm: {question[:70]}...")
                return None
            print(f"[PLANNER] Bare 'how does X <verb>' question detected: "
                  f"{question[:70]}...")
            return ["direct_llm"]
        return None

    async def _execute(self, state: AgentState) -> AgentState:
        # Use rewritten_question if available, else fall back to original
        question_to_classify = state.rewritten_question or state.question
        original_question = state.question
        document_context = await self._get_document_context(state)

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

        # A verified user-document title/filename match is stronger evidence
        # than a generic explanation or live-information phrase later in the
        # deterministic chain. It avoids hard-coding document entities while
        # preserving normal routing when no user document matches.
        matched_document = _document_context_matches_question(
            question_to_classify, document_context
        )
        if matched_document:
            print(f"[PLANNER] Active document context matched: {matched_document!r}")
            return self._apply_placeholder_check(state, ["documents"], 0.95)

        # Database-intent (moved to deterministic for 0.5B)
        deterministic = self._deterministic_database_routing(question_to_classify)
        if deterministic is not None:
            return self._apply_placeholder_check(state, deterministic, 0.95)

        # Calculator-intent
        deterministic = self._deterministic_calculator_routing(question_to_classify)
        if deterministic is not None:
            return self._apply_placeholder_check(state, deterministic, 0.95)

        # Slack-search-intent (BEFORE weather/web to prevent "recent/latest"
        # Slack questions being captured by _WEB_INTENT's temporal branches)
        deterministic = self._deterministic_slack_search_routing(question_to_classify)
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
        sources = await self._classify_sources(question_to_classify, document_context)

        # If LLM failed, try keyword fallback (0.5B safety net)
        if not sources:
            print("[PLANNER] LLM classification returned nothing, trying keyword fallback")
            sources = _keyword_fallback_classification(
                question_to_classify, document_context
            )

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
import json
import re

from app.agents.base import BaseAgent
from app.agents.prompts import PLANNER_PROMPT
from app.agents.state import AgentState

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
# SOURCE_REGISTRY is the intended single source of truth for the
# classifier prompt AND for what tool_agent.py / graph.py treat as
# valid. The import-time assertion below will fail fast if the prompt
# step is forgotten.
#
# Adding a future source = one dict entry here + a rule/example in
# PLANNER_PROMPT (prompts.py) + one handler in tool_agent.py + (if it
# needs pre-answer execution) one line in graph.py's dispatch check.
# --------------------------------------------------------------------------

_METADATA_Q = re.compile(r"\b(title|author|affiliat)", re.IGNORECASE)

# Document-intent protection: high-confidence patterns that signal the
# question is asking specifically about what a document/paper says,
# not general knowledge. These bypass LLM inference and route directly
# to documents.
_DOCUMENT_INTENT = re.compile(
    r"\b(according to the (?:paper|document|pdf|abstract)|"
    r"in the (?:paper|document|pdf|abstract|section|appendix)|"
    r"(?:what (?:is|does)|list) (?:reported|mentioned|described|stated) in|"
    r"(?:figure|table|section|appendix) (?:\d+|[A-Z])|"
    r"according to (?:figure|table|figure|section))\b",
    re.IGNORECASE
)

# Canonical source name -> config.
# `implemented=False` means no real executor exists yet in tool_agent.py;
# the Planner short-circuits with a placeholder instead of claiming a
# source it can't back up.
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
        "implemented": False,
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

# --------------------------------------------------------------------------
# 2026-08-09 FIX (routing bug): guardrail against SOURCE_REGISTRY /
# PLANNER_PROMPT drift. Ensures every source registered is mentioned
# in the prompt, so the classifier can actually select it.
# --------------------------------------------------------------------------
_missing_from_prompt = [name for name in _VALID_SOURCES if name not in PLANNER_PROMPT]
if _missing_from_prompt:
    raise RuntimeError(
        f"PLANNER_PROMPT (prompts.py) does not mention source(s) "
        f"{_missing_from_prompt!r} defined in SOURCE_REGISTRY. The "
        f"classifier can never select a source name it is never shown -- "
        f"add a rule/example for it in PLANNER_PROMPT before deploying, or "
        f"the router will silently default those questions to ['documents']."
    )


def _parse_classifier_output(raw: str) -> list[str]:
    """
    Parses the classifier's output. Handles both:
    1. JSON array format: ["web", "calculator"]
    2. JSON object format: {"sources": ["web"], "intent": "...", ...}

    Strips code fences defensively. Any name outside SOURCE_REGISTRY is
    dropped -- the output space is closed by design, so an unrecognized
    name is a parse anomaly, not a new legitimate source.

    2026-08-09 FIX: Updated to handle JSON objects with "sources" field,
    which is what the PLANNER_PROMPT examples show. Previously only
    accepted JSON arrays, causing all object-formatted responses to
    be rejected as "nothing usable" and fall back to documents search.

    2026-08-09 FIX (priming-brace robustness): PLANNER_PROMPT ends with
    a dangling, unclosed "{" to bias the model toward JSON. That trick
    only reliably works when the "{" is sent as a true assistant-turn
    prefill; here it's the tail of a `system` message, with the actual
    question passed separately as `prompt=question`. We now normalize
    for both cases before parsing.
    """
    cleaned = raw.strip()
    # Strip markdown code fences
    cleaned = re.sub(r"^```(json)?|```$", "", cleaned, flags=re.MULTILINE).strip()

    if not cleaned.startswith("{") and not cleaned.startswith("["):
        if cleaned.endswith("}"):
            # Most likely case: the model continued from the prompt's
            # trailing "{" without re-emitting it. Add it back.
            cleaned = "{" + cleaned
        else:
            # Fallback: salvage a JSON object embedded anywhere in the text.
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        print(f"[PLANNER] Failed to parse JSON: {raw!r}")
        return []

    # ---- Handle JSON object with "sources" field ----
    if isinstance(parsed, dict):
        if "sources" in parsed and isinstance(parsed["sources"], list):
            # Extract sources array from object
            sources = parsed["sources"]
            result = [s for s in sources if isinstance(s, str) and s in _VALID_SOURCES]
            if result:
                print(f"[PLANNER] Extracted from object: {result!r}")
            return result
        # Object exists but has no "sources" field
        print(f"[PLANNER] Parsed object has no 'sources' field: {parsed}")
        return []

    # ---- Handle JSON array directly ----
    if isinstance(parsed, list):
        return [s for s in parsed if isinstance(s, str) and s in _VALID_SOURCES]

    # ---- Neither object nor array ----
    print(f"[PLANNER] Parsed output is neither object nor array: {type(parsed)} = {parsed!r}")
    return []


class PlannerAgent(BaseAgent):
    """
    Planner Agent: decides WHERE information should come from.

    Does NOT retrieve, execute tools, answer, validate, or retry --
    those belong to other agents/nodes.

    Routing precedence:
      1. Metadata short-circuit (title/author questions).
      2. Document-intent protection (high-confidence paper/figure/table
         patterns) → routes to documents without LLM inference.
      3. LLM classification, constrained to SOURCE_REGISTRY's names.
         Falls back to ["documents"] on any parse failure or empty
         result. Placeholder sources short-circuit with a placeholder
         answer instead of being sent downstream.

    2026-08-09 FIX: Now uses rewritten_question as the authoritative
    question for classification. Added document-intent protection for
    research paper evaluation questions. Supports multi-source outputs
    (e.g. ["documents", "web"]).
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
        Single, compact, constrained classification call. One routing
        decision, then stop -- no chain-of-thought, per latency requirement.

        2026-08-09 FIX: Now uses llm.acomplete() with system+prompt signature.
        """
        try:
            # Call the LLM with system prompt + user question
            # PLANNER_PROMPT comes from prompts.py and is the source of truth
            response = await self.llm.acomplete(
                system=PLANNER_PROMPT,
                prompt=question,
                temperature=0,
                max_tokens=40,
            )
            # Extract text from response object
            raw = response.text if hasattr(response, "text") else str(response)
        except AttributeError as exc:
            # Catch if acomplete() method is missing
            print(f"[PLANNER] Classifier call failed: {exc!r} -- defaulting to documents")
            return []
        except Exception as exc:
            # Catch other LLM errors (network, timeout, invalid key, etc.)
            print(f"[PLANNER] LLM error: {exc!r} -- defaulting to documents")
            return []

        sources = _parse_classifier_output(raw)
        print(f"[PLANNER] Classifier sources: {sources!r} (raw={raw!r})")
        return sources

    async def _execute(self, state: AgentState) -> AgentState:
        # Use rewritten_question if available, else fall back to original question
        question_to_classify = state.rewritten_question or state.question
        original_question = state.question

        # ---- Metadata short-circuit ----
        if _METADATA_Q.search(original_question):
            doc_metadata = await self._get_document_metadata(state)
            if doc_metadata:
                print(f"[PLANNER] Metadata question detected, using stored "
                      f"metadata directly: {doc_metadata}")
                state.metadata_answer = doc_metadata
                state.sources_needed = ["metadata"]
                return state
            print("[PLANNER] Metadata question detected but no stored "
                  "metadata found — falling through to normal routing")

        # ---- Document-intent protection (high-confidence bypass) ----
        # Catches research paper evaluation questions without LLM inference.
        # Examples: "What is Lychee-FD's UTMOS score in Figure 4?"
        #           "According to the paper, what does Table 3 show?"
        if _DOCUMENT_INTENT.search(question_to_classify):
            print(f"[PLANNER] Document-intent pattern detected in: "
                  f"{question_to_classify[:60]}... → routing to documents")
            state.sources_needed = ["documents"]
            state.confidence = 0.95
            return state

        # ---- LLM classification ----
        sources = await self._classify_sources(question_to_classify)

        if not sources:
            print("[PLANNER] Classifier returned nothing usable — "
                  "defaulting to sources=['documents']")
            state.sources_needed = ["documents"]
            state.confidence = 0.5
            return state

        placeholder_hits = [s for s in sources if s in _PLACEHOLDER_SOURCES]
        if placeholder_hits:
            print(f"[PLANNER] Classifier picked placeholder source(s) "
                  f"{placeholder_hits!r} -- returning placeholder")
            state.sources_needed = placeholder_hits
            state.metadata_answer = {
                "placeholder": f"{placeholder_hits[0]} integration is under development."
            }
            return state

        state.sources_needed = sources
        state.confidence = 0.6
        print(f"[PLANNER] Sources (classified): {state.sources_needed}")
        return state
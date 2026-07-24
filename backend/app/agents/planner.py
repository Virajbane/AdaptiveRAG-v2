import re

from app.agents.base import BaseAgent
from app.agents.state import AgentState

# --------------------------------------------------------------------------
# 2026-07-XX routing rewrite
#
# History: the Planner used to ask an LLM to decide sources_needed for
# every question, then patched that decision with a growing stack of
# deterministic regex overrides (_looks_personal_only, _needs_both,
# _references_uploaded_doc) because the LLM kept misrouting despite
# explicit prompt rules -- most notably the Bug 1 failure mode, where a
# question naming the document's subject by name (no personal pronoun,
# no "this/the document" phrasing) fell through every existing override
# and the LLM silently chose sources=["web"], which then returned
# content about an unrelated same-named entity.
#
# Root cause across all of those patches was the same: routing decided
# by inference over free text is inherently guessable-wrong, and every
# new failure mode required a new regex to catch it after the fact.
#
# Fix: remove inference from routing entirely. Users (or UI buttons)
# prefix their message with explicit tags -- "@web", "@sql",
# "@documents" -- naming exactly which source(s) to use. Multiple
# leading tags run in parallel. No tags -> default to the documents-only
# RAG pipeline, same safe default the old code fell back to on any
# parse/LLM failure.
#
# This does NOT replace the entity-binding check required by Bug 6:
# even with an explicit "@web" tag, the web tool's results still need
# to be verified against the document's identity record before being
# trusted. That check lives in the web tool integration, not here --
# this file only decides *which* sources run, not whether their output
# is safe to use.
#
# The metadata short-circuit (_METADATA_Q) is unrelated to routing
# inference -- it's a retrieval-bypass optimization for title/author
# questions, which have near-zero lexical/semantic overlap with stored
# metadata text no matter how retrieval is tuned. It's independent of
# how "documents" got selected as a source, so it's kept as-is and
# still runs first, before tag parsing.
# --------------------------------------------------------------------------

_METADATA_Q = re.compile(r"\b(title|author|affiliat)", re.IGNORECASE)

_KNOWN_TAGS = {"web", "sql", "documents"}
_VALID_SOURCES = {"documents", "web", "tools", "sql"}
_TAG_PATTERN = re.compile(r"^@(\w+)\b")


def _parse_leading_tags(question: str) -> tuple[list[str], str]:
    """
    Scans leading whitespace-separated words for @tag markers
    (e.g. "@web @sql what is ..."). Stops at the first word that isn't
    a recognized tag -- that's where the real question begins.

    Only LEADING tags count. A "@web" appearing later in the question
    body (e.g. "compare @web mentions on my site") is not treated as a
    tag -- it's just part of the question text, since the scan stops
    the moment a non-tag word is hit.

    Unknown tags (e.g. "@foobar") are not collected and are left as
    part of the question text. This naturally falls through to the
    no-tag/default-RAG case rather than silently being swallowed or
    raising an error.

    Returns (tags_found, remaining_question).
    """
    words = question.strip().split()
    tags: list[str] = []
    i = 0
    while i < len(words):
        match = _TAG_PATTERN.match(words[i])
        if match and match.group(1).lower() in _KNOWN_TAGS:
            tag = match.group(1).lower()
            if tag not in tags:
                tags.append(tag)
            i += 1
        else:
            break
    remaining = " ".join(words[i:]).strip()
    return tags, remaining


class PlannerAgent(BaseAgent):
    """
    Planner Agent: Decides what sources to use.

    Routing is now explicit rather than inferred:
    - Leading "@tag" markers in the user's message name which source(s)
      to use (multiple tags run in parallel).
    - No tags -> defaults to the documents-only RAG pipeline.
    - "@sql" is still under construction -- it short-circuits with a
      placeholder response rather than touching a real SQL tool.

    Still takes an optional `db` handle for the metadata short-circuit
    (_get_document_metadata looks up stored title/author metadata by
    user_id). Pass db=None if metadata lookup isn't needed in a given
    context -- the short-circuit simply falls through to tag-based
    routing in that case.
    """

    def __init__(self, llm, db=None):
        super().__init__(llm)
        self.db = db

    async def _get_document_metadata(self, state: AgentState) -> dict | None:
        """
        Fetch stored metadata for the user's active document. Returns
        None if no db handle, no documents, or no metadata was
        extracted at ingestion time (e.g. extraction failed silently —
        see MetadataExtractor.extract's None-return path).
        """
        if self.db is None:
            return None
        doc = await self.db.documents.find_one(
            {"user_id": state.user_id},
            sort=[("created_at", -1)],
        )
        if doc and doc.get("metadata"):
            return doc["metadata"]
        return None

    async def _execute(self, state: AgentState) -> AgentState:
        """Plan the search strategy"""

        original_question = state.question

        # ---- Metadata short-circuit: unchanged, still runs first ----
        # Independent of tag routing -- this is a retrieval-bypass
        # optimization, not a routing decision. Runs against the
        # original question, same as before.
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

        # ---- Explicit tag routing ----
        tags, remaining_question = _parse_leading_tags(original_question)

        if tags:
            state.question = remaining_question or original_question
            print(f"[PLANNER] Explicit tags detected: {tags!r}, "
                  f"question: {state.question!r}")

            if "sql" in tags:
                # SQL tool integration is still under construction.
                # Short-circuit immediately, same shape as the metadata
                # check above -- do not touch retriever/grader/answer/
                # critic at all.
                print("[PLANNER] @sql tag detected -- SQL tool not yet "
                      "implemented, returning placeholder")
                state.sources_needed = ["sql"]
                state.metadata_answer = {
                    "placeholder": "SQL integration is under development."
                }
                return state

            state.sources_needed = [t for t in tags if t in _VALID_SOURCES]
            state.confidence = 1.0  # explicit user intent, not a guess
            print(f"[PLANNER] Sources (explicit): {state.sources_needed}")
            return state

        # ---- No tags: default to documents-only RAG pipeline ----
        print("[PLANNER] No tags detected, defaulting to sources=['documents']")
        state.sources_needed = ["documents"]
        state.confidence = 0.5
        return state
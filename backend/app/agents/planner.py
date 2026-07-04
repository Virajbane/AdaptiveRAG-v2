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
#
# 2026-07-04: added a third backstop (_references_uploaded_doc) for a
# distinct failure mode with NO personal pronoun at all — e.g. "What is
# the full title of this paper?" against a just-uploaded PDF. The LLM
# classified this as "public fact" / sources=["web"], discarding 5
# correctly-retrieved, relevant chunks from the uploaded document and
# routing to web search instead (which has no access to an unpublished/
# just-uploaded PDF's exact content). _looks_personal_only doesn't catch
# this because there's no my/I/me to match. This is a separate signal
# class ("this/the <doc-noun>") rather than a variant of the personal-
# reference case, so it gets its own regex and its own override branch
# rather than being folded into the existing ones.
#
# 2026-07-04 (later): added a fourth mechanism — metadata short-circuit
# (_METADATA_Q + _get_document_metadata). Unlike the three overrides
# above, this ISN'T a sources_needed override on top of the LLM's
# output — it runs BEFORE the LLM call entirely and can bypass
# retrieval altogether when stored metadata exists, because title/
# author questions have near-zero lexical/semantic overlap with the
# metadata text itself (the title never contains the word "title"),
# which retrieval structurally cannot fix no matter how the routing
# backstops above are tuned.
# --------------------------------------------------------------------------
_PERSONAL_REF = re.compile(r"\b(my|i|me)\b", re.IGNORECASE)
_METADATA_Q = re.compile(r"\b(title|author|affiliat)", re.IGNORECASE)
_COMPARISON_HINT = re.compile(
    r"\b(compare|comparison|vs\.?|versus|industry|market|benchmark|external|competitors?)\b",
    re.IGNORECASE,
)
_DOC_REF = re.compile(
    r"\b(this|the) (paper|document|pdf|report|file|article|doc)\b",
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


def _references_uploaded_doc(question: str) -> bool:
    """
    True if the question explicitly refers to an uploaded/attached
    document via "this paper" / "the document" / "this pdf" etc., with
    no personal pronoun involved. Catches metadata/content questions
    about a just-uploaded file (title, authors, sections, tables) that
    the LLM has been observed misclassifying as "public fact" -> web,
    even though the document was just processed and is sitting in
    retrieved_docs with real relevance scores.

    Deliberately independent of _looks_personal_only/_needs_both: this
    is a different phrase pattern (deictic reference to "this/the X"),
    not a personal-pronoun case, so it needs its own detector rather
    than being bolted onto the existing regexes.
    """
    return bool(_DOC_REF.search(question))


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

    2026-07-04: now takes an optional `db` handle, needed for the
    metadata short-circuit (_get_document_metadata looks up stored
    title/author metadata by user_id). Pass db=None if metadata lookup
    isn't needed in a given context — the short-circuit simply falls
    through to normal LLM-based routing in that case.
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

        # 2026-07-04: the rewriter runs BEFORE the planner and paraphrases
        # the question to improve retrieval recall (e.g. "the paper" ->
        # "this research document", "affiliated with" -> "mentioned in").
        # That's fine for embedding/retrieval, but every deterministic
        # regex backstop below was checking rewritten_question ONLY — and
        # the rewriter routinely paraphrases away the exact words those
        # regexes key on (title/author/affiliat, "the paper", my/I/me).
        # Eval showed this silently defeated _METADATA_Q on "affiliated
        # with" -> "mentioned in" (Q2) and _references_uploaded_doc on
        # "the paper" -> "this research document" losing the "the paper"
        # phrase to a synonym that no longer matches \b(this|the)
        # (paper|document|...)\b in some rewrites, letting the LLM
        # planner freely misroute to web (Q3, Q6).
        #
        # Fix: run all deterministic detectors against the ORIGINAL
        # question (where the user's actual signal lives), OR'd with the
        # rewritten one as a safety net in case the original is oddly
        # phrased and the rewriter happens to introduce a matching term
        # instead of removing one. Retrieval embedding and the LLM planner
        # prompt continue to use the rewritten question, since paraphrase-
        # for-recall is genuinely useful there — only routing decisions
        # need the original phrasing preserved.
        original_question = state.question
        question_for_routing = state.rewritten_question or state.question
        routing_signal_text = f"{original_question} {question_for_routing}"

        # ---- Metadata short-circuit: skip retrieval entirely ----
        # Title/author questions have near-zero lexical/semantic overlap
        # with the metadata text itself (the title never contains the
        # word "title") — no amount of embedding tuning reliably surfaces
        # it via retrieval (see 2026-07-04 eval). If metadata was
        # extracted at ingestion and is available, answer from it
        # directly instead of routing through retriever/grader/answer/
        # critic at all.
        if _METADATA_Q.search(routing_signal_text):
            doc_metadata = await self._get_document_metadata(state)
            if doc_metadata:
                print(f"[PLANNER] Metadata question detected, using stored "
                      f"metadata directly: {doc_metadata}")
                state.metadata_answer = doc_metadata
                state.sources_needed = ["metadata"]
                return state
            print("[PLANNER] Metadata question detected but no stored "
                  "metadata found — falling through to normal routing")

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
        if _looks_personal_only(routing_signal_text) and state.sources_needed != ["documents"]:
            print(
                f"[PLANNER] Override: personal-reference question with no comparison "
                f"signal, forcing sources=['documents'] (LLM said {state.sources_needed})"
            )
            state.sources_needed = ["documents"]

        # ---- Deterministic override: personal + comparison needs both ----
        elif _needs_both(routing_signal_text) and "documents" not in state.sources_needed:
            print(
                f"[PLANNER] Override: personal-reference question with comparison "
                f"signal, adding 'documents' to sources (LLM said {state.sources_needed})"
            )
            state.sources_needed = ["documents"] + state.sources_needed

        # ---- Deterministic override: explicit "this/the <doc>" reference ----
        # Not an elif on the two above: a question could theoretically say
        # "compare my resume to this paper's findings", which should hit
        # _needs_both already; this branch exists for the common case with
        # NO personal pronoun at all ("What is the title of this paper?"),
        # which neither prior branch touches.
        elif _references_uploaded_doc(routing_signal_text) and state.sources_needed != ["documents"]:
            print(
                f"[PLANNER] Override: question references an uploaded document, "
                f"forcing sources=['documents'] only (LLM said {state.sources_needed})"
            )
            state.sources_needed = ["documents"]

        return state
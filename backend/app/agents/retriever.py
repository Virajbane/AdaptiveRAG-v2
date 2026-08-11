from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.services.retrieval.hybrid_search import HybridSearchEngine
from app.services.retrieval.document_resolver import resolve_document_filter
from bson import ObjectId
from bson.errors import InvalidId
import re
import time

# 2026-07-14 fix: production top_k was a flat 5 for every question, and
# that 5 was ALSO what got shown straight to the LLM. Confirmed via
# eval_rag.py --top-k 12 probe: two known-good chunks
# (retrieval_table_llamaq_st_1, retrieval_table_webq_ss_1) were sitting
# at rank 9 and rank 7 -- correctly ranked, just outside a 5-wide window.
# Table/figure-metric questions are the ones at risk here: they compete
# against many similar-looking numeric rows for the same benchmark, so
# the right one sometimes lands just past the cutoff. Prose questions
# (~92%+ recall) don't have this problem.
#
# Naive fix (raise top_k to 12 for everyone, pass all 12 to the LLM) was
# rejected: more context isn't free. Larger contexts risk (a) "lost in
# the middle" -- the model paying less attention to chunks buried in a
# long context even when they're present, and (b) worse cross-entity
# confusion, i.e. exactly the class of bug already fixed in §2.2 (the
# UTMOS entity-attribution fabrication), which gets MORE likely, not
# less, if the model has more similar-looking numbers in front of it at
# once.
#
# Fix instead: search wide, answer narrow. For metric-style questions,
# pull a bigger CANDIDATE pool (12) from the search engine, but only
# forward the top FINAL_CONTEXT_SIZE of those (reranked) to
# state.retrieved_docs -- the same size context the LLM always saw.
# This works because HybridSearchEngine already reranks every candidate
# it returns (see rerank_score below) -- we're just giving the reranker
# a wider pool to choose the best 5 from, not asking the LLM to read
# more.
# 2026-07-14 fix, part 3: diagnostic on retrieval_prose_backchannel_prob_1
# showed the correct chunk (Sec 11.2, "User Backchannels: ...") sitting at
# rank 7 -- same ranking-window bug as the table/figure cases above, just
# for a question phrased with "probability" instead of "accuracy/score/
# rate", which the original keyword list didn't cover. Confirmed the
# question was never classified as metric-style at all (query type:
# default, candidates searched: 5), so no widening ever triggered.
# Widening the trigger list rather than trying to enumerate every
# possible numeric-value phrasing exhaustively -- same tradeoff as
# before: heuristic and inspectable, not exhaustive.
_METRIC_KEYWORDS = re.compile(
    r"\b(accuracy|score|rate|gflops?|utmos|wer|srr|sir|tor|mrr|"
    r"probability|percentage|ratio|threshold|"
    r"table\s*\d|figure\s*\d)\b",
    re.IGNORECASE,
)

DEFAULT_CANDIDATE_K = 5    # unchanged behavior for ordinary prose questions
METRIC_CANDIDATE_K = 12    # wider net for table/figure-metric questions --
                            # matches the value confirmed to surface both
                            # known ranking-window misses in the eval probe
FINAL_CONTEXT_SIZE = 5     # what actually reaches the LLM, regardless of
                            # how wide the candidate search was. Keeps
                            # context size (and confusion risk) constant.


def _is_metric_style_query(question: str) -> bool:
    """True if the question is asking for a specific numeric metric value
    (accuracy/score/rate/etc, often tied to a benchmark table or figure).
    These compete against many visually/textually similar numeric rows,
    so they benefit from a wider CANDIDATE search -- but not a wider
    final context; see FINAL_CONTEXT_SIZE above."""
    return bool(_METRIC_KEYWORDS.search(question))


# 2026-07-14 fix, part 2: diagnostic (test_retriever_patch.py, full
# 12-candidate dump) showed the correct chunk WAS being retrieved for
# both llamaq_st_1 and webq_ss_1 -- it just scored far below prose
# chunks that merely *mention* the table narratively. Confirmed directly:
#   score=0.5385 | Row [Lychee-FD (Ours) w/o Sem -]: ...LlamaQ.S->T=73.7...
#   score=0.9978 | [Section 4.4] Result. As presented in Table...
# The reranker (trained on natural language) systematically underrates
# the stiff "Row [X]: Field=value, Field=value" chunk format compared to
# ordinary prose that just talks ABOUT the table -- so table-row chunks
# get crowded out of the final narrowing even when they're the single
# chunk that actually contains the number asked for.
#
# This is a stopgap: it guarantees any retrieved table-row chunk survives
# into the final context for metric-style questions, regardless of its
# rerank score. The real fix is upstream (rewrite table rows into natural
# sentences at ingestion time so the reranker judges them fairly) -- this
# patch just stops the current row-chunk format from being penalized
# until that ingestion-side fix lands.
_TABLE_ROW_PATTERN = re.compile(r"^\s*Row\s*\[", re.IGNORECASE)

# 2026-08-06 FIX #3 follow-up: the guarantee above ("any table-row chunk
# survives, regardless of score") had a blind spot -- it guarantees EVERY
# row chunk, not just the one for the entity actually asked about. If
# retrieval surfaces both "Row [Lychee-FD]: UTMOS=4.50" and
# "Row [Moshi]: UTMOS=4.44" (adjacent rows in the same benchmark table,
# so both are plausible retrieval hits), the old logic force-included
# BOTH into the final 5-chunk context -- which is exactly the
# entity-attribution mix-up traced in the eval (fab_01/fab_02: "4.50
# appears in context but is not attributed to the entity asked about").
# hybrid_search.py's filter_comparison_table_rows() only capped REPEATS
# of the same entity, so two different entities each appearing once each
# sailed through untouched, and this guarantee step then blindly forwarded
# whatever row chunks it was handed. Fix: only guarantee row chunks whose
# entity is actually named in the question (by canonical name, stripping
# qualifiers like "(Ours)"/"w/o Sem"), unless the question is an explicit
# comparison. If no entity can be identified in the question at all, fall
# back to the original behavior (guarantee whatever row chunks came back)
# rather than dropping everything blind.
_ROW_ENTITY_PATTERN = re.compile(r"^\s*Row\s*\[(.*?)\]", re.IGNORECASE)

_COMPARISON_KEYWORDS = {
    "compare", "vs", "versus", "better", "worse", "difference",
    "outperform", "beat", "surpass", "how much", "which is",
    "contrast", "similar", "differ", "both",
}


def _row_entity(doc: dict) -> str | None:
    m = _ROW_ENTITY_PATTERN.match(doc.get("text", ""))
    return m.group(1) if m else None


def _canonical_entity_name(entity: str) -> str:
    """Strip qualifiers like "(Ours)", "w/o Sem", trailing " -" so
    "Lychee-FD (Ours) w/o Sem -" and "Lychee-FD" match as the same
    entity when checked against the question text."""
    canonical = re.split(r"\(| w/o |w/o ", entity, maxsplit=1)[0]
    return canonical.strip(" -").strip()


def _entity_matches_question(entity: str, question: str) -> bool:
    canonical = _canonical_entity_name(entity).lower()
    return bool(canonical) and canonical in question.lower()


def _is_comparison_query(question: str) -> bool:
    q_lower = question.lower()
    return (
        any(kw in q_lower for kw in _COMPARISON_KEYWORDS)
        or question.count(" and ") + question.count(" vs ") > 0
    )


def _is_table_row_chunk(doc: dict) -> bool:
    return bool(_TABLE_ROW_PATTERN.match(doc.get("text", "")))


# 2026-08-10 FIX (Test 8 — Figure grounding): Extend the row-chunk survival
# guarantee to figure chunks. Tables are extracted as "Row [Entity]: Field=value..."
# and figures are extracted as "[Section: ...]\nFigure: ...\n<description>".
# Both are structured data that the reranker (trained on prose) underrates
# compared to prose mentions. The fix for tables (2026-07-14) guaranteed row
# chunks survive metric-query narrowing. Figures need the same guarantee.
_FIGURE_CHUNK_PATTERN = re.compile(r"^\s*Figure:", re.MULTILINE)


def _is_figure_chunk(doc: dict) -> bool:
    """True if this chunk is a figure/image with extracted description."""
    return bool(_FIGURE_CHUNK_PATTERN.search(doc.get("text", "")))


# --- Range/trend annotation for numeric value extraction ---------------
_RANGE_RE = re.compile(
    r'(?:from|rising from|surging from|grew from|increas\w* from)\s+'
    r'([\d.]+)\s*(?:%|)\s*to\s+([\d.]+)\s*(?:%|)\s*(?:as|when|at|over)?\s*([^.,]*)',
    re.IGNORECASE,
)


def _annotate_ranges(text: str) -> str:
    """Attach explicit annotations to range/trend statements so AnswerAgent
    can read endpoint values directly without inferring them."""
    matches = _RANGE_RE.findall(text)
    if not matches:
        return text
    annotations = []
    for start, end, condition in matches:
        condition = condition.strip()
        annotations.append(
            f"[Note: stated explicitly in the source — the value is {start} "
            f"at the starting/lowest point (i.e. before {condition} has occurred, "
            f"often meaning zero), and rises to {end} at the point where {condition}.]"
        )
    return text + "\n" + "\n".join(annotations)
# -------------------------------------------------------------------------


def _rerank_key(doc: dict) -> float:
    """Sort key for narrowing candidates back down. Prefers rerank_score
    (the BGE cross-encoder score, which is what actually determines real
    ranking quality) and falls back to combined_score (RRF fusion score)
    only if reranking didn't run -- same precedence already used by the
    debug-logging block below, kept consistent here."""
    return doc.get("rerank_score", doc.get("combined_score", 0.0))


class RetrieverAgent(BaseAgent):
    """
    Retriever Agent: Searches documents only.

    Responsibilities:
    - Resolves whether the question names a specific uploaded document
    - Performs hybrid search (vector + keyword BM25), scoped to that
      document if confidently resolved
    - Retrieves top documents using configurable candidate-K expansion
      for metric-style questions (wide search → narrow final context)
    - Executes reranking to refine candidate ranking
    - Applies entity-aware table-row and figure protection for benchmark metrics
    - Attaches source filenames for display in answer/sources card
    - Annotates numeric ranges so AnswerAgent can read endpoint values

    CRITICAL 2026-08-09 FIX: Uses canonical question
    `state.rewritten_question or state.question`
    Same query as PlannerAgent, guarantees consistent routing.

    NOTE: Runs in PARALLEL with PlannerAgent. It always searches; the
    orchestrator decides whether to use results based on planner's
    sources_needed AFTER both complete.

    DOES NOT:
    - Change sources_needed
    - Decide web/tool routing
    - Answer questions
    - Grade answers
    - Retry itself
    """

    def __init__(self, llm=None, db=None):
        super().__init__(llm)
        self.db = db   # Mongo handle, needed to look up filenames for
                        # document-scoped retrieval AND for the filename
                        # enrichment step below. May be None if not wired
                        # through — both degrade gracefully.

    async def _attach_filenames(self, results: list[dict]) -> None:
        """
        Batch-look up filenames for every unique doc_id in the result set
        and attach them in place as result['filename']. One query total,
        regardless of how many chunks were retrieved.

        Mutates `results` in place; no-ops gracefully if self.db is
        unavailable or no chunks were returned.
        """
        if self.db is None or not results:
            return

        unique_doc_ids = {doc["doc_id"] for doc in results}

        object_ids = []
        for doc_id in unique_doc_ids:
            try:
                object_ids.append(ObjectId(doc_id))
            except (InvalidId, TypeError):
                # doc_id wasn't a valid ObjectId string — skip rather than
                # crash retrieval over a metadata-enrichment problem.
                print(f"[RETRIEVER] Skipping filename lookup for malformed doc_id: {doc_id!r}")

        if not object_ids:
            return

        cursor = self.db.documents.find(
            {"_id": {"$in": object_ids}},
            {"filename": 1},
        )
        filename_map = {str(doc["_id"]): doc["filename"] async for doc in cursor}

        for doc in results:
            doc["filename"] = filename_map.get(doc["doc_id"], "Unknown document")

    async def _execute(self, state: AgentState) -> AgentState:
        """Search for relevant documents — always runs, results used conditionally."""

        try:
            start_time = time.time()
            
            # 2026-08-09 FIX: Use canonical question (same as PlannerAgent)
            question = state.rewritten_question or state.question

            document_id = await resolve_document_filter(question, state.user_id, self.db)

            is_metric_query = _is_metric_style_query(question)
            candidate_k = METRIC_CANDIDATE_K if is_metric_query else DEFAULT_CANDIDATE_K

            search_engine = HybridSearchEngine()
            candidates = await search_engine.search(
                query=question,
                user_id=state.user_id,
                top_k=candidate_k,
                document_id=document_id,
            )

            # Narrow back down to a constant-size final context. For
            # ordinary prose questions candidate_k == FINAL_CONTEXT_SIZE
            # already, so this is a no-op slice.
            #
            # For metric-style questions: table-row AND figure chunks are
            # guaranteed a spot regardless of rerank score (confirmed via
            # diagnostic that the reranker underrates both structured formats
            # even when they're the chunks with the actual answer). Remaining
            # slots are filled by rerank score as before.
            #
            # 2026-08-06 FIX #3 follow-up: the guarantee is now scoped to
            # row chunks whose entity is actually named in the question
            # (unless it's an explicit comparison question, or no entity
            # could be identified in the question at all). This stops the
            # cross-entity mix-up (e.g., "What is Moshi's UTMOS?" must not
            # return Lychee-FD's value).
            #
            # 2026-08-10 FIX: Extended this guarantee to figure chunks as
            # well, using the same entity-matching logic.
            if is_metric_query and len(candidates) > FINAL_CONTEXT_SIZE:
                is_comparison = _is_comparison_query(question)
                
                # Gather both table rows AND figures for the survival guarantee
                all_row_chunks = [c for c in candidates if _is_table_row_chunk(c)]
                all_figure_chunks = [c for c in candidates if _is_figure_chunk(c)]
                all_structured_chunks = all_row_chunks + all_figure_chunks

                entities_matched = {
                    _row_entity(c) for c in all_row_chunks
                    if _row_entity(c) and _entity_matches_question(_row_entity(c), question)
                }

                if is_comparison or not entities_matched:
                    # Can't (or shouldn't) narrow by entity -- keep old
                    # behavior: guarantee every row and figure chunk that
                    # survived retrieval, regardless of entity.
                    structured_chunks = all_structured_chunks
                else:
                    # Only keep row chunks for matched entities
                    row_chunks = [
                        c for c in all_row_chunks
                        if _row_entity(c) in entities_matched
                    ]
                    dropped = len(all_row_chunks) - len(row_chunks)
                    if dropped:
                        print(
                            f"[RETRIEVER] Dropped {dropped} row chunk(s) for "
                            f"entities not asked about (question refers to "
                            f"{sorted(entities_matched)})"
                        )
                    # Keep all figure chunks (no entity info in figures, can't filter)
                    structured_chunks = row_chunks + all_figure_chunks

                other_chunks = sorted(
                    (c for c in candidates if not (_is_table_row_chunk(c) or _is_figure_chunk(c))),
                    key=_rerank_key, reverse=True,
                )
                remaining_slots = max(FINAL_CONTEXT_SIZE - len(structured_chunks), 0)
                results = structured_chunks + other_chunks[:remaining_slots]
                results = sorted(results, key=_rerank_key, reverse=True)
            else:
                results = candidates

            # Annotate range/trend statements so AnswerAgent can read
            # endpoint values directly instead of having to infer them
            for doc in results:
                doc["text"] = _annotate_ranges(doc["text"])

            await self._attach_filenames(results)

            state.retrieved_docs = results
            state.search_time_ms = (time.time() - start_time) * 1000

            print(f"[RETRIEVER] Canonical question: {question[:60]}...")
            print(f"[RETRIEVER] document_id filter: {document_id}")
            print(
                f"[RETRIEVER] query type: {'metric-style' if is_metric_query else 'default'} "
                f"| candidates searched: {candidate_k} | forwarded to LLM: {len(results)}"
            )
            print(f"[RETRIEVER] Found {len(results)} documents")
            for i, doc in enumerate(results, 1):
                if 'rerank_score' in doc:
                    score_label = "rerank"
                    score_value = doc['rerank_score']
                else:
                    score_label = "rrf (no rerank)"
                    score_value = doc.get('combined_score', 0.0)

                print(f"  {i}. [{score_label}] Score: {score_value:.4f} | {doc.get('filename', '?')} | {doc['text'][:60]}...")

        except Exception as e:
            import traceback
            traceback.print_exc()
            state.error = f"Retriever error: {str(e)}"

        return state
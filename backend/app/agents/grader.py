"""
backend/app/agents/grader.py

GraderAgent: pre-retrieval relevance filter. Runs after retrieval, before
the answer is generated, so AnswerAgent never sees chunks that are
essentially noise relative to the best match.

Design notes:
- No LLM call. Reuses the score already computed during retrieval
  (rerank_score from the BGE cross-encoder, or combined_score/RRF if
  reranking didn't run) — there's no need to re-derive relevance via a
  separate model call, and avoiding it keeps this node ~free on
  CPU-bound hardware (see graph.py's per-node timing notes).
- RELATIVE threshold, not absolute (for filtering WITHIN a batch). Rerank/RRF
  score ranges vary a lot per query (a strong match for one question might
  score 0.95, for another 0.3, depending on phrasing/embedding behavior) — a
  single fixed cutoff would either be too strict for weak-but-correct queries
  or too loose for strong ones. Instead, a chunk is dropped if its score
  is below `RELATIVE_FLOOR` (default 2%) of the TOP chunk's score in
  THIS result set. This adapts per-query automatically.

  Kept deliberately forgiving (2%, not a steeper cutoff): the grader
  can only see score gaps, not which chunk actually contains the
  answer. A steep floor risks dropping a genuinely relevant chunk that
  happens to rank 2nd/3rd/4th with a moderate score, just because the
  top chunk scored unusually high on something else. A 2% floor only
  removes chunks that are orders of magnitude weaker than the best
  match — true noise, not "second-best but still relevant" chunks.
- Always keeps at least MIN_KEEP chunks (default 3), even if every
  other chunk technically fails the relative threshold — so multi-hop
  or summary-style questions, where the answer can legitimately span
  several moderately-scored chunks rather than one dominant one, still
  reach AnswerAgent with enough context instead of being starved down
  to a single chunk.
- Mutates state.retrieved_docs in place (filters it down) so no other
  node needs to change — AnswerAgent, the source-card builder, etc. all
  keep reading state.retrieved_docs exactly as before.

2026-07-04 addition — ABSOLUTE floor (soft signal, not a gate):
  Eval against a real arXiv paper surfaced a gap the relative floor can't
  catch: a query ("What is the full title of this paper?") where EVERY
  retrieved chunk was weak in absolute terms (top_score=0.0970, next
  best 0.0203, decaying to 0.0060) — all pulled from the References
  section rather than the actual title on page 1. The relative floor
  correctly keeps all 5 chunks (they're all within 2% of *each other*
  relative to the top), because relative filtering has no way to know
  the whole batch is weak — it can only compare chunks to each other,
  not to some notion of "good enough to actually answer with."

  ABSOLUTE_FLOOR is a second, independent check: if the TOP chunk scores
  below this floor, the batch is flagged as low-confidence by prepending
  a short, explicit warning to the top chunk's own text (mutating a copy
  of a real, already-cited chunk, not a synthetic fake one, keeps the
  "Sources Cited" UI accurate — it's still a genuine retrieved source,
  just a weak one).

  0.05 is a starting estimate for the BGE cross-encoder's typical output
  range observed in this pipeline's logs — tune based on continued eval
  observations rather than treating it as precisely calibrated.

2026-07-25 fix — ABSOLUTE_FLOOR is no longer a hard gate before generation:
  QA against the resume test doc surfaced the real cost of the previous
  hard-stop: "What CGPA did Viraj get?" (top_score=0.0038, explicit answer
  "CGPA 8.1" present in the doc), "What certifications does Viraj hold?"
  (top_score=0.0038), and "Which project was built most recently?"
  (top_score=0.0012) were all rejected before AnswerAgent ever ran. Root
  cause: short factual chunks (a single number, a header, a date) score
  low against a conversationally-phrased question not because the answer
  is absent, but because the embedding model doesn't reward that kind of
  chunk highly — cosine/rerank similarity can't distinguish "no answer
  exists" from "answer exists but the chunk is naturally low-signal."

  The floor no longer skips generation. Below ABSOLUTE_FLOOR, the batch
  is annotated with LOW_CONFIDENCE_NOTE (as it already was) and passed
  through to AnswerAgent like any other batch, which is the layer with
  the actual context to make an absence call. The real "is this
  answerable" decision has moved to AnswerAgent's grounding instruction
  + CriticAgent's grounding check downstream — that combination has
  already proven reliable in eval logs (correctly declined an
  unverifiable number in a separate case) and is a better signal than
  chunk-level cosine similarity for short factual content. This costs a
  generate() + critic round trip on genuinely-empty retrievals that the
  old hard-stop would have short-circuited faster, but that's the right
  trade: a few extra seconds on true negatives beats confidently
  discarding true positives.

  state.low_confidence replaces the generation-skipping use of
  state.retrieval_rejected. retrieval_rejected is now reserved for the
  only case where there is truly nothing to forward (no chunks at all).
  NOTE: state.low_confidence needs to exist as a field on AgentState —
  add it if it isn't there yet (bool, default False). Verify the
  "[GRADER] Low confidence" log line appears during testing.

2026-07-04 addition — BIDIRECTIONAL source override:
  PDF eval logs showed the Planner defaulting to including 'web' in
  sources_needed for the large majority of document-grounded questions
  (introduction contributions, entity-extraction techniques, ASR tool
  names — all purely in-corpus facts). The existing override only ever
  ADDED 'documents' when the Planner said web-only but doc confidence
  was high — it never REMOVED 'web' when doc confidence was already
  strong enough on its own. Effect: most answers were synthesized from
  3 doc chunks + 5 unrelated web results together, which (a) doubled
  generation latency (8-source context vs 3), (b) introduced irrelevant
  noise into AnswerAgent's context (e.g. generic "how call centers
  route audio" results for a question about a specific paper's
  methodology), and (c) is the direct mechanism behind the open
  "prevent web contamination when strong document evidence exists"
  TODO.

  Now symmetric: high doc confidence both adds 'documents' if missing
  AND drops 'web' if present, so a strong in-corpus match routes to
  documents-only. DOC_CONFIDENCE_OVERRIDE (0.5) reuses the same value
  already observed working correctly in production logs for the
  add-side of this override — not newly invented.

  NOTE: this assumes state.sources_needed is a mutable list attribute
  on AgentState, set upstream by PlannerAgent. If your actual field
  name/type differs, this block needs to be updated to match — it will
  silently no-op (AttributeError swallowed via getattr) rather than
  crash, so verify the [GRADER] override log line actually appears
  during testing rather than assuming it's active.

2026-07-05 addition — two-tier override:
  A single 0.5 threshold for BOTH "should documents be included at all"
  and "should web be dropped" turned out to be doing two different jobs
  with one number. Eval logs surfaced two confirmed wrong answers
  (correct model: GPT-2, answered: "DBRX, Llama 2 70B, Mistral Large";
  correct tool: WhisperX, answered: "Reverb models from Rev") where the
  real, relevant document chunk scored 0.30 and 0.37 - genuinely
  on-topic, well below noise, but under 0.5 - so it was never added to
  sources_needed at all, and web fully displaced it. ADD_DOCUMENTS_
  THRESHOLD is a much lower bar (roughly 3x the Grader's own
  ABSOLUTE_FLOOR of 0.05) for "this is a real signal worth including
  alongside web," while DOC_CONFIDENCE_OVERRIDE stays the higher bar
  for "this is strong enough that web can be dropped entirely" - the
  two failure modes (missing content vs. contaminating noise) need
  different bars, not one compromise value.
"""

from app.agents.base import BaseAgent
from app.agents.state import AgentState

RELATIVE_FLOOR = 0.02   # keep chunks scoring >= 2% of the top chunk's score
MIN_KEEP = 3             # always keep at least this many chunks if any exist
ABSOLUTE_FLOOR = 0.05    # flag the whole batch as low-confidence below this (soft signal, not a gate)
DOC_CONFIDENCE_OVERRIDE = 0.5  # top score at/above this = documents alone are trustworthy
ADD_DOCUMENTS_THRESHOLD = 0.15  # top score at/above this = real signal worth including alongside web

LOW_CONFIDENCE_NOTE = (
    "[RETRIEVAL NOTE: This chunk scored very low relevance for the "
    "question asked and may not actually answer it. If it doesn't "
    "clearly and directly answer the question, say the document does "
    "not appear to contain that information rather than guessing.] "
)


def _doc_score(doc: dict) -> float:
    """Same scoring precedence used everywhere else in the pipeline:
    rerank_score (BGE cross-encoder) if reranking ran, else
    combined_score (RRF fusion)."""
    return doc.get("rerank_score", doc.get("combined_score", 0.0))


class GraderAgent(BaseAgent):
    """
    Drops retrieved chunks that are far weaker than the best match for
    this query, before they reach AnswerAgent. See module docstring.

    Does NOT decide "answerable or not" anymore — that call belongs to
    AnswerAgent (grounded generation) + CriticAgent (grounding check),
    which have the actual context to make it. This agent only trims
    noise and flags low-confidence batches for those downstream agents.
    """

    async def _execute(self, state: AgentState) -> AgentState:
        docs = state.retrieved_docs
        state.low_confidence = False

        if not docs:
            print("[GRADER] No retrieved docs to grade — skipping")
            state.retrieval_rejected = True
            return state

        scored = [(doc, _doc_score(doc)) for doc in docs]
        top_score = max(score for _, score in scored)

        if top_score <= 0:
            print("[GRADER] Top score is 0 — no relative filtering possible, "
                  "flagging low-confidence and forwarding all chunks as-is")
            state.low_confidence = True
            state.retrieval_rejected = False
            return state

        threshold = top_score * RELATIVE_FLOOR
        kept = [doc for doc, score in scored if score >= threshold]

        if len(kept) < MIN_KEEP:
            kept = [doc for doc, _ in sorted(scored, key=lambda pair: pair[1], reverse=True)[:MIN_KEEP]]

        dropped_count = len(docs) - len(kept)
        if dropped_count > 0:
            print(f"[GRADER] Dropped {dropped_count}/{len(docs)} chunks below "
                  f"{threshold:.4f} (top_score={top_score:.4f}, floor={RELATIVE_FLOOR})")
        else:
            print(f"[GRADER] Kept all {len(docs)} chunks (top_score={top_score:.4f})")

        # ── Bidirectional source override ──
        sources = getattr(state, "sources_needed", None)
        if sources is not None and top_score >= ADD_DOCUMENTS_THRESHOLD:
            changed = False
            if "documents" not in sources:
                sources.append("documents")
                changed = True

            if top_score >= DOC_CONFIDENCE_OVERRIDE and "web" in sources:
                sources.remove("web")
                changed = True

            if changed:
                print(f"[GRADER] Override: doc confidence top_score={top_score:.4f} "
                      f"(add>={ADD_DOCUMENTS_THRESHOLD}, drop_web>={DOC_CONFIDENCE_OVERRIDE}) "
                      f"— sources_needed set to {sources}")
            state.sources_needed = sources

        # ── Absolute floor check — soft signal only, no longer skips generation ──
        # Short factual chunks (a number, a header, a date) can legitimately
        # score below ABSOLUTE_FLOOR while still containing the correct
        # answer, so this no longer short-circuits to "not found." Instead:
        # annotate the top kept chunk with a warning and let AnswerAgent /
        # CriticAgent make the real absence call with actual context.
        if top_score < ABSOLUTE_FLOOR:
            print(f"[GRADER] Top score {top_score:.4f} below absolute floor "
                  f"{ABSOLUTE_FLOOR} — flagging low-confidence, forwarding to AnswerAgent")
            state.low_confidence = True
            annotated_top = dict(kept[0])
            annotated_top["text"] = LOW_CONFIDENCE_NOTE + annotated_top.get("text", "")
            kept = [annotated_top] + kept[1:]

        state.retrieval_rejected = False
        state.retrieved_docs = kept
        return state
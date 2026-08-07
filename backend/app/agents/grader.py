"""
backend/app/agents/grader.py (REFACTORED)

GraderAgent: Retrieval Quality Assessment

Simplified responsibility: evaluate whether retrieved chunks represent 
usable context for answering, and adjust routing if retrieval quality 
is unusually strong.

Design notes:
- NO filtering. Retriever already reranked and narrowed to FINAL_CONTEXT_SIZE (5).
  AnswerAgent later selects only top 3 anyway, so filtering docs 4-5 is redundant.
- NO text mutation. Grader assesses; AnswerAgent (the generation layer) decides 
  how to act on that assessment.
- LOW_CONFIDENCE flag: Set if top_score < ABSOLUTE_FLOOR. Signals downstream 
  that retrieval was weak and generation should be cautious.
- BIDIRECTIONAL SOURCE OVERRIDE: If retrieval quality is high, adjust routing 
  to drop unnecessary web searches. This is pragmatically a routing concern, 
  but motivated purely by retrieval assessment, so lives here for simplicity.
- NO LLM CALLS: Reuses scores already computed during retrieval.

2026-07-04 ABSOLUTE FLOOR:
  If top_score < 0.05, the batch is flagged as low-confidence (soft signal, 
  not a gate). Short factual chunks can score low while still containing 
  correct answers, so generation must still run — AnswerAgent and CriticAgent 
  have the context to decide. Grader only flags.

2026-07-25 FIX:
  ABSOLUTE_FLOOR no longer skips generation. It sets a flag. AnswerAgent 
  decides whether to proceed or decline based on actual generation context.

2026-08-06 REFACTOR:
  Removed RELATIVE_FLOOR filtering (redundant; AnswerAgent's top-3 selection 
  is sufficient). Removed text mutation (LOW_CONFIDENCE_NOTE prepending — that 
  decision belongs to AnswerAgent). Kept source override (pragmatic despite 
  technically being a routing concern; minimal and justified by retrieval QA).
  Kept low_confidence flagging (essential signal to downstream).
"""

from app.agents.base import BaseAgent
from app.agents.state import AgentState

# Soft threshold: if top score falls below this, flag for downstream caution
ABSOLUTE_FLOOR = 0.05

# Bidirectional source override: high doc confidence can bypass web
DOC_CONFIDENCE_OVERRIDE = 0.5      # drop web entirely
ADD_DOCUMENTS_THRESHOLD = 0.15     # add documents even if not planned


def _doc_score(doc: dict) -> float:
    """
    Scoring precedence (same as everywhere else in pipeline):
    rerank_score (BGE cross-encoder) if available, else combined_score (RRF).
    """
    return doc.get("rerank_score", doc.get("combined_score", 0.0))


class GraderAgent(BaseAgent):
    """
    Assesses whether retrieved chunks represent usable context.
    
    Does NOT:
      - Filter chunks (Retriever already ranked; AnswerAgent selects top 3)
      - Mutate chunk text (AnswerAgent is the generation layer)
      - Make final answerable/not-answerable calls (AnswerAgent + CriticAgent do)
    
    Does:
      - Flag low-confidence batches (set state.low_confidence if top_score weak)
      - Override routing if retrieval is strong (drop unnecessary web)
      - Log quality assessment for debugging
    """

    async def _execute(self, state: AgentState) -> AgentState:
        docs = state.retrieved_docs
        state.low_confidence = False

        # ── No docs at all ──
        if not docs:
            print("[GRADER] No retrieved docs — skipping assessment")
            state.retrieval_rejected = True
            return state

        # ── Assess quality ──
        top_score = _doc_score(docs[0])

        if top_score <= 0:
            print("[GRADER] Top score is 0 — no retrieval signal")
            state.low_confidence = True
            state.retrieval_rejected = False
            return state

        # ── Absolute floor check: soft signal only ──
        if top_score < ABSOLUTE_FLOOR:
            print(f"[GRADER] Top score {top_score:.4f} below absolute floor "
                  f"{ABSOLUTE_FLOOR} — flagging low-confidence")
            state.low_confidence = True
        else:
            print(f"[GRADER] Top score {top_score:.4f} — confidence OK")

        # ── Bidirectional source override ──
        # Use retrieval quality to optimize routing: if docs are strong enough,
        # don't waste context budget on web results.
        # 
        # This is technically a routing concern, but it's:
        # (a) motivated purely by retrieval assessment (top_score),
        # (b) minimal and deterministic,
        # (c) needed here because Planner runs in parallel with Retriever.
        #
        # A more "pure" design would extract this to a separate Router node,
        # but that adds complexity for a small win. Keeping it here for now.
        sources = getattr(state, "sources_needed", None)
        if sources is not None and top_score >= ADD_DOCUMENTS_THRESHOLD:
            changed = False

            # Add 'documents' if not already present
            if "documents" not in sources:
                sources.append("documents")
                changed = True

            # Drop 'web' if doc confidence is very high (sufficient on its own)
            if top_score >= DOC_CONFIDENCE_OVERRIDE and "web" in sources:
                sources.remove("web")
                changed = True

            if changed:
                print(f"[GRADER] Override: top_score={top_score:.4f} "
                      f"(add_docs>={ADD_DOCUMENTS_THRESHOLD}, "
                      f"drop_web>={DOC_CONFIDENCE_OVERRIDE}) "
                      f"— sources_needed = {sources}")
            state.sources_needed = sources

        # ── No filtering, no text mutation ──
        # Retriever already reranked and narrowed to FINAL_CONTEXT_SIZE.
        # Filtering docs 4-5 here doesn't help because AnswerAgent only uses top 3.
        # Let AnswerAgent select what it needs.
        state.retrieval_rejected = False
        return state
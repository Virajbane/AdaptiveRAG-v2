"""
backend/app/agents/grader.py (REFACTORED)

GraderAgent: Retrieval Quality Assessment

Simplified responsibility: evaluate whether retrieved chunks represent 
usable context for answering.

Design notes:
- ASSESSMENT ONLY: Grader evaluates retrieval quality and sets flags.
  It does NOT modify routing (sources_needed). Routing is Planner's 
  exclusive responsibility.
- NO FILTERING: Retriever already reranked and narrowed to FINAL_CONTEXT_SIZE (5).
  AnswerAgent later selects only top 3 anyway, so filtering docs 4-5 is redundant.
- NO TEXT MUTATION: Grader assesses; AnswerAgent (the generation layer) decides 
  how to act on that assessment.
- LOW_CONFIDENCE FLAG: Set if top_score < ABSOLUTE_FLOOR. Soft signal only.
  Signals downstream that retrieval was weak. AnswerAgent and CriticAgent 
  have the context to decide whether generation can proceed.
- NO SOURCE MUTATION: Grader no longer modifies sources_needed based on 
  retrieval quality. That responsibility belongs exclusively to PlannerAgent.
- NO LLM CALLS: Reuses scores already computed during retrieval.

2026-07-04 ABSOLUTE_FLOOR:
  If top_score < 0.05, the batch is flagged as low_confidence (soft signal, 
  not a gate). Short factual chunks can score low while still containing 
  correct answers, so generation must still run — AnswerAgent and CriticAgent 
  have the context to decide. Grader only flags.

2026-08-09 FIX: REMOVED SOURCE MUTATION
  Removed the bidirectional source override (ADD_DOCUMENTS_THRESHOLD, 
  DOC_CONFIDENCE_OVERRIDE) that was modifying sources_needed based on 
  retrieval quality. Routing decisions are Planner's responsibility exclusively.
  Grader assesses only; it does not route.
"""

from app.agents.base import BaseAgent
from app.agents.state import AgentState

# Soft threshold: if top score falls below this, flag for downstream caution
ABSOLUTE_FLOOR = 0.05


def _doc_score(doc: dict) -> float:
    """
    Scoring precedence (same as everywhere else in pipeline):
    rerank_score (BGE cross-encoder) if available, else combined_score (RRF).
    """
    return doc.get("rerank_score", doc.get("combined_score", 0.0))


class GraderAgent(BaseAgent):
    """
    Assesses retrieval quality deterministically.
    
    Writes:
      - low_confidence: True if top_score < ABSOLUTE_FLOOR (soft signal)
      - retrieval_rejected: True if no docs retrieved at all
    
    Does NOT write:
      - sources_needed (Planner's exclusive responsibility)
      - retrieved_docs (only reads, never filters or mutates)
    
    Does NOT:
      - Filter chunks (Retriever already ranked; AnswerAgent selects top 3)
      - Mutate chunk text (AnswerAgent is the generation layer)
      - Make final answerable/not-answerable calls (AnswerAgent + CriticAgent do)
      - Modify routing (that's Planner's job)
    
    Does:
      - Flag low-confidence batches (set state.low_confidence if top_score weak)
      - Log quality assessment for debugging
    """

    async def _execute(self, state: AgentState) -> AgentState:
        docs = state.retrieved_docs
        state.low_confidence = False
        state.retrieval_rejected = False

        # ── No docs at all ──
        if not docs:
            print("[GRADER] No retrieved docs — marking retrieval_rejected")
            state.retrieval_rejected = True
            return state

        # ── Assess quality ──
        top_score = _doc_score(docs[0])

        if top_score <= 0:
            print("[GRADER] Top score is 0 — no retrieval signal, flagging low_confidence")
            state.low_confidence = True
            return state

        # ── Absolute floor check: soft signal only ──
        if top_score < ABSOLUTE_FLOOR:
            print(f"[GRADER] Top score {top_score:.4f} below absolute floor "
                  f"{ABSOLUTE_FLOOR} — flagging low_confidence")
            state.low_confidence = True
        else:
            print(f"[GRADER] Top score {top_score:.4f} — confidence OK")

        # ── Retrieval assessment complete ──
        # No filtering, no text mutation, no source mutation.
        # Let AnswerAgent select what it needs based on context.
        return state
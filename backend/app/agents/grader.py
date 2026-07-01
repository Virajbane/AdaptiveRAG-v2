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
- RELATIVE threshold, not absolute. Rerank/RRF score ranges vary a lot
  per query (a strong match for one question might score 0.95, for
  another 0.3, depending on phrasing/embedding behavior) — a single
  fixed cutoff would either be too strict for weak-but-correct queries
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
"""

from app.agents.base import BaseAgent
from app.agents.state import AgentState

RELATIVE_FLOOR = 0.02   # keep chunks scoring >= 2% of the top chunk's score
MIN_KEEP = 3             # always keep at least this many chunks if any exist


def _doc_score(doc: dict) -> float:
    """Same scoring precedence used everywhere else in the pipeline:
    rerank_score (BGE cross-encoder) if reranking ran, else
    combined_score (RRF fusion)."""
    return doc.get("rerank_score", doc.get("combined_score", 0.0))


class GraderAgent(BaseAgent):
    """
    Drops retrieved chunks that are far weaker than the best match for
    this query, before they reach AnswerAgent. See module docstring.
    """

    async def _execute(self, state: AgentState) -> AgentState:
        docs = state.retrieved_docs

        if not docs:
            print("[GRADER] No retrieved docs to grade — skipping")
            return state

        scored = [(doc, _doc_score(doc)) for doc in docs]
        top_score = max(score for _, score in scored)

        if top_score <= 0:
            # Nothing scored above zero at all — grading can't distinguish
            # anything, so leave the set untouched rather than dropping
            # everything.
            print("[GRADER] Top score is 0 — no filtering applied")
            return state

        threshold = top_score * RELATIVE_FLOOR

        kept = [doc for doc, score in scored if score >= threshold]

        if len(kept) < MIN_KEEP:
            # Relative threshold dropped too many — pad back up to the
            # top MIN_KEEP chunks by score, so summary/multi-hop
            # questions still get enough context even when no chunk
            # dominates by a wide margin.
            kept = [doc for doc, _ in sorted(scored, key=lambda pair: pair[1], reverse=True)[:MIN_KEEP]]

        dropped_count = len(docs) - len(kept)
        if dropped_count > 0:
            print(f"[GRADER] Dropped {dropped_count}/{len(docs)} chunks below "
                  f"{threshold:.4f} (top_score={top_score:.4f}, floor={RELATIVE_FLOOR})")
        else:
            print(f"[GRADER] Kept all {len(docs)} chunks (top_score={top_score:.4f})")

        state.retrieved_docs = kept
        return state
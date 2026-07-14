"""
eval_rag.py

Lightweight, zero-new-dependency eval harness for the RAG pipeline.

RETRIEVAL EVAL (ready to run now):
    Calls HybridSearchEngine.search() directly. A question is scored as
    "found" if ANY of the top-k retrieved chunks' text contains the
    expected keyword(s) -- matching by CONTENT, not by exact chunk_index.
    This is deliberately more robust than index-matching: your chunker's
    real tokenizer produces different chunk boundaries than any
    approximation used to build the golden set, so exact-index matching
    would silently fail even when retrieval actually works correctly.
    What matters is whether the right information came back, not which
    numbered slot it landed in.

ANSWER EVAL:
    Runs the full pipeline via PIPELINE_ENTRYPOINT and scores each answer
    on THREE independent axes:
      - keyword hits          (cheap, can look like a pass while wrong)
      - faithfulness          (is the answer grounded in retrieved context?)
      - relevance             (does the answer actually address the question?)
    Faithfulness and relevance are deliberately separate judges: a faithful
    answer can still dodge the question, and a relevant-sounding answer can
    still be fabricated. Reporting all three avoids the failure mode where
    a keyword match hides a real hallucination (see UTMOS eval history).

    2026-07-14 fix: faithfulness now also runs a DETERMINISTIC
    entity-attribution cross-check (_numeric_claims_entity_mismatches),
    layered on top of the LLM judge rather than replacing it. Confirmed
    directly: the local LLM judge scored BOTH confirmed UTMOS fabrications
    (answer_utmos_freezeomni_1, answer_utmos_moshi_1) as faithfulness=1.0
    -- a false negative on the two cases that mattered most, because the
    judge (like a plain substring check) can only see that a number
    appears SOMEWHERE in context, not whether it's attached to the right
    entity. Same fix already applied to AnswerAgent's own numeric
    fabrication guard; duplicated here (not imported) to keep this script
    independently runnable per its zero-new-dependency design.

Run:
    python eval_rag.py --golden golden_set.json --user-id <test_user_id>
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Awaitable
import re

@dataclass
class RetrievalResult:
    id: str
    question: str
    found: bool
    rank: Optional[int]


@dataclass
class AnswerResult:
    id: str
    question: str
    answer: str
    keyword_hits: int
    keyword_total: int
    faithfulness_score: Optional[float]
    relevance_score: Optional[float]
    unsupported_claims: List[str] = field(default_factory=list)


def load_golden_set(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    placeholders = [it["id"] for it in items if "REPLACE" in json.dumps(it) or "SET_AFTER" in json.dumps(it)]
    if placeholders:
        print(f"[WARN] {len(placeholders)} golden set entries still contain "
              f"placeholder values and will be skipped: {placeholders}")
    return [it for it in items if "REPLACE" not in json.dumps(it) and "SET_AFTER" not in json.dumps(it)]


def _content_match(text: str, expected_keywords: List[str]) -> bool:
    """A chunk 'matches' if it contains ANY one of the expected keywords
    (case-insensitive). Using ANY (not ALL) because expected_answer_contains
    often lists alternative acceptable phrasings (e.g. "140K" or "140,000"),
    not a checklist every chunk must satisfy simultaneously."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in expected_keywords)


async def run_retrieval_eval(item: dict, hybrid_engine, user_id: str, top_k: int = 6) -> RetrievalResult:
    results = await hybrid_engine.search(
        query=item["question"],
        user_id=user_id,
        top_k=top_k,
        document_id=item.get("document_id"),
    )

    expected_keywords = item.get("expected_answer_contains", [])
    found = False
    rank = None
    for i, r in enumerate(results, start=1):
        if _content_match(r.get("text", ""), expected_keywords):
            found = True
            rank = i
            break

    return RetrievalResult(id=item["id"], question=item["question"], found=found, rank=rank)


FAITHFULNESS_JUDGE_PROMPT = """You are grading whether an AI-generated answer is fully supported by the given context.

Context:
{context}

Answer to grade:
{answer}

Instructions:
- List any claims in the Answer that are NOT directly supported by the Context.
- If every claim is supported, output exactly: NONE
- Otherwise output each unsupported claim on its own line, prefixed with "- "
- Do not explain your reasoning, just list the claims or output NONE.
"""


RELEVANCE_JUDGE_PROMPT = """You are grading whether an AI-generated answer actually addresses the question asked.

Question: {question}
Answer to grade: {answer}

Instructions:
- Score 1.0 if the answer directly and completely addresses the question
- Score 0.5 if the answer is on-topic but partial, vague, or sidesteps part of the question
- Score 0.0 if the answer doesn't address the question at all
- A clear, honest "this isn't in the document" response to a question whose answer
  genuinely isn't in the source counts as fully relevant (1.0) -- declining
  correctly is a correct answer, not a dodge.
- Output ONLY the number (1.0, 0.5, or 0.0), nothing else.
"""


_QUESTION_STOPWORDS = {
    "how", "what", "which", "who", "whom", "when", "where", "why",
    "is", "are", "was", "were", "does", "did", "do", "the", "a", "an",
}


def _numeric_claims_entity_mismatches(answer: str, context: str, question: str) -> List[str]:
    """
    Deterministic cross-check for entity-attribution fabrication (the
    UTMOS bug: an answer reused a REAL number from context, but for the
    WRONG entity -- e.g. "Freeze-Omni achieved 4.50" when 4.50 is
    Lychee-FD's own score, not Freeze-Omni's). This is exactly the class
    of error the LLM judge got wrong: it scored BOTH confirmed
    fabrications as faithfulness=1.0, because "4.50" genuinely does
    appear in the context -- neither the judge nor a plain substring
    check can tell WHICH entity a number belongs to.

    Same logic as AnswerAgent's _numeric_claims_grounded guard --
    duplicated here (not imported) to keep this eval script
    zero-new-dependency and independently runnable per its own docstring.
    If these two implementations ever drift, this is the one to check
    first.

    Returns a list of number strings from `answer` that do NOT co-occur
    with a discriminating entity from `question` in any single context
    span. Empty list = no mismatch detected (either genuinely grounded,
    or no named entity in the question to check against -- e.g. a plain
    "how many layers" question, which this check deliberately sits out
    of rather than false-flagging).
    """
    numbers_in_answer = re.findall(r"\d+\.\d+|\d+", answer)
    if not numbers_in_answer:
        return []

    entities = [
        e for e in re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", question)
        if e.lower() not in _QUESTION_STOPWORDS
    ]
    if not entities:
        return []  # no entity to anchor against -- not this check's job

    # Split on '.' only when NOT between two digits, so decimal numbers
    # like "4.21" don't get fractured into separate spans.
    context_spans = re.split(r"(?<!\d)\.(?!\d)|\n", context)

    # Drop entities that appear in EVERY numbered span (e.g. a metric
    # name like "UTMOS" mentioned in every row) -- those don't
    # discriminate between rows, so matching against them defeats the
    # point of the check.
    spans_with_numbers = [
        s for s in context_spans
        if re.search(r"\d", s) and not re.match(r"^\s*\[.*\]\s*$", s)
    ]
    discriminating = [
        e for e in entities
        if 0 < sum(1 for s in spans_with_numbers if e.lower() in s.lower()) < len(spans_with_numbers)
    ]
    entities = discriminating or entities

    mismatches = []
    for num in numbers_in_answer:
        cooccurs = any(
            num in span and any(e.lower() in span.lower() for e in entities)
            for span in context_spans
        )
        if not cooccurs:
            mismatches.append(num)
    return mismatches


async def judge_faithfulness(llm, context: str, answer: str, question: str) -> tuple:
    """Uses your own LLMProvider.generate() as judge -- confirmed interface
    match, no guessing needed here.

    2026-07-14 fix: the LLM judge's verdict is no longer the only signal.
    A deterministic entity-attribution check runs alongside it (see
    _numeric_claims_entity_mismatches), because the LLM judge alone missed
    both confirmed UTMOS fabrications (scored them faithfulness=1.0).
    Entity mismatches are treated as high-confidence fabrication evidence
    and hard-cap the score at 0.3, regardless of what the LLM judge said,
    so a long otherwise-fine-sounding answer can't dilute a confirmed
    mix-up into a near-passing score via the proportional formula below.
    """
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(context=context, answer=answer)

    llm_call_failed = False
    llm_claims: List[str] = []
    try:
        judge_response = await llm.generate(prompt)
        text = judge_response.strip()
        if not text.upper().startswith("NONE"):
            llm_claims = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
            llm_claims = [c for c in llm_claims if c]
    except Exception as e:
        print(f"[JUDGE] Faithfulness LLM call failed: {e}")
        llm_call_failed = True

    entity_mismatches = _numeric_claims_entity_mismatches(answer, context, question)
    if entity_mismatches:
        print(f"[JUDGE] Deterministic entity-attribution check flagged "
              f"{len(entity_mismatches)} number(s) not attributed to the "
              f"asked-about entity: {entity_mismatches}")
    entity_claim_descs = [
        f"the number {num} appears in context but is not attributed to "
        f"the entity asked about (possible entity-attribution mix-up)"
        for num in entity_mismatches
    ]
    claims = llm_claims + [c for c in entity_claim_descs if c not in llm_claims]

    if not claims:
        if llm_call_failed:
            return None, []
        return 1.0, []

    approx_answer_sentences = max(answer.count(".") + answer.count("\n"), 1)
    score = max(0.0, 1 - len(claims) / approx_answer_sentences)

    if entity_mismatches:
        score = min(score, 0.3)

    return score, claims


# 2026-07-10 fix: local judge model doesn't reliably follow the "clean
# decline = 1.0 relevance" carve-out buried in the prompt -- confirmed
# directly: both decline_nonexistent_dataset_1 and decline_wrong_paper_fact_1
# produced textbook-correct, non-fabricating decline answers ("The sources
# do not contain information about...") yet scored relevance=0.00. This is
# a judge-instruction-following failure, not a pipeline failure -- same
# class of problem as documented judge unreliability elsewhere (Ragas
# timeouts on small model). Fix: deterministic pre-check before trusting
# the LLM judge on this specific carve-out, same pattern as the rewriter's
# acronym guard.
# 2026-07-10 fix v2: exact-phrase DECLINE_PATTERNS list kept missing
# rephrasing ("do not mention", "do not provide" -- neither was in the
# original list, confirmed directly in eval output). The model paraphrases
# declines a different way most runs, so listing exact phrases is
# whack-a-mole. Switched to a structural regex that matches the
# "do/does not + verb" and "no information/not available" *shape* of a
# decline rather than specific wording -- catches "do not mention", "do
# not provide", "does not contain", etc. without enumerating every verb.
DECLINE_REGEX = re.compile(
    r"\b(do(es)?\s+not|didn'?t|is\s+not|are\s+not|no\s+information|"
    r"not\s+(available|found|specified|stated|mentioned|provide[d]?))\b",
    re.IGNORECASE,
)

async def judge_relevance(llm, question: str, answer: str) -> Optional[float]:
    """Separate from faithfulness on purpose: a grounded answer can still
    dodge the actual question, and this axis catches that independently."""
    if DECLINE_REGEX.search(answer):
        return 1.0

    prompt = RELEVANCE_JUDGE_PROMPT.format(question=question, answer=answer)
    try:
        response = await llm.generate(prompt)
        return float(response.strip())
    except Exception as e:
        print(f"[JUDGE] Relevance LLM call failed: {e}")
        return None


async def run_answer_eval(
    item: dict,
    run_pipeline_fn,
    hybrid_engine,
    llm,
    user_id: str,
) -> Optional[AnswerResult]:
    if run_pipeline_fn is None:
        print(f"[SKIP] '{item['id']}' -- no PIPELINE_ENTRYPOINT wired up yet. "
              f"AnswerAgent needs full AgentState from your orchestrator; "
              f"paste that entrypoint and I'll wire this in.")
        return None

    state = await run_pipeline_fn(item["question"], user_id)
    answer_text = state.get("answer", "") if isinstance(state, dict) else getattr(state, "answer", "")

    print(f"[RAW ANSWER] {item['id']}: {answer_text!r}")   

    # 2026-07-10 fix: state["sources"] is built by AnswerAgent with text
    # deliberately truncated to 200 chars (doc["text"][:200]) -- correct
    # for its real purpose, a citation preview, but useless as judging
    # context: if the supporting sentence lands past char 200 of its
    # chunk, the faithfulness/relevance judges never see it and wrongly
    # flag a fully-correct answer as unsupported. Confirmed directly:
    # "4 layers" and "80K voice prompts" are both verbatim-correct per
    # the source PDF, yet scored faithfulness=0.00 against the truncated
    # sources text, while CriticAgent's own grounding check (which reads
    # full-text state.retrieved_docs, not the truncated citation preview)
    # correctly scored 1.00 on the same answer in the same run.
    #
    # Fix: always judge against a fresh, full-text retrieval rather than
    # the truncated citation stub. This intentionally does NOT reuse
    # state["sources"] at all anymore.
    retrieved = await hybrid_engine.search(query=item["question"], user_id=user_id, top_k=6)
    context_text = "\n\n".join(r["text"] for r in retrieved)

    expected_keywords = item.get("expected_answer_contains", [])
    answer_lower = answer_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)

    faith_score, unsupported = await judge_faithfulness(llm, context_text, answer_text, item["question"])
    relevance_score = await judge_relevance(llm, item["question"], answer_text)

    return AnswerResult(
        id=item["id"], question=item["question"], answer=answer_text,
        keyword_hits=hits, keyword_total=len(expected_keywords),
        faithfulness_score=faith_score, relevance_score=relevance_score,
        unsupported_claims=unsupported,
    )


def print_report(retrieval_results, answer_results, top_k: int):
    print("\n" + "=" * 60)
    print("RETRIEVAL EVAL (content-based matching)")
    print("=" * 60)
    if retrieval_results:
        recall = sum(1 for r in retrieval_results if r.found) / len(retrieval_results)
        mrr = sum((1 / r.rank) if r.found else 0 for r in retrieval_results) / len(retrieval_results)
        print(f"Recall@{top_k}: {recall:.2%}  ({sum(1 for r in retrieval_results if r.found)}/{len(retrieval_results)})")
        print(f"MRR:        {mrr:.3f}")
        for r in retrieval_results:
            status = f"rank {r.rank}" if r.found else "NOT FOUND"
            print(f"  [{r.id}] {status} -- {r.question[:60]}")
    else:
        print("(no retrieval-type questions in golden set)")

    print("\n" + "=" * 60)
    print("ANSWER EVAL")
    print("=" * 60)
    if answer_results:
        faith_scores = [r.faithfulness_score for r in answer_results if r.faithfulness_score is not None]
        rel_scores = [r.relevance_score for r in answer_results if r.relevance_score is not None]
        if faith_scores:
            print(f"Avg faithfulness: {sum(faith_scores)/len(faith_scores):.3f}  (n={len(faith_scores)})")
        if rel_scores:
            print(f"Avg relevance:    {sum(rel_scores)/len(rel_scores):.3f}  (n={len(rel_scores)})")
        print()
        for r in answer_results:
            kw_str = f"{r.keyword_hits}/{r.keyword_total} keywords" if r.keyword_total else "n/a"
            faith_str = f"{r.faithfulness_score:.2f}" if r.faithfulness_score is not None else "judge failed"
            rel_str = f"{r.relevance_score:.2f}" if r.relevance_score is not None else "judge failed"
            print(f"  [{r.id}] keywords={kw_str}  faithfulness={faith_str}  relevance={rel_str}")
            if r.unsupported_claims:
                print(f"      unsupported claims flagged: {r.unsupported_claims}")
    else:
        print("(no answer-type questions ran -- see [SKIP] messages above)")
    print()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="golden_set.json")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    golden_items = load_golden_set(args.golden)
    if not golden_items:
        print("No usable golden set entries. Fill in golden_set.json first.")
        sys.exit(1)

    from app.services.retrieval.hybrid_search import HybridSearchEngine
    from app.services.llm.provider import LLMProvider  # confirmed: adjust path if this differs
    from app.db.mongodb.client import connect_to_mongo
    from app.services.retrieval.bm25_bootstrap import rebuild_bm25_indexes

    await connect_to_mongo()       # without this, db=None flows through
                                    # AgentOrchestrator -> build_agent_graph -> planner/retriever,
                                    # breaking metadata lookup and document_resolver silently
                                    # (confirmed via "[DOC_RESOLVER] db is None, skipping filter")

    await rebuild_bm25_indexes()   # hydrates keyword_manager from Qdrant -- a fresh
                                    # process starts with an empty BM25 index otherwise
                                    # (confirmed via "BM25 indexed users: []")

    hybrid_engine = HybridSearchEngine()
    llm = LLMProvider()

    # --- PIPELINE_ENTRYPOINT -------------------------------------------
    # AgentOrchestrator.process() runs the full LangGraph pipeline
    # (rewriter -> planner/retriever -> join -> grader -> route ->
    # tool_agent/no_answer/metadata_answer -> answer -> critic) and
    # returns a plain dict with "answer" / "sources" keys already —
    # matches run_answer_eval's state.get("answer","") / .get("sources",[])
    # exactly, no AgentState reconstruction needed here.
    from app.agents.orchestrator import AgentOrchestrator
    _orchestrator = AgentOrchestrator()

    async def PIPELINE_ENTRYPOINT(question: str, user_id: str):
        return await _orchestrator.process(question, user_id)
    # ---------------------------------------------------------------------

    retrieval_results = []
    answer_results = []

    for item in golden_items:
        if item["type"] == "retrieval":
            retrieval_results.append(
                await run_retrieval_eval(item, hybrid_engine, args.user_id, top_k=args.top_k)
            )
        elif item["type"] == "answer":
            result = await run_answer_eval(item, PIPELINE_ENTRYPOINT, hybrid_engine, llm, args.user_id)
            if result:
                answer_results.append(result)

    print_report(retrieval_results, answer_results, args.top_k)


if __name__ == "__main__":
    asyncio.run(main())
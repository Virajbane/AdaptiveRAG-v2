"""
rag_eval_common.py

Shared metric implementations used by BOTH eval_rag.py and ragas_eval.py,
so the two harnesses can never silently drift on how a given metric is
defined. This is the exact failure mode the bug report flagged in
eval_rag.py's docstring: "duplicated here (not imported)... if these two
implementations ever drift, this is the one to check first." That warning
is now resolved -- there's only one implementation, both scripts import it.

Covers the metrics neither harness reported on its own before this pass:
  - entity_attribution_check   -- was folded into faithfulness only in
                                   eval_rag.py, missing entirely in
                                   ragas_eval.py. Now a standalone metric
                                   in both, since it's the one that caught
                                   the confirmed UTMOS fabrications that
                                   faithfulness alone (LLM judge OR Ragas)
                                   scored as 1.0.
  - context_precision_keyword  -- zero-dependency retrieval-quality metric
                                   for eval_rag.py (Ragas's version needs
                                   the judge LLM; this needs nothing)
  - hallucination_trap scoring -- pass/fail on "should this have declined"
                                   questions, separated out from the
                                   relevance-score carve-out so it reports
                                   as its own number, not buried inside
                                   an averaged relevance score
  - ingestion_completeness     -- pre-flight gate (§2.1 of the bug
                                   report): don't trust any retrieval/
                                   answer metric if pages never made it
                                   into the index in the first place
  - CacheMetricTracker         -- cache hit/miss correctness + cold vs
                                   warm latency, run against the real
                                   pipeline entrypoint

Golden set schema additions this module expects (all optional/backward
compatible -- old golden_set.json files with no new fields still work,
they just won't populate the new metrics):

    Per-item:
      "expects_decline": true          # for hallucination-trap questions
                                        # (id prefix "decline_" also works,
                                        # kept for backward compat with
                                        # the existing eval_rag.py logic)
      "type": "cache"                  # new item type, alongside the
                                        # existing "retrieval" / "answer"
                                        # id prefix "cache_neardup_" means
                                        # a MISS is the expected/correct
                                        # behavior; anything else under
                                        # type "cache" expects a HIT

    Top-level (golden_set.json can now be EITHER a plain list, as before,
    OR a dict shaped like this -- load_golden_set_v2 below handles both):
      {
        "items": [ ... same item objects as before ... ],
        "ingestion_check": {
          "expected_pages": [1, 2, 3, ..., 22]
        }
      }
"""

import re
import time
from typing import List, Optional, Dict, Any, Union

# ---------------------------------------------------------------------------
# Entity-attribution check (moved here from eval_rag.py, unchanged logic)
# ---------------------------------------------------------------------------

# 2026-07-22 fix: added "in", "at", "on", "not", "with", "as", "of", "to",
# "for" -- generic connector words that get capitalized purely by sitting
# at the start of a question sentence (e.g. "In the layer ablation
# study..."), not because they're real named entities. Confirmed root
# cause of false entity-attribution flags on fab_10 ("In" and "Figure"
# were being treated as anchor entities, and since neither ever appears
# near the answer's numbers in context, every number in a correct answer
# got flagged as a mismatch). Deliberately NOT stripping "Figure"/"Table"
# here -- those ARE legitimate discriminating entities for table/figure
# -scoped questions elsewhere in the golden set.
_QUESTION_STOPWORDS = {
    "how", "what", "which", "who", "whom", "when", "where", "why",
    "is", "are", "was", "were", "does", "did", "do", "the", "a", "an",
    "in", "at", "on", "not", "with", "as", "of", "to", "for",
}

# 2026-07-22 fix: excludes digits embedded inside hyphenated/alphanumeric
# identifiers (e.g. "Step-Audio-2", "GPT-5-Duplex") -- those aren't
# factual claims to verify, they're part of a proper noun. This mirrors
# the identical fix already applied in app/agents/answer.py's _NUM_RE;
# this eval-side copy had drifted and was missing it, causing a false
# entity-attribution flag on fab_02 (the "2" in "Step-Audio-2" was
# treated as an unverified numeric claim). Kept as a separate constant
# here rather than importing from app.agents.answer, since this module
# is meant to have zero dependency on the app's internal agent code.
_NUM_RE = re.compile(r'(?<![A-Za-z0-9-])\d+(?:\.\d+)?%?(?![A-Za-z0-9-])')


def numeric_claims_entity_mismatches(answer: str, context: str, question: str) -> List[str]:
    """Deterministic cross-check for entity-attribution fabrication (the
    UTMOS bug: a REAL number from context attributed to the WRONG entity).
    Returns number strings from `answer` that do not co-occur with a
    discriminating entity from `question` in any single context span.
    Empty list = no mismatch detected, OR no named entity to check
    against (deliberately sits out rather than false-flagging plain
    non-entity questions like "how many layers").
    """
    numbers_in_answer = _NUM_RE.findall(answer)
    if not numbers_in_answer:
        return []

    entities = [
        e for e in re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", question)
        if e.lower() not in _QUESTION_STOPWORDS
    ]
    if not entities:
        return []

    context_spans = re.split(r"(?<!\d)\.(?!\d)|\n", context)
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


def entity_attribution_pass(answer: str, context: str, question: str) -> Optional[bool]:
    """Convenience wrapper for the standalone metric: True = no mismatch
    detected (pass), False = at least one number attributed to the wrong
    entity (fail), None = check didn't apply (no entity in question)."""
    numbers_in_answer = _NUM_RE.findall(answer)
    entities = [
        e for e in re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", question)
        if e.lower() not in _QUESTION_STOPWORDS
    ]
    if not numbers_in_answer or not entities:
        return None
    return len(numeric_claims_entity_mismatches(answer, context, question)) == 0


# ---------------------------------------------------------------------------
# Zero-dependency context precision (for eval_rag.py, which has no judge
# LLM call in its retrieval path)
# ---------------------------------------------------------------------------

def context_precision_keyword(retrieved_texts: List[str], expected_keywords: List[str]) -> Optional[float]:
    """Of the chunks actually retrieved, what fraction contain at least
    one expected keyword? Cruder than an LLM-judged per-chunk relevance
    score (that's what ragas_eval.py's LLMContextPrecisionWithoutReference
    gives you), but needs zero judge calls -- useful as a fast sanity
    check that should broadly agree with the Ragas number. A big gap
    between the two is itself a signal worth investigating.
    """
    if not retrieved_texts or not expected_keywords:
        return None
    relevant = sum(
        1 for t in retrieved_texts
        if any(kw.lower() in t.lower() for kw in expected_keywords)
    )
    return relevant / len(retrieved_texts)


# ---------------------------------------------------------------------------
# Hallucination trap / abstention correctness
# ---------------------------------------------------------------------------

DECLINE_REGEX = re.compile(
    r"\b(do(es)?\s+not|didn'?t|is\s+not|are\s+not|no\s+information|"
    r"not\s+(available|found|specified|stated|mentioned|provide[d]?))\b",
    re.IGNORECASE,
)


def score_hallucination_trap(item: Dict[str, Any], answer_text: str) -> Optional[bool]:
    """True = correctly declined a trap question. False = fabricated
    an answer to a question with no valid answer in the corpus (the
    worst-case failure). None = not a trap item, this check doesn't apply.

    Uses the SAME structural decline regex as the relevance-judge carve-
    out (kept consistent on purpose -- two different decline-detectors
    would just create a new drift risk of their own).
    """
    is_trap = item.get("expects_decline") or item.get("id", "").startswith("decline_")
    if not is_trap:
        return None
    return bool(DECLINE_REGEX.search(answer_text))


def score_false_decline(item: Dict[str, Any], answer_text: str) -> Optional[bool]:
    """The mirror-image failure: did the system decline on a question
    that DOES have a valid answer in the corpus? True = false decline
    (bad -- unnecessarily unhelpful), False = answered as it should have,
    None = this item isn't a plain answerable question.
    """
    is_trap = item.get("expects_decline") or item.get("id", "").startswith("decline_")
    if is_trap or item.get("type") != "answer":
        return None
    return bool(DECLINE_REGEX.search(answer_text))


# ---------------------------------------------------------------------------
# Ingestion completeness gate (§2.1)
# ---------------------------------------------------------------------------

def check_ingestion_completeness(
    chunks_by_page: Dict[int, int], expected_pages: List[int]
) -> Dict[str, Any]:
    """Pre-flight gate: confirms every expected page produced at least one
    indexed chunk BEFORE any retrieval/answer metric is trusted. Run this
    FIRST and print/fail loudly if `complete` is False -- a downstream
    Recall@6 of 89% is meaningless if it's silently computed over a
    corpus that's missing 4 pages, which is exactly what happened
    undetected in the original eval run.

    `chunks_by_page` should map page_number -> chunk_count. Build this
    from whatever store holds chunk metadata (Qdrant payload / Mongo doc)
    -- schema-agnostic on purpose since that wiring is specific to your
    ingestion pipeline. See the wiring note in each script's main().
    """
    missing_pages = [p for p in expected_pages if chunks_by_page.get(p, 0) == 0]
    coverage = 1 - (len(missing_pages) / len(expected_pages)) if expected_pages else None
    return {
        "expected_pages": expected_pages,
        "missing_pages": missing_pages,
        "coverage": coverage,
        "complete": len(missing_pages) == 0,
    }


# ---------------------------------------------------------------------------
# Cache hit accuracy + latency
# ---------------------------------------------------------------------------

class CacheMetricTracker:
    """Runs each cache-test item TWICE (cold, then immediate repeat) and
    records latency + whether the repeat was actually served from cache.
    Works with any pipeline fn shaped `async def fn(question, user_id) ->
    dict`, matching both PIPELINE_ENTRYPOINT (eval_rag.py) and
    orchestrator.process (ragas_eval.py) as-is.

    id prefix "cache_neardup_" -> expected behavior is a MISS (semantic
    rephrase, exact-hash cache correctly treats it as new -- per the bug
    report's "documented not hidden" note). Anything else under
    type=="cache" expects a HIT.
    """

    def __init__(self, hit_threshold_s: float = 5.0):
        self.rows: List[Dict[str, Any]] = []
        self.hit_threshold_s = hit_threshold_s

    async def run(self, item: Dict[str, Any], pipeline_fn, user_id: str):
        q = item["question"]

        t0 = time.perf_counter()
        await pipeline_fn(q, user_id)
        cold_latency = time.perf_counter() - t0

        t1 = time.perf_counter()
        await pipeline_fn(q, user_id)
        warm_latency = time.perf_counter() - t1

        expected_hit = not item.get("id", "").startswith("cache_neardup_")
        hit_detected = warm_latency < self.hit_threshold_s
        correct = hit_detected == expected_hit

        row = {
            "id": item["id"],
            "question": q,
            "cold_latency_s": round(cold_latency, 2),
            "warm_latency_s": round(warm_latency, 2),
            "expected_cache_hit": expected_hit,
            "detected_cache_hit": hit_detected,
            "correct": correct,
        }
        self.rows.append(row)
        return row

    def summary(self) -> Dict[str, Any]:
        if not self.rows:
            return {}
        accuracy = sum(1 for r in self.rows if r["correct"]) / len(self.rows)
        avg_cold = sum(r["cold_latency_s"] for r in self.rows) / len(self.rows)
        avg_warm = sum(r["warm_latency_s"] for r in self.rows) / len(self.rows)
        return {
            "cache_accuracy": accuracy,
            "avg_cold_latency_s": round(avg_cold, 2),
            "avg_warm_latency_s": round(avg_warm, 2),
            "n": len(self.rows),
        }


# ---------------------------------------------------------------------------
# Backward-compatible golden set loader (list OR dict-with-metadata)
# ---------------------------------------------------------------------------

def load_golden_set_v2(path: str) -> Dict[str, Any]:
    """Returns {"items": [...], "ingestion_check": {...} or None}.
    Accepts EITHER the old plain-list golden_set.json format, or a new
    dict format with an "ingestion_check" key alongside "items". Old
    files keep working unchanged -- ingestion_check just comes back None
    and that section of the report is skipped with a clear note, not a
    crash.
    
    Automatically filters out comment-only entries (items with no "type" field),
    which are useful for documentation but not test items. This prevents
    KeyError: 'type' crashes when the eval loop processes items.
    """
    import json

    with open(path, "r", encoding="utf-8") as f:
        raw: Union[list, dict] = json.load(f)

    if isinstance(raw, list):
        items = raw
        ingestion_check = None
    else:
        items = raw.get("items", [])
        ingestion_check = raw.get("ingestion_check")

    # ✅ NEW: Filter out comment-only entries (no "type" field)
    # Items with only a "comment" field are documentation, not test items.
    # This prevents KeyError: 'type' in eval_rag.py line 464.
    comment_items = [it for it in items if isinstance(it, dict) and "type" not in it]
    if comment_items:
        print(f"[INFO] Filtered out {len(comment_items)} comment-only entries from golden set")
    items = [it for it in items if isinstance(it, dict) and "type" in it]

    # Filter out items with placeholder values
    placeholders = [it.get("id", "?") for it in items if "REPLACE" in json.dumps(it) or "SET_AFTER" in json.dumps(it)]
    if placeholders:
        print(f"[WARN] {len(placeholders)} golden set entries still contain "
              f"placeholder values and will be skipped: {placeholders}")
    items = [it for it in items if "REPLACE" not in json.dumps(it) and "SET_AFTER" not in json.dumps(it)]

    return {"items": items, "ingestion_check": ingestion_check}
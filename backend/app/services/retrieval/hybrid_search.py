import asyncio
import re
from typing import List, Optional, Dict, Set
from app.services.retrieval.vector_search import VectorSearchEngine
from app.services.retrieval.keyword_search import keyword_manager
from app.services.retrieval.reranker import bge_reranker


# =============================================================================
# FIX #3 (2026-08-06): COMPARISON TABLE ROW DEDUPLICATION FILTER
# =============================================================================
# Prevents retrieving multiple rows from the same comparison table unless
# the user explicitly asked for a comparison. Without this, the LLM can see
# e.g. both "Row [Lychee-FD]" and "Row [Moshi]" in the same context and
# confuse which values belong to which entity.

# Pattern to detect table row chunks: "Row [Entity]: field=value, ..."
_ROW_PATTERN = re.compile(r'^\s*Row \[(.*?)\]:', re.MULTILINE)


def _extract_table_rows_from_chunk(text: str) -> Set[str]:
    """
    Returns set of entity labels found in table rows within this chunk.
    E.g., from "Row [Lychee-FD]: S→S=4.50, ..." returns {"Lychee-FD"}
    """
    return set(m.group(1) for m in _ROW_PATTERN.finditer(text))


def _canonical_entity_name(entity: str) -> str:
    """
    Strip qualifiers like "(Ours)", "w/o Sem", trailing " -" etc. so
    "Lychee-FD (Ours) w/o Sem -" and "Lychee-FD" are recognized as the
    same entity when matching against the question text.
    """
    canonical = re.split(r"\(| w/o |w/o ", entity, maxsplit=1)[0]
    return canonical.strip(" -").strip()


def _entity_matches_question(entity: str, question: str) -> bool:
    """True if this row's entity (by canonical name) is actually referenced
    in the question -- not just present somewhere in the retrieved set."""
    canonical = _canonical_entity_name(entity).lower()
    return bool(canonical) and canonical in question.lower()


def _detect_comparison_intent(question: str) -> bool:
    """
    Heuristic: is the question explicitly asking for a comparison?
    """
    comparison_keywords = {
        "compare", "vs", "versus", "better", "worse", "difference",
        "outperform", "beat", "surpass", "how much", "which is",
        "contrast", "similar", "differ", "both", "contrast"
    }
    q_lower = question.lower()
    return any(kw in q_lower for kw in comparison_keywords)


def filter_comparison_table_rows(
    retrieved_docs: List[Dict],
    question: str,
    max_rows_per_table: int = 1,
    allow_multiple_if_comparison: bool = True
) -> List[Dict]:
    """
    Post-processing filter for hybrid_search results.

    After RRF fusion + reranking have run, inspect the retrieved docs for
    table row chunks. If multiple rows from the same comparison table
    appear, keep only the top-ranked one (unless comparison was explicitly
    asked for, in which case allow more).
    """
    if not retrieved_docs:
        return retrieved_docs

    # Detect if this is a comparison query
    is_comparison = (
        _detect_comparison_intent(question) or
        question.count(" and ") + question.count(" vs ") > 0
    )

    # Collect all entities mentioned in retrieved rows, matched by
    # canonical name against the question -- NOT raw substring. A label
    # like "Lychee-FD (Ours) w/o Sem -" won't literally appear in the
    # question even though "Lychee-FD" clearly refers to that row.
    all_entities = set()
    for doc in retrieved_docs:
        all_entities.update(_extract_table_rows_from_chunk(doc.get("text", "")))

    entities_matched_in_question = {
        entity for entity in all_entities
        if _entity_matches_question(entity, question)
    }
    # Could we identify ANY entity by name in the question at all? If not,
    # we can't tell which row is relevant -- fall back to trusting rank
    # order alone (old behavior) rather than dropping everything.
    entity_identifiable = len(entities_matched_in_question) > 0
    entities_in_question = len(entities_matched_in_question)

    # Adjust the limit based on context
    effective_max = max_rows_per_table
    if allow_multiple_if_comparison and is_comparison:
        effective_max = 3  # Allow up to 3 rows for explicit comparisons
    elif entities_in_question > 1:
        effective_max = 2  # Allow 2 if multiple entities mentioned in question

    # Track which row entities we've already included
    included_rows = {}  # maps entity -> count

    filtered = []
    for doc in retrieved_docs:
        text = doc.get("text", "")
        rows_in_chunk = _extract_table_rows_from_chunk(text)

        if not rows_in_chunk:
            # Not a table row -- keep it unconditionally
            filtered.append(doc)
            continue

        row_entities = list(rows_in_chunk)
        primary_entity = row_entities[0]

        # NEW: entity-relevance check. Per-entity repeat-counting alone
        # can't catch two DIFFERENT entities each appearing once (e.g. a
        # Lychee-FD row + a Moshi row) -- each is "within limit" on its
        # own. If we can identify entities by name in the question and
        # this isn't a comparison query, drop rows for entities that
        # aren't the one actually being asked about.
        if (
            entity_identifiable
            and not is_comparison
            and primary_entity not in entities_matched_in_question
        ):
            print(
                f"[RETRIEVAL][DEDUP] Dropped irrelevant-entity row "
                f"[{primary_entity}] (question refers to "
                f"{sorted(entities_matched_in_question)})"
            )
            continue

        included_rows[primary_entity] = included_rows.get(primary_entity, 0) + 1

        if included_rows[primary_entity] <= effective_max:
            filtered.append(doc)
        else:
            print(
                f"[RETRIEVAL][DEDUP] Dropped redundant row "
                f"[{primary_entity}] (already have {effective_max})"
            )

    return filtered


# =============================================================================
# HYBRID SEARCH ENGINE
# =============================================================================

class HybridSearchEngine:
    """
    Hybrid search: BM25 + Vector + Reciprocal Rank Fusion (RRF).
    RRF is rank-based, not score-based — immune to score scale mismatches
    between BM25 and cosine similarity. Industry standard for hybrid RAG.

    2026-08-06 FIX #3: Added comparison table row filtering to prevent
    multiple ambiguous rows from the same table landing in the context window.
    """

    RRF_K = 60  # standard constant, prevents top rank from dominating

    def __init__(self):
        self.vector_engine = VectorSearchEngine()
        self.keyword_engine = keyword_manager
        self.reranker = bge_reranker

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 6,
        rerank_pool_size: int = 20,
        document_id: Optional[str] = None,
    ) -> List[dict]:

        # Run both searches in parallel, fetch more candidates
        vector_results, keyword_results = await asyncio.gather(
            self.vector_engine.search(query, user_id, top_k=20, document_id=document_id),
            self.keyword_engine.search(user_id, query, top_k=20, document_id=document_id)
        )

        # DEBUG
        print(f"[HYBRID DEBUG] user_id: {user_id}")
        print(f"[HYBRID DEBUG] document_id filter: {document_id}")
        print(f"[HYBRID DEBUG] vector results: {len(vector_results)}")
        print(f"[HYBRID DEBUG] keyword results: {len(keyword_results)}")
        print(f"[HYBRID DEBUG] BM25 indexed users: {list(self.keyword_engine.user_indexes.keys())}")

        # Fuse with RRF
        fused = self._reciprocal_rank_fusion(vector_results, keyword_results)

        # Rerank top candidates with BGE cross-encoder if available
        candidate_pool = fused[:rerank_pool_size]

        if self.reranker.available and candidate_pool:
            loop = asyncio.get_event_loop()
            reranked = await loop.run_in_executor(
                None,
                self.reranker.rerank,
                query,
                candidate_pool,
                # Rerank a bit past top_k so the dedup filter below has
                # room to drop redundant rows without starving results.
                min(len(candidate_pool), top_k + rerank_pool_size - top_k)
            )
            # ===== FIX #3: dedup filter, applied after ranking, before slice =====
            reranked = filter_comparison_table_rows(
                reranked,
                question=query,
                max_rows_per_table=1,
                allow_multiple_if_comparison=True
            )
            return reranked[:top_k]

        # ===== FIX #3: dedup filter, applied after ranking, before slice =====
        candidate_pool = filter_comparison_table_rows(
            candidate_pool,
            question=query,
            max_rows_per_table=1,
            allow_multiple_if_comparison=True
        )
        return candidate_pool[:top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[dict],
        keyword_results: List[dict],
    ) -> List[dict]:
        """
        RRF formula: score(d) = sum(1 / (k + rank(d)))
        Each result list contributes a rank-based score.
        A chunk appearing in both lists gets scores from both — no weighting needed.
        """
        scores = {}   # key -> rrf_score
        docs = {}     # key -> doc dict

        for rank, doc in enumerate(vector_results):
            key = f"{doc['doc_id']}:{doc['chunk_index']}"
            scores[key] = scores.get(key, 0) + 1 / (self.RRF_K + rank + 1)
            docs[key] = {**doc, 'vector_score': doc['score'], 'keyword_score': 0}

        for rank, doc in enumerate(keyword_results):
            key = f"{doc['doc_id']}:{doc['chunk_index']}"
            scores[key] = scores.get(key, 0) + 1 / (self.RRF_K + rank + 1)
            if key in docs:
                docs[key]['keyword_score'] = max(0, doc['score'])
            else:
                docs[key] = {**doc, 'vector_score': 0, 'keyword_score': max(0, doc['score'])}

        # Attach RRF score as combined_score (keeps downstream code unchanged)
        for key, doc in docs.items():
            doc['combined_score'] = scores[key]

        return sorted(docs.values(), key=lambda x: x['combined_score'], reverse=True)

    def _deduplicate(self, results: List[dict]) -> List[dict]:
        seen = {}
        for result in results:
            key = f"{result['doc_id']}:{result['chunk_index']}"
            if key not in seen or result['combined_score'] > seen[key]['combined_score']:
                seen[key] = result
        return list(seen.values())
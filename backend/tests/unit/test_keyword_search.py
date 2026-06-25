"""
Unit tests for app/services/retrieval/keyword_search.py

Tests cover:
  - KeywordSearchEngine: tokenization, indexing, BM25 scoring, zero-score filtering
  - KeywordSearchManager: per-user isolation, chunk accumulation, search delegation

No mocking required — everything is in-memory (rank_bm25 only).

Known gap documented:
  - KeywordSearchManager.index_document() rebuilds the full BM25 index on every
    call (O(total_chunks)), not just the newly added chunks. As the user accumulates
    documents this cost grows without bound. Acceptable for an MVP; will need
    incremental indexing or a batch-rebuild strategy at scale.
"""

import pytest
from app.services.retrieval.keyword_search import (
    KeywordSearchEngine,
    KeywordSearchManager,
    keyword_manager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunks(doc_id: str, texts: list[str]) -> list[dict]:
    return [
        {"doc_id": doc_id, "chunk_index": i, "text": text}
        for i, text in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# KeywordSearchEngine — tokenization
# ---------------------------------------------------------------------------

class TestKeywordSearchEngineTokenize:
    def test_lowercases_text(self):
        engine = KeywordSearchEngine()
        tokens = engine._tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_strips_punctuation(self):
        engine = KeywordSearchEngine()
        tokens = engine._tokenize("Hello, world! It's fine.")
        # re.findall(r'\w+', ...) keeps word chars and underscores
        assert "hello" in tokens
        assert "world" in tokens
        assert "fine" in tokens
        # Punctuation must not appear
        assert "," not in tokens
        assert "!" not in tokens

    def test_empty_string_returns_empty_list(self):
        engine = KeywordSearchEngine()
        assert engine._tokenize("") == []

    def test_numbers_kept_as_tokens(self):
        engine = KeywordSearchEngine()
        tokens = engine._tokenize("Q3 2024 revenue")
        assert "q3" in tokens
        assert "2024" in tokens


# ---------------------------------------------------------------------------
# KeywordSearchEngine — build_index and search
# ---------------------------------------------------------------------------

class TestKeywordSearchEngineSearch:
    def test_search_before_index_returns_empty(self):
        engine = KeywordSearchEngine()
        results = engine.search("anything", top_k=5)
        assert results == []

    def test_search_returns_relevant_results(self):
        engine = KeywordSearchEngine()
        chunks = make_chunks("doc1", [
            "The quarterly revenue figures for Q3 exceeded targets.",
            "Employee onboarding procedures were updated last month.",
        ])
        engine.build_index(chunks)
        results = engine.search("quarterly revenue Q3", top_k=5)
        assert len(results) >= 1
        assert results[0]["doc_id"] == "doc1"
        assert results[0]["chunk_index"] == 0

    def test_search_top_result_outranks_unrelated_chunk(self):
        """
        CHANGED from test_search_filters_zero_score_results.

        Under BM25Plus, a tiny corpus's unrelated chunk can carry a small
        positive floor score (see test_zero_overlap_can_still_be_returned_
        in_tiny_corpus) rather than being excluded outright at 0. The
        meaningful guarantee is RANKING, not exclusion: the relevant chunk
        must always outscore the unrelated one.
        """
        engine = KeywordSearchEngine()
        chunks = make_chunks("doc1", [
            "Python programming and software engineering.",
            "Cooking recipes and culinary arts.",
        ])
        engine.build_index(chunks)
        results = engine.search("python software", top_k=10)
        assert results[0]["text"] == "Python programming and software engineering."
        if len(results) > 1:
            assert results[0]["score"] > results[1]["score"]

    def test_zero_overlap_can_still_be_returned_in_tiny_corpus(self):
        """
        KNOWN / ACCEPTED CONSEQUENCE of switching to BM25Plus:

        BM25Plus adds an IDF floor (delta) so small corpora don't collapse
        exact matches to 0 (see test_search_returns_relevant_results). The
        flip side: a chunk with ZERO query-term overlap can still receive a
        small positive score in a tiny corpus. This is not a leak of
        irrelevant results "by accident" — it is the documented, accepted
        tradeoff of fixing the negative/zero-score bug. Downstream callers
        (HybridSearchEngine, top_k slicing) treat this as a ranking signal,
        not a hard yes/no gate, same as vector search scores.
        """
        engine = KeywordSearchEngine()
        chunks = make_chunks("doc1", [
            "Python programming and software engineering.",
            "Cooking recipes and culinary arts.",
        ])
        engine.build_index(chunks)
        results = engine.search("python software", top_k=10)
        cooking_result = next(
            (r for r in results if r["text"] == "Cooking recipes and culinary arts."),
            None,
        )
        python_result = next(
            r for r in results if r["text"] == "Python programming and software engineering."
        )
        if cooking_result is not None:
            assert cooking_result["score"] < python_result["score"]

    def test_search_result_shape(self):
        """Every result must carry the required fields."""
        engine = KeywordSearchEngine()
        chunks = make_chunks("doc42", ["Machine learning model training."])
        engine.build_index(chunks)
        results = engine.search("machine learning", top_k=5)
        assert len(results) == 1
        r = results[0]
        assert r["doc_id"] == "doc42"
        assert r["chunk_index"] == 0
        assert isinstance(r["score"], float)
        assert r["score"] > 0
        assert r["search_type"] == "keyword"
        assert "text" in r

    def test_top_k_limits_results(self):
        engine = KeywordSearchEngine()
        # 10 chunks all containing the query word
        chunks = make_chunks("doc1", [f"revenue report section {i}" for i in range(10)])
        engine.build_index(chunks)
        results = engine.search("revenue", top_k=3)
        assert len(results) <= 3

    def test_results_ordered_by_score_descending(self):
        engine = KeywordSearchEngine()
        chunks = make_chunks("doc1", [
            # High relevance: all three query terms
            "machine learning neural network deep learning",
            # Medium relevance: one query term
            "machine tools in a factory",
            # Low/no relevance
            "cooking pasta with tomatoes",
        ])
        engine.build_index(chunks)
        results = engine.search("machine learning neural", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rebuild_index_replaces_previous(self):
        """build_index() should replace the old index, not append to it."""
        engine = KeywordSearchEngine()
        engine.build_index(make_chunks("doc1", ["old content about cats"]))
        engine.build_index(make_chunks("doc2", ["new content about dogs"]))
        results = engine.search("cats", top_k=5)
        # "cats" no longer in the index
        assert len(results) == 0


# ---------------------------------------------------------------------------
# KeywordSearchManager — per-user isolation and accumulation
# ---------------------------------------------------------------------------

class TestKeywordSearchManager:
    """
    Each test creates a *fresh* KeywordSearchManager so that the module-level
    `keyword_manager` singleton is never contaminated between tests.
    The global instance itself is not tested here to avoid cross-test pollution
    (same rationale as rate_limit_store in conftest.py).
    """

    @pytest.fixture
    def manager(self):
        return KeywordSearchManager()

    @pytest.mark.asyncio
    async def test_search_unknown_user_returns_empty(self, manager):
        results = await manager.search("user_x", "anything", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_index_and_search_basic(self, manager):
        chunks = make_chunks("doc1", ["Revenue increased by 20% in Q3 2024."])
        await manager.index_document("user_1", chunks)
        results = await manager.search("user_1", "revenue Q3", top_k=5)
        assert len(results) >= 1
        assert results[0]["doc_id"] == "doc1"

    @pytest.mark.asyncio
    async def test_user_isolation(self, manager):
        """User A's documents must not appear in User B's search results."""
        await manager.index_document(
            "user_a",
            make_chunks("docA", ["Confidential financial report Q3."]),
        )
        await manager.index_document(
            "user_b",
            make_chunks("docB", ["Public marketing campaign overview."]),
        )
        results_b = await manager.search("user_b", "financial report", top_k=5)
        doc_ids = [r["doc_id"] for r in results_b]
        assert "docA" not in doc_ids

    @pytest.mark.asyncio
    async def test_chunks_accumulate_across_documents(self, manager):
        """
        Indexing a second document must NOT wipe out the first document's chunks.
        Both should be searchable after two index_document() calls.
        """
        await manager.index_document(
            "user_1",
            make_chunks("doc1", ["Machine learning algorithms and neural networks."]),
        )
        await manager.index_document(
            "user_1",
            make_chunks("doc2", ["Revenue and financial performance this quarter."]),
        )
        ml_results = await manager.search("user_1", "machine learning", top_k=5)
        finance_results = await manager.search("user_1", "revenue financial", top_k=5)

        ml_doc_ids = [r["doc_id"] for r in ml_results]
        finance_doc_ids = [r["doc_id"] for r in finance_results]

        assert "doc1" in ml_doc_ids
        assert "doc2" in finance_doc_ids

    @pytest.mark.asyncio
    async def test_index_rebuild_cost_grows_with_chunks(self, manager):
        """
        KNOWN GAP: index_document() rebuilds the full BM25 index over ALL
        accumulated chunks on every call — O(total_chunks), not O(new_chunks).

        This test documents the behaviour: after N index_document() calls the
        internal chunk list has N * chunks_per_call entries, confirming the
        rebuild touches an ever-growing corpus.

        Fix needed before scaling: replace full rebuild with incremental indexing
        or a periodic batch-rebuild strategy.
        """
        user_id = "user_rebuild_test"
        for i in range(5):
            await manager.index_document(
                user_id,
                make_chunks(f"doc{i}", [f"Unique content for document number {i}."]),
            )
        # After 5 calls of 1 chunk each, internal list should have 5 chunks
        assert len(manager.user_chunks[user_id]) == 5
        # And all are searchable
        results = await manager.search(user_id, "unique content document", top_k=10)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_multiple_users_independent_indexes(self, manager):
        """Each user gets their own BM25 index object."""
        await manager.index_document("u1", make_chunks("d1", ["alpha beta gamma"]))
        await manager.index_document("u2", make_chunks("d2", ["delta epsilon zeta"]))
        assert "u1" in manager.user_indexes
        assert "u2" in manager.user_indexes
        assert manager.user_indexes["u1"] is not manager.user_indexes["u2"]
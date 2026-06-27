"""
Unit tests for:
  - app/services/retrieval/vector_search.py  (VectorSearchEngine)
  - app/services/retrieval/hybrid_search.py  (HybridSearchEngine)

Both classes build external dependencies in __init__, so we patch at the
class level before instantiation:

  VectorSearchEngine:
    - patches QdrantVectorDB() (hits Qdrant at construction)
    - patches EmbeddingGenerator() (hits Ollama at construction)

  HybridSearchEngine:
    - patches VectorSearchEngine() (which would chain to both above)
    - patches keyword_manager (the module-level singleton)

Known gaps documented inline.

Phase 14 note: HybridSearchEngine now reranks its candidate pool with
the real BGE cross-encoder singleton (app.services.retrieval.reranker.
bge_reranker) when available, before applying top_k. This is NOT
patched/mocked in the hybrid_engine fixture below - the real model
loads and runs during these tests, which means the final result order
can differ from plain combined_score ordering. Tests that specifically
verify combined_score-based sorting/combination logic in isolation
disable the reranker for the duration of that test
(hybrid_engine.reranker.available = False) so they keep testing what
they were always meant to test, independent of reranking behavior.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# ===========================================================================
# VectorSearchEngine
# ===========================================================================

class TestVectorSearchEngine:
    """
    VectorSearchEngine.__init__() creates QdrantVectorDB() and
    EmbeddingGenerator(). Both must be patched before import so no network
    calls happen.
    """

    @pytest.fixture
    def mock_vector_db(self):
        return MagicMock()

    @pytest.fixture
    def mock_embedder(self):
        m = MagicMock()
        # embed_text is async
        m.embed_text = AsyncMock(return_value=[0.1] * 768)
        return m

    @pytest.fixture
    def vector_engine(self, mock_vector_db, mock_embedder):
        with patch("app.services.retrieval.vector_search.QdrantVectorDB") as MockDB, \
             patch("app.services.retrieval.vector_search.EmbeddingGenerator") as MockEmb:
            MockDB.return_value = mock_vector_db
            MockEmb.return_value = mock_embedder

            from app.services.retrieval.vector_search import VectorSearchEngine
            engine = VectorSearchEngine()
            # Expose mocks for assertion
            engine._mock_db = mock_vector_db
            engine._mock_embedder = mock_embedder
            yield engine

    # --- search() ---

    @pytest.mark.asyncio
    async def test_search_embeds_query_first(self, vector_engine):
        vector_engine._mock_db.search = AsyncMock(return_value=[])
        await vector_engine.search("what is revenue", "user1", top_k=5)
        vector_engine._mock_embedder.embed_text.assert_awaited_once_with("what is revenue")

    @pytest.mark.asyncio
    async def test_search_passes_embedding_to_qdrant(self, vector_engine):
        fake_embedding = [0.5] * 768
        vector_engine._mock_embedder.embed_text = AsyncMock(return_value=fake_embedding)
        vector_engine._mock_db.search = AsyncMock(return_value=[])

        await vector_engine.search("query", "user1", top_k=3)

        vector_engine._mock_db.search.assert_awaited_once_with(
            query_vector=fake_embedding,
            user_id="user1",
            top_k=3,
        )

    @pytest.mark.asyncio
    async def test_search_adds_search_type_vector(self, vector_engine):
        """Every result returned by Qdrant must be tagged search_type='vector'."""
        qdrant_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "some text", "score": 0.9, "tokens": 10},
            {"doc_id": "doc1", "chunk_index": 1, "text": "more text", "score": 0.8, "tokens": 8},
        ]
        vector_engine._mock_db.search = AsyncMock(return_value=qdrant_results)

        results = await vector_engine.search("query", "user1")

        for r in results:
            assert r["search_type"] == "vector"

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_qdrant_empty(self, vector_engine):
        vector_engine._mock_db.search = AsyncMock(return_value=[])
        results = await vector_engine.search("query", "user1")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_result_count_matches_qdrant(self, vector_engine):
        qdrant_results = [
            {"doc_id": f"doc{i}", "chunk_index": 0, "text": "text", "score": 0.9 - i * 0.1, "tokens": 5}
            for i in range(4)
        ]
        vector_engine._mock_db.search = AsyncMock(return_value=qdrant_results)
        results = await vector_engine.search("query", "user1", top_k=4)
        assert len(results) == 4


# ===========================================================================
# HybridSearchEngine
# ===========================================================================

class TestHybridSearchEngine:
    """
    HybridSearchEngine.__init__() calls VectorSearchEngine() which chains to
    QdrantVectorDB + EmbeddingGenerator. Easiest approach: patch
    VectorSearchEngine and keyword_manager directly on the module.
    """

    @pytest.fixture
    def mock_vector_engine(self):
        m = MagicMock()
        m.search = AsyncMock(return_value=[])
        return m

    @pytest.fixture
    def mock_keyword_engine(self):
        m = MagicMock()
        m.search = AsyncMock(return_value=[])
        return m

    @pytest.fixture
    def hybrid_engine(self, mock_vector_engine, mock_keyword_engine):
        with patch("app.services.retrieval.hybrid_search.VectorSearchEngine") as MockVec, \
             patch("app.services.retrieval.hybrid_search.keyword_manager", mock_keyword_engine):
            MockVec.return_value = mock_vector_engine

            from app.services.retrieval.hybrid_search import HybridSearchEngine
            engine = HybridSearchEngine()
            engine._mock_vector = mock_vector_engine
            engine._mock_keyword = mock_keyword_engine
            yield engine

    # --- search() orchestration ---

    @pytest.mark.asyncio
    async def test_both_engines_called(self, hybrid_engine):
        await hybrid_engine.search("revenue Q3", "user1", top_k=5)
        hybrid_engine._mock_vector.search.assert_awaited_once()
        hybrid_engine._mock_keyword.search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_vector_engine_receives_correct_args(self, hybrid_engine):
        await hybrid_engine.search("revenue Q3", "user1", top_k=5)
        hybrid_engine._mock_vector.search.assert_awaited_once_with(
            "revenue Q3", "user1", top_k=10
        )

    @pytest.mark.asyncio
    async def test_keyword_engine_receives_correct_args(self, hybrid_engine):
        await hybrid_engine.search("revenue Q3", "user1", top_k=5)
        hybrid_engine._mock_keyword.search.assert_awaited_once_with(
            "user1", "revenue Q3", top_k=10
        )

    @pytest.mark.asyncio
    async def test_returns_empty_when_both_empty(self, hybrid_engine):
        results = await hybrid_engine.search("anything", "user1", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_top_k_limits_final_results(self, hybrid_engine):
        # 8 vector results, none in keyword — should still cap at top_k=3
        vector_results = [
            {"doc_id": "doc1", "chunk_index": i, "text": f"text {i}",
             "score": 0.9 - i * 0.05, "search_type": "vector"}
            for i in range(8)
        ]
        hybrid_engine._mock_vector.search = AsyncMock(return_value=vector_results)

        results = await hybrid_engine.search("query", "user1", top_k=3)
        assert len(results) <= 3

    # --- _combine_results() ---

    def test_combine_vector_only(self, hybrid_engine):
        vector_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "text", "score": 0.8, "search_type": "vector"},
        ]
        combined = hybrid_engine._combine_results(vector_results, [], 0.6, 0.4)
        assert len(combined) == 1
        r = combined[0]
        assert r["vector_score"] == 0.8
        assert r["keyword_score"] == 0
        assert pytest.approx(r["combined_score"]) == 0.8 * 0.6

    def test_combine_keyword_only(self, hybrid_engine):
        keyword_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "text", "score": 5.0, "search_type": "keyword"},
        ]
        combined = hybrid_engine._combine_results([], keyword_results, 0.6, 0.4)
        assert len(combined) == 1
        r = combined[0]
        assert r["vector_score"] == 0
        assert r["keyword_score"] == 5.0
        assert pytest.approx(r["combined_score"]) == 5.0 * 0.4

    def test_combine_same_chunk_both_engines(self, hybrid_engine):
        """A chunk appearing in both result sets must be merged, not duplicated."""
        shared_chunk = {"doc_id": "doc1", "chunk_index": 0, "text": "text"}
        vector_results = [{**shared_chunk, "score": 0.8, "search_type": "vector"}]
        keyword_results = [{**shared_chunk, "score": 3.0, "search_type": "keyword"}]

        combined = hybrid_engine._combine_results(vector_results, keyword_results, 0.6, 0.4)
        assert len(combined) == 1
        r = combined[0]
        assert r["vector_score"] == 0.8
        assert r["keyword_score"] == 3.0
        assert pytest.approx(r["combined_score"]) == (0.8 * 0.6) + (3.0 * 0.4)

    def test_combine_different_chunks_both_present(self, hybrid_engine):
        """Chunks exclusive to each engine must both appear in the combined output."""
        vector_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "v only", "score": 0.9, "search_type": "vector"},
        ]
        keyword_results = [
            {"doc_id": "doc2", "chunk_index": 1, "text": "k only", "score": 4.0, "search_type": "keyword"},
        ]
        combined = hybrid_engine._combine_results(vector_results, keyword_results, 0.6, 0.4)
        assert len(combined) == 2

    def test_combine_clips_negative_keyword_scores(self, hybrid_engine):
        """
        BM25 can return slightly negative scores. _combine_results() clips them
        to 0 so a weak keyword match never penalises the combined score.
        """
        vector_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "text", "score": 0.7, "search_type": "vector"},
        ]
        keyword_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "text", "score": -0.5, "search_type": "keyword"},
        ]
        combined = hybrid_engine._combine_results(vector_results, keyword_results, 0.6, 0.4)
        r = combined[0]
        assert r["keyword_score"] == 0
        # combined_score must not be reduced by the negative keyword score
        assert pytest.approx(r["combined_score"]) == 0.7 * 0.6

    # --- _deduplicate() and known gap ---

    def test_deduplicate_removes_lower_scoring_duplicate(self, hybrid_engine):
        results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "text", "combined_score": 0.5},
            {"doc_id": "doc1", "chunk_index": 0, "text": "text", "combined_score": 0.9},
        ]
        deduped = hybrid_engine._deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0]["combined_score"] == 0.9

    def test_deduplicate_known_gap_unreachable_in_practice(self, hybrid_engine):
        """
        KNOWN GAP: _deduplicate() is dead code in normal operation.

        _combine_results() builds a dict keyed by 'doc_id:chunk_index', which
        already ensures each chunk appears at most once. Any result passing
        through _combine_results() cannot contain true duplicates, so
        _deduplicate() will always be a no-op.

        This test confirms the gap: feeding _combine_results() a vector result
        and a keyword result for the *same* chunk produces exactly 1 entry —
        the deduplication already happened inside _combine_results().
        """
        vector_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "t", "score": 0.8, "search_type": "vector"},
        ]
        keyword_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "t", "score": 2.0, "search_type": "keyword"},
        ]
        combined = hybrid_engine._combine_results(vector_results, keyword_results, 0.6, 0.4)
        # Already deduplicated — _deduplicate() receives 1 item and returns 1 item
        assert len(combined) == 1
        deduped = hybrid_engine._deduplicate(combined)
        assert len(deduped) == 1

    # --- weight customisation ---

    @pytest.mark.asyncio
    async def test_custom_weights_applied(self, hybrid_engine):
        """Non-default weights must flow through to combined_score."""
        # Single-candidate pool: reranking a pool of size 1 can't reorder
        # anything, but it CAN overwrite/add a rerank_score - it does not
        # touch combined_score, which is what this test asserts on, so no
        # need to disable the reranker here. Documented for clarity since
        # the test below this one does need to disable it.
        vector_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "t", "score": 1.0, "search_type": "vector"},
        ]
        hybrid_engine._mock_vector.search = AsyncMock(return_value=vector_results)

        results = await hybrid_engine.search(
            "query", "user1", top_k=5,
            vector_weight=0.8,
            keyword_weight=0.2,
        )
        assert len(results) == 1
        assert pytest.approx(results[0]["combined_score"]) == 1.0 * 0.8

    # --- sorting ---

    @pytest.mark.asyncio
    async def test_results_ordered_by_combined_score_descending(self, hybrid_engine):
        # This test verifies the combine/dedupe/sort-by-combined_score
        # pipeline in isolation. Reranking (when the BGE model is
        # available) intentionally reorders the final result by
        # rerank_score instead, which is a deliberate behavior change
        # from Phase 14 - covered by its own dedicated reranker test,
        # not this one. Disabling it here keeps this test asserting on
        # what it was always meant to verify: the pre-rerank combined
        # score ordering produced by _combine_results/_deduplicate/sort.
        hybrid_engine.reranker.available = False

        vector_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "low", "score": 0.4, "search_type": "vector"},
            {"doc_id": "doc1", "chunk_index": 1, "text": "high", "score": 0.9, "search_type": "vector"},
            {"doc_id": "doc1", "chunk_index": 2, "text": "mid", "score": 0.6, "search_type": "vector"},
        ]
        hybrid_engine._mock_vector.search = AsyncMock(return_value=vector_results)

        results = await hybrid_engine.search("query", "user1", top_k=5)
        scores = [r["combined_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
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
    @pytest.mark.asyncio
    async def test_search_embeds_query_first(self, vector_engine):
        vector_engine._mock_db.search = AsyncMock(return_value=[])
        await vector_engine.search("what is revenue", "user1", top_k=5)
        vector_engine._mock_embedder.embed_text.assert_awaited_once_with(
            "what is revenue", task="search_query"
        )

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
            document_id=None,
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
        """
        HybridSearchEngine always fetches a fixed 20-candidate pool from
        each engine internally, regardless of the top_k the caller asked
        for — top_k only affects the final slice after fusion/reranking.
        document_id is forwarded through for document-scoped retrieval.
        """
        await hybrid_engine.search("revenue Q3", "user1", top_k=5)
        hybrid_engine._mock_vector.search.assert_awaited_once_with(
            "revenue Q3", "user1", top_k=20, document_id=None
        )

    @pytest.mark.asyncio
    async def test_keyword_engine_receives_correct_args(self, hybrid_engine):
        await hybrid_engine.search("revenue Q3", "user1", top_k=5)
        hybrid_engine._mock_keyword.search.assert_awaited_once_with(
            "user1", "revenue Q3", top_k=20, document_id=None
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

    # --- _reciprocal_rank_fusion() ---
    # 2026-xx-xx: renamed from _combine_results, weighted scoring (score *
    # vector_weight + score * keyword_weight) replaced with Reciprocal Rank
    # Fusion — score(d) = sum(1 / (RRF_K + rank + 1)) across whichever
    # result list(s) a chunk appears in. RRF is rank-based, not magnitude-
    # based, so there are no weight params anymore — see class docstring.

    def test_rrf_vector_only(self, hybrid_engine):
        vector_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "text", "score": 0.8, "search_type": "vector"},
        ]
        fused = hybrid_engine._reciprocal_rank_fusion(vector_results, [])
        assert len(fused) == 1
        r = fused[0]
        assert r["vector_score"] == 0.8
        assert r["keyword_score"] == 0
        assert pytest.approx(r["combined_score"]) == 1 / (60 + 0 + 1)

    def test_rrf_keyword_only(self, hybrid_engine):
        keyword_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "text", "score": 5.0, "search_type": "keyword"},
        ]
        fused = hybrid_engine._reciprocal_rank_fusion([], keyword_results)
        assert len(fused) == 1
        r = fused[0]
        assert r["vector_score"] == 0
        assert r["keyword_score"] == 5.0
        assert pytest.approx(r["combined_score"]) == 1 / (60 + 0 + 1)

    def test_rrf_same_chunk_both_engines_sums_contributions(self, hybrid_engine):
        """A chunk appearing in both lists at rank 0 gets a contribution
        from each list — merged, not duplicated."""
        shared_chunk = {"doc_id": "doc1", "chunk_index": 0, "text": "text"}
        vector_results = [{**shared_chunk, "score": 0.8, "search_type": "vector"}]
        keyword_results = [{**shared_chunk, "score": 3.0, "search_type": "keyword"}]

        fused = hybrid_engine._reciprocal_rank_fusion(vector_results, keyword_results)
        assert len(fused) == 1
        r = fused[0]
        assert r["vector_score"] == 0.8
        assert r["keyword_score"] == 3.0
        expected = 1 / (60 + 0 + 1) + 1 / (60 + 0 + 1)
        assert pytest.approx(r["combined_score"]) == expected

    def test_rrf_different_chunks_both_present(self, hybrid_engine):
        """Chunks exclusive to each engine must both appear in the fused output."""
        vector_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "v only", "score": 0.9, "search_type": "vector"},
        ]
        keyword_results = [
            {"doc_id": "doc2", "chunk_index": 1, "text": "k only", "score": 4.0, "search_type": "keyword"},
        ]
        fused = hybrid_engine._reciprocal_rank_fusion(vector_results, keyword_results)
        assert len(fused) == 2

    def test_rrf_clips_negative_keyword_scores_in_displayed_score_only(self, hybrid_engine):
        """
        BM25 can return slightly negative scores. keyword_score is clipped
        to 0 for display, but RRF's combined_score is rank-based, not
        magnitude-based — the negative underlying score doesn't reduce
        combined_score at all, since a list contributes 1/(k+rank+1)
        purely by virtue of the chunk appearing in that list.
        """
        vector_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "text", "score": 0.7, "search_type": "vector"},
        ]
        keyword_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "text", "score": -0.5, "search_type": "keyword"},
        ]
        fused = hybrid_engine._reciprocal_rank_fusion(vector_results, keyword_results)
        r = fused[0]
        assert r["keyword_score"] == 0
        expected = 1 / (60 + 0 + 1) + 1 / (60 + 0 + 1)
        assert pytest.approx(r["combined_score"]) == expected

    def test_rrf_lower_rank_contributes_less(self, hybrid_engine):
        """Sanity check on the RRF formula itself: a chunk at rank 1 in
        the vector list must score lower than one at rank 0."""
        vector_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "first", "score": 0.9, "search_type": "vector"},
            {"doc_id": "doc2", "chunk_index": 0, "text": "second", "score": 0.8, "search_type": "vector"},
        ]
        fused = hybrid_engine._reciprocal_rank_fusion(vector_results, [])
        by_id = {r["doc_id"]: r["combined_score"] for r in fused}
        assert by_id["doc1"] > by_id["doc2"]

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

        _reciprocal_rank_fusion() builds a dict keyed by
        'doc_id:chunk_index', which already ensures each chunk appears at
        most once. Any result passing through fusion cannot contain true
        duplicates, so _deduplicate() will always be a no-op.
        """
        vector_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "t", "score": 0.8, "search_type": "vector"},
        ]
        keyword_results = [
            {"doc_id": "doc1", "chunk_index": 0, "text": "t", "score": 2.0, "search_type": "keyword"},
        ]
        fused = hybrid_engine._reciprocal_rank_fusion(vector_results, keyword_results)
        assert len(fused) == 1
        deduped = hybrid_engine._deduplicate(fused)
        assert len(deduped) == 1

    # --- weight kwargs removed ---

    @pytest.mark.asyncio
    async def test_search_no_longer_accepts_weight_kwargs(self, hybrid_engine):
        """
        RRF replaced weighted score combination — search() no longer
        accepts vector_weight/keyword_weight at all. This documents the
        removal explicitly rather than letting it fail silently/unnoticed.
        """
        with pytest.raises(TypeError):
            await hybrid_engine.search(
                "query", "user1", top_k=5,
                vector_weight=0.8,
                keyword_weight=0.2,
            )

    # --- sorting ---

    @pytest.mark.asyncio
    async def test_results_ordered_by_combined_score_descending(self, hybrid_engine):
        # Reranking (when the BGE model is available) intentionally
        # reorders by rerank_score instead — covered by its own dedicated
        # reranker test, not this one. Disabling it here keeps this test
        # asserting on the pre-rerank RRF/dedupe/sort pipeline only.
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
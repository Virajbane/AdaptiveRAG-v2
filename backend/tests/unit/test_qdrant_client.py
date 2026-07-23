"""
Unit tests for app/db/qdrant/client.py  (QdrantVectorDB)

External dependency: qdrant_client.QdrantClient
Strategy: patch QdrantClient at the module level so that __init__ never
          touches a real Qdrant server. Each test controls what the mock
          client returns.

Known gaps / notes:
  - store_vectors() and search() are declared `async` but call the Qdrant
    SDK synchronously (no `await`). This is intentional — the sync SDK is
    used — but it means these methods block the event loop. An async SDK or
    run_in_executor() wrapper would be the production fix.

UPDATED 2026-07-23: _ensure_collection() previously had a bare `except:`
that silently swallowed ANY error probing Qdrant (misconfigured URL, auth
failure, network timeout) and fell through to create_collection(), losing
the original error. That gap is fixed — get_collections() failures are now
always re-raised (logged first if they're a known UnexpectedResponse /
ResponseHandlingException), so a misconfigured Qdrant connection fails
loudly at startup instead of silently. The existence check itself also
changed: it now lists all collections via get_collections() and checks
name membership, rather than probing a single collection by name.
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_qdrant_client():
    """
    Patch QdrantClient for the entire module so the constructor never
    attempts a real network connection. Defaults get_collections() to
    return no existing collections, so construction takes the
    create_collection() path unless a test overrides it.
    """
    with patch("app.db.qdrant.client.QdrantClient") as MockClient:
        instance = MagicMock()
        instance.get_collections.return_value = MagicMock(collections=[])
        MockClient.return_value = instance
        yield instance


@pytest.fixture
def vector_db(mock_qdrant_client):
    """Return a QdrantVectorDB with a fully mocked underlying client."""
    from app.db.qdrant.client import QdrantVectorDB
    return QdrantVectorDB(url="http://fake-qdrant:6333")


def fake_chunks(n: int = 3) -> list[dict]:
    return [
        {"text": f"chunk text {i}", "tokens": 10 + i, "chunk_index": i}
        for i in range(n)
    ]


def fake_embeddings(n: int = 3, dim: int = 768) -> list[list[float]]:
    return [[0.1 * i] * dim for i in range(n)]


# ---------------------------------------------------------------------------
# Construction / _ensure_collection
# ---------------------------------------------------------------------------

class TestQdrantVectorDBConstruction:
    def test_get_collections_called_on_init(self, mock_qdrant_client):
        """_ensure_collection() must list existing collections on construction."""
        from app.db.qdrant.client import QdrantVectorDB
        QdrantVectorDB(url="http://fake:6333")
        mock_qdrant_client.get_collections.assert_called_once()

    def test_create_collection_called_when_absent_from_list(self, mock_qdrant_client):
        """If the collection isn't in the listed collections, create_collection() must be called."""
        from app.db.qdrant.client import QdrantVectorDB
        QdrantVectorDB(url="http://fake:6333")

        mock_qdrant_client.create_collection.assert_called_once()
        call_kwargs = mock_qdrant_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "documents_embeddings"

    def test_create_collection_not_called_when_present_in_list(self, mock_qdrant_client):
        """If the collection is already listed, create_collection() must NOT be called."""
        existing = MagicMock()
        existing.name = "documents_embeddings"
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[existing])

        from app.db.qdrant.client import QdrantVectorDB
        QdrantVectorDB(url="http://fake:6333")
        mock_qdrant_client.create_collection.assert_not_called()

    def test_get_collections_failure_propagates(self):
        """
        Any exception from get_collections() now propagates rather than
        being silently swallowed - a misconfigured/unreachable Qdrant
        instance must fail loudly at startup.
        """
        with patch("app.db.qdrant.client.QdrantClient") as MockClient:
            instance = MagicMock()
            instance.get_collections.side_effect = ConnectionRefusedError("unreachable")
            MockClient.return_value = instance

            from app.db.qdrant.client import QdrantVectorDB
            with pytest.raises(ConnectionRefusedError):
                QdrantVectorDB(url="http://fake:6333")


# ---------------------------------------------------------------------------
# store_vectors
# ---------------------------------------------------------------------------

class TestStoreVectors:
    @pytest.mark.asyncio
    async def test_returns_count_of_stored_points(self, vector_db):
        chunks = fake_chunks(3)
        embeddings = fake_embeddings(3)
        result = await vector_db.store_vectors("doc1", "user1", chunks, embeddings)
        assert result["stored_count"] == 3
        assert result["failed_count"] == 0
        assert result["failed_chunk_indices"] == []

    @pytest.mark.asyncio
    async def test_upsert_called_once(self, vector_db, mock_qdrant_client):
        await vector_db.store_vectors("doc1", "user1", fake_chunks(2), fake_embeddings(2))
        mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_targets_correct_collection(self, vector_db, mock_qdrant_client):
        await vector_db.store_vectors("doc1", "user1", fake_chunks(1), fake_embeddings(1))
        call_kwargs = mock_qdrant_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "documents_embeddings"

    @pytest.mark.asyncio
    async def test_payload_contains_required_fields(self, vector_db, mock_qdrant_client):
        chunks = [{"text": "hello world", "tokens": 5, "chunk_index": 0}]
        embeddings = [[0.1] * 768]
        await vector_db.store_vectors("doc99", "user42", chunks, embeddings)

        points = mock_qdrant_client.upsert.call_args[1]["points"]
        assert len(points) == 1
        payload = points[0].payload
        assert payload["doc_id"] == "doc99"
        assert payload["user_id"] == "user42"
        assert payload["chunk_index"] == 0
        assert payload["chunk_text"] == "hello world"
        assert payload["tokens"] == 5
        assert payload["namespace"] == "user_user42"

    @pytest.mark.asyncio
    async def test_each_point_gets_unique_uuid(self, vector_db, mock_qdrant_client):
        chunks = fake_chunks(3)
        await vector_db.store_vectors("doc1", "user1", chunks, fake_embeddings(3))
        points = mock_qdrant_client.upsert.call_args[1]["points"]
        ids = [p.id for p in points]
        assert len(set(ids)) == 3  # all unique

    @pytest.mark.asyncio
    async def test_point_id_is_deterministic_per_doc_and_chunk_index(self, vector_db, mock_qdrant_client):
        """
        Point IDs are now derived deterministically (uuid5) from
        (doc_id, chunk_index) instead of a random uuid4, so re-running
        store_vectors for the same doc/chunk on retry replaces the
        existing vector instead of creating a duplicate.
        """
        chunks = [{"text": "same chunk", "tokens": 5, "chunk_index": 0}]
        embeddings = [[0.1] * 768]

        await vector_db.store_vectors("doc1", "user1", chunks, embeddings)
        first_id = mock_qdrant_client.upsert.call_args[1]["points"][0].id

        await vector_db.store_vectors("doc1", "user1", chunks, embeddings)
        second_id = mock_qdrant_client.upsert.call_args[1]["points"][0].id

        assert first_id == second_id

    @pytest.mark.asyncio
    async def test_zero_chunks_stores_nothing(self, vector_db, mock_qdrant_client):
        """
        With no chunks, the batching loop (range over 0 points) never
        executes a single iteration, so upsert() is not called at all -
        not called-with-empty-points as in the previous implementation.
        """
        result = await vector_db.store_vectors("doc1", "user1", [], [])
        assert result["stored_count"] == 0
        mock_qdrant_client.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def _make_mock_point(self, score, doc_id, chunk_index, text, tokens):
        point = MagicMock()
        point.score = score
        point.payload = {
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "chunk_text": text,
            "tokens": tokens,
        }
        return point

    @pytest.mark.asyncio
    async def test_search_returns_correct_shape(self, vector_db, mock_qdrant_client):
        # search() calls self.client.search(...) directly and iterates the
        # returned list - there is no wrapping `.points` response object.
        mock_qdrant_client.search.return_value = [
            self._make_mock_point(0.95, "doc1", 0, "relevant text", 12),
        ]

        results = await vector_db.search([0.1] * 768, "user1", top_k=5)

        assert len(results) == 1
        r = results[0]
        assert r["score"] == 0.95
        assert r["doc_id"] == "doc1"
        assert r["chunk_index"] == 0
        assert r["text"] == "relevant text"
        assert r["tokens"] == 12

    @pytest.mark.asyncio
    async def test_search_passes_user_id_filter(self, vector_db, mock_qdrant_client):
        """User isolation: query_filter must restrict results to user_id."""
        mock_qdrant_client.search.return_value = []

        await vector_db.search([0.0] * 768, "user_abc", top_k=3)

        call_kwargs = mock_qdrant_client.search.call_args[1]
        # query_filter is a real qdrant_client Filter object, not a dict.
        query_filter = call_kwargs["query_filter"]
        must_clause = query_filter.must[0]
        assert must_clause.key == "user_id"
        assert must_clause.match.value == "user_abc"

    @pytest.mark.asyncio
    async def test_search_passes_top_k_as_limit(self, vector_db, mock_qdrant_client):
        mock_qdrant_client.search.return_value = []

        await vector_db.search([0.0] * 768, "user1", top_k=7)

        call_kwargs = mock_qdrant_client.search.call_args[1]
        assert call_kwargs["limit"] == 7

    @pytest.mark.asyncio
    async def test_search_empty_response(self, vector_db, mock_qdrant_client):
        mock_qdrant_client.search.return_value = []
        results = await vector_db.search([0.0] * 768, "user1")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_passes_document_id_filter_when_provided(self, vector_db, mock_qdrant_client):
        """
        New: search() now accepts an optional document_id, which adds a
        second must-clause filtering results down to a single document
        (used when a question is scoped to one uploaded doc).
        """
        mock_qdrant_client.search.return_value = []

        await vector_db.search([0.0] * 768, "user1", top_k=5, document_id="doc42")

        call_kwargs = mock_qdrant_client.search.call_args[1]
        query_filter = call_kwargs["query_filter"]
        keys = {clause.key: clause.match.value for clause in query_filter.must}
        assert keys["user_id"] == "user1"
        assert keys["doc_id"] == "doc42"

    @pytest.mark.asyncio
    async def test_search_omits_doc_filter_when_document_id_not_provided(self, vector_db, mock_qdrant_client):
        mock_qdrant_client.search.return_value = []

        await vector_db.search([0.0] * 768, "user1", top_k=5)

        call_kwargs = mock_qdrant_client.search.call_args[1]
        query_filter = call_kwargs["query_filter"]
        assert len(query_filter.must) == 1
        assert query_filter.must[0].key == "user_id"


# ---------------------------------------------------------------------------
# delete_document_vectors
# ---------------------------------------------------------------------------

class TestDeleteDocumentVectors:
    @pytest.mark.asyncio
    async def test_returns_delete_status(self, vector_db, mock_qdrant_client):
        """
        delete_document_vectors returns result.status (the Qdrant
        operation status), not a deleted-point count - Qdrant's delete()
        response doesn't include a point count.
        """
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_qdrant_client.delete.return_value = mock_result

        status = await vector_db.delete_document_vectors("doc1", "user1")
        assert status == "completed"

    @pytest.mark.asyncio
    async def test_delete_filters_by_doc_and_user(self, vector_db, mock_qdrant_client):
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_qdrant_client.delete.return_value = mock_result

        await vector_db.delete_document_vectors("doc99", "user42")

        call_kwargs = mock_qdrant_client.delete.call_args[1]
        # points_selector is a real Filter object, not a dict.
        must = call_kwargs["points_selector"].must
        keys = {clause.key: clause.match.value for clause in must}
        assert keys["doc_id"] == "doc99"
        assert keys["user_id"] == "user42"
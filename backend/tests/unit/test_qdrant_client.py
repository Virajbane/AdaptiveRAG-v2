"""
Unit tests for app/db/qdrant/client.py  (QdrantVectorDB)

External dependency: qdrant_client.QdrantClient
Strategy: patch QdrantClient at the module level so that __init__ never
          touches a real Qdrant server. Each test controls what the mock
          client returns.

Known gaps documented:
  - _ensure_collection() uses a bare `except:` clause — any exception
    (misconfigured URL, auth failure, network timeout) is silently swallowed
    and the code falls through to create_collection(). If create_collection()
    also fails, that exception *does* propagate, but the original error is lost.
  - store_vectors() and search() are declared `async` but call the Qdrant
    SDK synchronously (no `await`). This is intentional — the sync SDK is
    used — but it means these methods block the event loop. An async SDK or
    run_in_executor() wrapper would be the production fix.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_qdrant_client():
    """
    Patch QdrantClient for the entire module so the constructor never
    attempts a real network connection.
    """
    with patch("app.db.qdrant.client.QdrantClient") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        yield instance


@pytest.fixture
def vector_db(mock_qdrant_client):
    """Return a QdrantVectorDB with a fully mocked underlying client."""
    from app.db.qdrant.client import QdrantVectorDB
    return QdrantVectorDB(url="http://fake-qdrant:6333")


def fake_chunks(n: int = 3) -> list[dict]:
    return [{"text": f"chunk text {i}", "tokens": 10 + i} for i in range(n)]


def fake_embeddings(n: int = 3, dim: int = 768) -> list[list[float]]:
    return [[0.1 * i] * dim for i in range(n)]


# ---------------------------------------------------------------------------
# Construction / _ensure_collection
# ---------------------------------------------------------------------------

class TestQdrantVectorDBConstruction:
    def test_get_collection_called_on_init(self, mock_qdrant_client):
        """_ensure_collection() must probe Qdrant on construction."""
        from app.db.qdrant.client import QdrantVectorDB
        QdrantVectorDB(url="http://fake:6333")
        mock_qdrant_client.get_collection.assert_called_once_with("documents_embeddings")

    def test_create_collection_called_when_get_raises(self):
        """If the collection does not exist, create_collection() must be called."""
        with patch("app.db.qdrant.client.QdrantClient") as MockClient:
            instance = MagicMock()
            instance.get_collection.side_effect = Exception("collection not found")
            MockClient.return_value = instance

            from app.db.qdrant.client import QdrantVectorDB
            QdrantVectorDB(url="http://fake:6333")

            instance.create_collection.assert_called_once()
            call_kwargs = instance.create_collection.call_args[1]
            assert call_kwargs["collection_name"] == "documents_embeddings"

    def test_create_collection_not_called_when_get_succeeds(self, mock_qdrant_client):
        """If the collection already exists, create_collection() must NOT be called."""
        mock_qdrant_client.get_collection.return_value = MagicMock()
        from app.db.qdrant.client import QdrantVectorDB
        QdrantVectorDB(url="http://fake:6333")
        mock_qdrant_client.create_collection.assert_not_called()

    def test_silent_swallow_known_gap(self):
        """
        KNOWN GAP: bare `except:` in _ensure_collection() silently discards
        any exception from get_collection() — including misconfigured URLs,
        auth failures, or network timeouts — before attempting create_collection().
        The original error is lost. This test confirms the current (flawed)
        behaviour so a future fix is detectable.
        """
        with patch("app.db.qdrant.client.QdrantClient") as MockClient:
            instance = MagicMock()
            instance.get_collection.side_effect = ConnectionRefusedError("unreachable")
            # create_collection succeeds — no exception should propagate
            MockClient.return_value = instance
            from app.db.qdrant.client import QdrantVectorDB
            db = QdrantVectorDB(url="http://fake:6333")  # must not raise
            assert db is not None


# ---------------------------------------------------------------------------
# store_vectors
# ---------------------------------------------------------------------------

class TestStoreVectors:
    @pytest.mark.asyncio
    async def test_returns_count_of_stored_points(self, vector_db):
        chunks = fake_chunks(3)
        embeddings = fake_embeddings(3)
        count = await vector_db.store_vectors("doc1", "user1", chunks, embeddings)
        assert count == 3

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
        chunks = [{"text": "hello world", "tokens": 5}]
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
    async def test_zero_chunks_stores_nothing(self, vector_db, mock_qdrant_client):
        count = await vector_db.store_vectors("doc1", "user1", [], [])
        assert count == 0
        # upsert is still called but with an empty points list
        points = mock_qdrant_client.upsert.call_args[1]["points"]
        assert points == []


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
        mock_response = MagicMock()
        mock_response.points = [
            self._make_mock_point(0.95, "doc1", 0, "relevant text", 12),
        ]
        mock_qdrant_client.query_points.return_value = mock_response

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
        """User isolation: the query_filter must restrict results to user_id."""
        mock_response = MagicMock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response

        await vector_db.search([0.0] * 768, "user_abc", top_k=3)

        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        query_filter = call_kwargs["query_filter"]
        must_clause = query_filter["must"][0]
        assert must_clause["key"] == "user_id"
        assert must_clause["match"]["value"] == "user_abc"

    @pytest.mark.asyncio
    async def test_search_passes_top_k_as_limit(self, vector_db, mock_qdrant_client):
        mock_response = MagicMock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response

        await vector_db.search([0.0] * 768, "user1", top_k=7)

        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs["limit"] == 7

    @pytest.mark.asyncio
    async def test_search_empty_response(self, vector_db, mock_qdrant_client):
        mock_response = MagicMock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response
        results = await vector_db.search([0.0] * 768, "user1")
        assert results == []


# ---------------------------------------------------------------------------
# delete_document_vectors
# ---------------------------------------------------------------------------

class TestDeleteDocumentVectors:
    @pytest.mark.asyncio
    async def test_returns_deleted_count(self, vector_db, mock_qdrant_client):
        mock_result = MagicMock()
        mock_result.deleted = 5
        mock_qdrant_client.delete.return_value = mock_result

        count = await vector_db.delete_document_vectors("doc1", "user1")
        assert count == 5

    @pytest.mark.asyncio
    async def test_delete_filters_by_doc_and_user(self, vector_db, mock_qdrant_client):
        mock_result = MagicMock()
        mock_result.deleted = 0
        mock_qdrant_client.delete.return_value = mock_result

        await vector_db.delete_document_vectors("doc99", "user42")

        call_kwargs = mock_qdrant_client.delete.call_args[1]
        must = call_kwargs["points_selector"]["filter"]["must"]
        keys = {clause["key"]: clause["match"]["value"] for clause in must}
        assert keys["doc_id"] == "doc99"
        assert keys["user_id"] == "user42"
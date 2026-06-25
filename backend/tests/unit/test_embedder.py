"""
Unit tests for EmbeddingGenerator.

Ollama is mocked throughout - a real embedding call depends on a
locally running Ollama server, which makes tests slow and flaky.
Mocking keeps these fast and deterministic, and isolates testing to
the EmbeddingGenerator's own logic (batching, error handling) rather
than Ollama's actual model behavior.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.document.embedder import EmbeddingGenerator


FAKE_EMBEDDING = [0.1] * 768


@pytest.fixture
def embedder():
    return EmbeddingGenerator()


@pytest.mark.asyncio
async def test_embed_text_returns_vector_of_correct_dimension(embedder):
    with patch.object(embedder, "_embed_sync", return_value=FAKE_EMBEDDING):
        result = await embedder.embed_text("hello world")
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)


@pytest.mark.asyncio
async def test_embed_text_propagates_errors_as_value_error(embedder):
    with patch.object(
        embedder, "_embed_sync", side_effect=RuntimeError("Ollama unreachable")
    ):
        with pytest.raises(ValueError, match="Error embedding text"):
            await embedder.embed_text("hello world")


@pytest.mark.asyncio
async def test_embed_batch_calls_embed_text_once_per_item(embedder):
    texts = ["chunk one", "chunk two", "chunk three"]

    with patch.object(embedder, "_embed_sync", return_value=FAKE_EMBEDDING) as mock_sync:
        results = await embedder.embed_batch(texts)

        assert len(results) == 3
        assert mock_sync.call_count == 3
        for vec in results:
            assert len(vec) == 768


@pytest.mark.asyncio
async def test_embed_batch_aborts_entirely_on_single_failure(embedder):
    """
    KNOWN BEHAVIOR (not necessarily desired): embed_batch has no
    per-item error isolation. If any single item in the batch fails,
    the whole batch raises and earlier successfully-embedded items
    are discarded by the caller, since embed_batch never returns
    partial results.

    This test documents that current behavior. If error isolation
    is added later (e.g. returning partial results + a list of
    failures), this test should be updated to match the new contract.
    """
    call_count = {"n": 0}

    def flaky_embed(text):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated failure on second item")
        return FAKE_EMBEDDING

    with patch.object(embedder, "_embed_sync", side_effect=flaky_embed):
        with pytest.raises(ValueError):
            await embedder.embed_batch(["first", "second", "third"])

        # Only 2 calls happened before the failure - the third item
        # was never attempted, confirming the all-or-nothing behavior
        assert call_count["n"] == 2
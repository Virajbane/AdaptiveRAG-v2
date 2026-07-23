"""
Unit tests for EmbeddingGenerator.

Ollama is mocked throughout - a real embedding call depends on a
locally running Ollama server, which makes tests slow and flaky.
Mocking keeps these fast and deterministic, and isolates testing to
the EmbeddingGenerator's own logic (batching, retries, error handling)
rather than Ollama's actual model behavior.

These tests assume no VOYAGE_API_KEY is set in the test environment,
so EmbeddingGenerator.use_voyage is False and the Ollama code path
(_embed_sync_ollama) is the one exercised. If Voyage is enabled in CI
in the future, the equivalent _embed_sync_voyage tests should be added
alongside these rather than replacing them.
"""

import pytest
from unittest.mock import patch

from app.services.document.embedder import EmbeddingGenerator


FAKE_EMBEDDING = [0.1] * 768


@pytest.fixture
def embedder():
    return EmbeddingGenerator()


@pytest.mark.asyncio
async def test_embed_text_returns_vector_of_correct_dimension(embedder):
    with patch.object(embedder, "_embed_sync_ollama", return_value=FAKE_EMBEDDING):
        result = await embedder.embed_text("hello world")
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)


@pytest.mark.asyncio
async def test_embed_text_propagates_errors_as_value_error(embedder):
    with patch.object(
        embedder, "_embed_sync_ollama", side_effect=RuntimeError("Ollama unreachable")
    ):
        with pytest.raises(ValueError, match="Error embedding text"):
            await embedder.embed_text("hello world")


@pytest.mark.asyncio
async def test_embed_batch_calls_embed_text_once_per_item(embedder):
    texts = ["chunk one", "chunk two", "chunk three"]

    with patch.object(embedder, "_embed_sync_ollama", return_value=FAKE_EMBEDDING) as mock_sync:
        result = await embedder.embed_batch(texts)

        assert len(result["embeddings"]) == 3
        assert result["failed_indices"] == []
        assert mock_sync.call_count == 3
        for vec in result["embeddings"]:
            assert len(vec) == 768


@pytest.mark.asyncio
async def test_embed_batch_isolates_failures_per_item(embedder):
    """
    UPDATED 2026-07-23: previously embed_batch had no per-item error
    isolation - any single failing item raised and aborted the whole
    batch (see git history for the old
    'test_embed_batch_aborts_entirely_on_single_failure' version of
    this test).

    embed_batch now retries each item up to max_retries times, and an
    item that still fails after exhausting retries is recorded in
    failed_indices rather than raising - the rest of the batch still
    gets embedded and returned. base_delay=0 is passed to skip the
    real retry backoff sleeps and keep the test fast.
    """
    def flaky_embed(text, task):
        if text == "second":
            raise RuntimeError("simulated permanent failure")
        return FAKE_EMBEDDING

    with patch.object(embedder, "_embed_sync_ollama", side_effect=flaky_embed):
        result = await embedder.embed_batch(
            ["first", "second", "third"], base_delay=0
        )

        assert result["failed_indices"] == [1]
        assert result["embeddings"][0] == FAKE_EMBEDDING
        assert result["embeddings"][1] is None
        assert result["embeddings"][2] == FAKE_EMBEDDING
"""
Unit tests for TextChunker.

No external services involved - tiktoken runs locally - so these
run fast and need no mocking.
"""

import pytest
from backend.app.services.document.chunker import TextChunker


@pytest.fixture
def chunker():
    return TextChunker()


def test_count_tokens_basic(chunker):
    count = chunker.count_tokens("Hello world")
    assert count > 0
    assert isinstance(count, int)


def test_count_tokens_empty_string(chunker):
    assert chunker.count_tokens("") == 0


def test_chunk_short_text_produces_one_chunk(chunker):
    text = "This is a short sentence. It has two sentences."
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert chunks[0]["tokens"] <= chunker.chunk_size


def test_chunk_empty_text_produces_no_chunks(chunker):
    chunks = chunker.chunk("")
    assert chunks == []


def test_chunk_respects_target_size_for_normal_text(chunker):
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * 200
    chunks = chunker.chunk(text)

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk["tokens"] <= chunker.chunk_size + 50


def test_chunk_preserves_all_content_roughly(chunker):
    text = (
        "Apples are red. Bananas are yellow. Cherries are also red. "
        "Dates grow on palm trees. Elderberries are dark purple."
    )
    chunks = chunker.chunk(text)
    combined = " ".join(c["text"] for c in chunks)

    assert "Apples are red" in combined
    assert "Elderberries are dark purple" in combined


def test_chunk_multiple_chunks_each_within_tolerance(chunker):
    sentence = "Sentence number {} appears here for overlap testing. "
    text = "".join(sentence.format(i) for i in range(100))
    chunks = chunker.chunk(text)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk["tokens"] <= chunker.chunk_size + 50


def test_single_oversized_sentence_is_not_split(chunker):
    """
    KNOWN GAP: a single 'sentence' with no terminal punctuation that
    exceeds chunk_size is NOT broken down further by the current
    implementation - it becomes one oversized chunk on its own.

    This documents actual current behavior. If this test starts
    failing, it likely means sentence-splitting for oversized
    sentences was added - a good change - and this test should be
    updated to assert the new correct behavior instead.
    """
    huge_sentence = "word " * 700
    chunks = chunker.chunk(huge_sentence)

    assert len(chunks) == 1
    assert chunks[0]["tokens"] > chunker.chunk_size
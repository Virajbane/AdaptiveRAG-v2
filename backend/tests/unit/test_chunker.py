"""
Unit tests for TextChunker.

No external services involved - tiktoken runs locally - so these
run fast and need no mocking.
"""

import pytest
from app.services.document.chunker import TextChunker


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
    assert chunks[0]["tokens"] <= chunker.max_tokens


def test_chunk_empty_text_produces_no_chunks(chunker):
    chunks = chunker.chunk("")
    assert chunks == []


def test_chunk_respects_target_size_for_normal_text(chunker):
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * 200
    chunks = chunker.chunk(text)

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk["tokens"] <= chunker.max_tokens + 50


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
        assert chunk["tokens"] <= chunker.max_tokens + 50


def test_oversized_sentence_with_no_punctuation_is_split_by_words(chunker):
    """
    UPDATED 2026-07-23: previously a single 'sentence' with no terminal
    punctuation that exceeded max_tokens was NOT broken down further and
    became one oversized chunk (see git history for the old
    'test_single_oversized_sentence_is_not_split' version of this test).

    _split_recursive now falls through to the space (" ") separator when
    no punctuation-based separator applies, so long punctuation-less text
    gets split word-by-word and re-merged into normal-sized chunks like
    any other text, instead of being left as one oversized blob.
    """
    huge_sentence = "word " * 700
    chunks = chunker.chunk(huge_sentence)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["tokens"] <= chunker.max_tokens + 50
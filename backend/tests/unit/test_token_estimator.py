"""
Unit tests for token_estimator.estimate_tokens / total_tokens.

These test the heuristic in isolation -- they do NOT tell you whether
memory trimming happens at a "good" moment for RAG quality. See
test_memory_eviction_recall.py for that (it exercises the estimator
inside the actual eviction -> summarize -> recall pipeline).
"""
import pytest
from app.services.memory.token_utils import estimate_tokens, total_tokens


class TestEstimateTokens:
    def test_empty_string_is_zero(self):
        assert estimate_tokens("") == 0

    def test_none_like_falsy_is_zero(self):
        assert estimate_tokens(None) == 0  # guarded by `if not text`

    def test_short_string_floors_to_one_not_zero(self):
        # "hi" is 2 chars -> 2//4 == 0, but function guarantees >=1
        # for any non-empty text so a real message never "disappears"
        # from the token budget.
        assert estimate_tokens("hi") == 1

    def test_matches_chars_per_token_ratio(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_rounds_down_within_a_token(self):
        text = "a" * 403  # 403 // 4 == 100 (not 101)
        assert estimate_tokens(text) == 100

    def test_does_not_crash_on_unicode(self):
        # CJK / emoji chars are undercounted by a char-based heuristic
        # (each char is usually its own token, sometimes several) --
        # this test documents that known blind spot rather than
        # asserting a "correct" value, since there isn't one without
        # a real tokenizer.
        text = "你好世界" * 50  # 200 chars
        result = estimate_tokens(text)
        assert result == 50  # 200 // 4 -- likely a real UNDERestimate for CJK

    def test_dense_numeric_table_text(self):
        # Mirrors the golden-set table_fig_* content style: short,
        # symbol/number-dense strings. The module's own docstring
        # claims this undercounts richness but "errs safe" by
        # triggering summarization earlier -- verify the char-based
        # estimate for table text tracks the char-based estimate for
        # plain prose of the SAME length (i.e. the heuristic itself
        # doesn't special-case content shape -- it's purely length
        # based, which is the documented, known limitation).
        table = "46.2 51.5 100 27.6 86.3 78.0" * 7
        prose = "a" * len(table)  # match length exactly, don't hardcode it
        assert estimate_tokens(prose) == estimate_tokens(table)
        assert estimate_tokens(table) == len(table) // 4


class TestTotalTokens:
    def test_empty_message_list(self):
        assert total_tokens([]) == 0

    def test_sums_across_messages(self):
        messages = [
            {"role": "user", "content": "a" * 40},   # 10 tokens
            {"role": "assistant", "content": "a" * 80},  # 20 tokens
        ]
        assert total_tokens(messages) == 30

    def test_missing_content_key_treated_as_empty(self):
        messages = [{"role": "user"}]  # no "content"
        assert total_tokens(messages) == 0

    def test_missing_content_does_not_crash_the_whole_sum(self):
        messages = [
            {"role": "user", "content": "a" * 40},
            {"role": "assistant"},  # malformed, should just contribute 0
        ]
        assert total_tokens(messages) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
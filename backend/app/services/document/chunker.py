import re
from typing import List
import tiktoken

class TextChunker:
    """
    Recursive character-based chunker.
    Works for any document type: resume, PDF, research paper, CSV, code, etc.
    No assumptions about structure — splits by paragraphs → sentences → words.

    2026-07-04 fix — separator preservation:
      _split_recursive previously used text.split(separator), which
      discards the separator from every piece, and _merge_with_overlap
      rejoined surviving pieces with a single " ". Net effect: chunk
      text silently lost its own punctuation and structure on rejoin —
      "Sentence one. Sentence two." (split on ". ") became "Sentence
      one Sentence two." (period after "one" gone), and the same
      applied to every "\\n\\n" paragraph break, "\\n" line break, and
      other separator in the list except the trailing word-level " ".
      This degrades embedding/retrieval quality corpus-wide since
      chunk text no longer matches the source document's real
      structure.

      Fix: each split now carries its own trailing separator forward
      (attached to whichever piece — direct or recursively-derived —
      ends up last before that separator in the original text), and
      merging concatenates with "" instead of " ", since separators
      are already embedded in the pieces themselves.
    """

    def __init__(self, model: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(model)
        self.max_tokens = 150      # sweet spot: not too large, not too small
        self.overlap_tokens = 50   # carry context across chunks

        # Priority order: try largest separator first, fall back to smaller.
        # No trailing "" (character) entry here — see _hard_split_by_tokens,
        # which is the real fallback for text with no separators at all.
        self.separators = [
            "\n\n",   # paragraph break
            "\n",     # line break (handles bullets, resume lines)
            ". ",     # sentence
            "? ",
            "! ",
            "; ",
            ", ",
            " ",      # word (last resort)
        ]

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chunk(self, text: str) -> List[dict]:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text).strip()
        raw_chunks = self._split_recursive(text, self.separators)
        merged = self._merge_with_overlap(raw_chunks)
        result = [{"text": c.strip(), "tokens": self.count_tokens(c)} for c in merged if c.strip()]
        # DEBUG
        print(f"[CHUNKER DEBUG] raw splits: {len(raw_chunks)}, merged chunks: {len(result)}")
        for i, ch in enumerate(result):
            print(f"[CHUNKER DEBUG] chunk {i+1} ({ch['tokens']} tokens): {ch['text'][:80]}")
        return result

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Try splitting by current separator.
        If any piece is still too large, recurse with next separator.

        Each returned piece carries the separator that followed it in
        the original text (except the very last piece, which had
        nothing after it) — so downstream joining with "" reconstructs
        the original punctuation/structure instead of losing it.
        """
        if not text.strip():
            return []

        if not separators:
            # No separator left to try (text has no spaces/punctuation at
            # all — e.g. a long URL, hash, or base64 blob — and still
            # exceeds max_tokens). Split on raw token boundaries instead
            # of falling back to individual characters.
            return self._hard_split_by_tokens(text)

        separator = separators[0]
        remaining = separators[1:]

        splits = text.split(separator)

        result = []
        for idx, raw_split in enumerate(splits):
            piece = raw_split.strip()
            if not piece:
                continue

            # This piece had `separator` after it in the original text,
            # unless it's the last split from this split() call.
            trailing = separator if idx < len(splits) - 1 else ""

            if self.count_tokens(piece) > self.max_tokens:
                # Still too large — recurse with smaller separator.
                sub_pieces = self._split_recursive(piece, remaining)
                if sub_pieces and trailing:
                    # Reattach this level's separator to the last
                    # sub-piece so the boundary isn't lost between this
                    # recursively-split group and the next top-level split.
                    sub_pieces[-1] = sub_pieces[-1] + trailing
                result.extend(sub_pieces)
            else:
                result.append(piece + trailing)

        return result

    def _hard_split_by_tokens(self, text: str) -> List[str]:
        """
        Absolute fallback for a single unbreakable span (no separators of
        any kind) that still exceeds max_tokens. Encodes to tokens, slices
        on token boundaries, decodes back — so pieces are correctly sized
        without needing any separator reattachment (there was none to begin
        with).
        """
        tokens = self.encoding.encode(text)
        return [
            self.encoding.decode(tokens[i:i + self.max_tokens])
            for i in range(0, len(tokens), self.max_tokens)
        ]

    def _merge_with_overlap(self, splits: List[str]) -> List[str]:
        """
        Merge small splits into chunks up to max_tokens.
        Add overlap from previous chunk for context continuity.

        Splits now carry their own trailing separators (see
        _split_recursive), so pieces are concatenated with "" rather
        than forcibly re-inserting " " between everything.
        """
        chunks = []
        current = []
        current_tokens = 0
        overlap_buffer = []  # last N tokens from previous chunk

        for split in splits:
            split_tokens = self.count_tokens(split)

            if current_tokens + split_tokens > self.max_tokens:
                if current:
                    chunk_text = "".join(current)
                    chunks.append(chunk_text)
                    # Build overlap buffer from end of this chunk
                    overlap_buffer = self._get_overlap_buffer(current)

                # Start new chunk with overlap
                current = overlap_buffer + [split]
                current_tokens = self.count_tokens("".join(current))
                overlap_buffer = []
            else:
                current.append(split)
                current_tokens += split_tokens

        if current:
            chunks.append("".join(current))

        return chunks

    def _get_overlap_buffer(self, splits: List[str]) -> List[str]:
        """Return last splits that fit within overlap_tokens budget."""
        buffer = []
        token_count = 0
        for split in reversed(splits):
            t = self.count_tokens(split)
            if token_count + t <= self.overlap_tokens:
                buffer.insert(0, split)
                token_count += t
            else:
                break
        return buffer
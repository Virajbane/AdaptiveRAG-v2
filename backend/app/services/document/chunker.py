import re
from typing import List
import tiktoken

class TextChunker:
    """
    Recursive character-based chunker.
    Works for any document type: resume, PDF, research paper, CSV, code, etc.
    No assumptions about structure — splits by paragraphs → sentences → words.
    """

    def __init__(self, model: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(model)
        self.max_tokens = 150      # sweet spot: not too large, not too small
        self.overlap_tokens = 50   # carry context across chunks

        # Priority order: try largest separator first, fall back to smaller
        self.separators = [
            "\n\n",   # paragraph break
            "\n",     # line break (handles bullets, resume lines)
            ". ",     # sentence
            "? ",
            "! ",
            "; ",
            ", ",
            " ",      # word (last resort)
            ""        # character (absolute fallback)
        ]

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chunk(self, text: str) -> List[dict]:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text).strip()
        raw_chunks = self._split_recursive(text, self.separators)
        merged = self._merge_with_overlap(raw_chunks)
        result = [{"text": c, "tokens": self.count_tokens(c)} for c in merged if c.strip()]
        # DEBUG
        print(f"[CHUNKER DEBUG] raw splits: {len(raw_chunks)}, merged chunks: {len(result)}")
        for i, ch in enumerate(result):
            print(f"[CHUNKER DEBUG] chunk {i+1} ({ch['tokens']} tokens): {ch['text'][:80]}")
        return result

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Try splitting by current separator.
        If any piece is still too large, recurse with next separator.
        """
        if not text.strip():
            return []

        if not separators:
            # Absolute fallback: split by characters
            return [text[i:i+100] for i in range(0, len(text), 100)]

        separator = separators[0]
        remaining = separators[1:]

        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        result = []
        for split in splits:
            split = split.strip()
            if not split:
                continue
            if self.count_tokens(split) > self.max_tokens:
                # Still too large — recurse with smaller separator
                result.extend(self._split_recursive(split, remaining))
            else:
                result.append(split)

        return result

    def _merge_with_overlap(self, splits: List[str]) -> List[str]:
        """
        Merge small splits into chunks up to max_tokens.
        Add overlap from previous chunk for context continuity.
        """
        chunks = []
        current = []
        current_tokens = 0
        overlap_buffer = []  # last N tokens from previous chunk

        for split in splits:
            split_tokens = self.count_tokens(split)

            if current_tokens + split_tokens > self.max_tokens:
                if current:
                    chunk_text = " ".join(current)
                    chunks.append(chunk_text)
                    # Build overlap buffer from end of this chunk
                    overlap_buffer = self._get_overlap_buffer(current)

                # Start new chunk with overlap
                current = overlap_buffer + [split]
                current_tokens = self.count_tokens(" ".join(current))
                overlap_buffer = []
            else:
                current.append(split)
                current_tokens += split_tokens

        if current:
            chunks.append(" ".join(current))

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
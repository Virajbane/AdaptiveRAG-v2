import re
from typing import List, Optional
from app.utils.tokenization import count_tokens as _shared_count_tokens, using_groq

class TextChunker:
    """
    Recursive character-based chunker.

    2026-07-04 fix — separator preservation (see chunk()/. _split_recursive).

    2026-07-05 fix — table fragmentation:
      Dense numeric tables were sliced mid-table by the generic splitter,
      separating labels from values (confirmed on Q17).

      First attempt: detect "Table N:" blocks and keep them atomic. Bug:
      the line-by-line classifier applied a strict "looks like a table
      row" check starting immediately after the title line -- but real
      table captions are multi-line descriptive PROSE ("Results across
      all processing stages, from voice recording to extracted
      features. The table reports both..."), which fails that check and
      broke the scan after just one line, silently no-op'ing the whole
      fix (confirmed: table detection regex matched fine in isolation,
      but chunk output was byte-identical to the unfixed version).

      Fixed: caption lines are now accepted unconditionally until the
      first genuine data row is seen (density-based check). Only once
      inside the actual data rows does the stricter table-line/prose
      check apply to decide where the table content ends.

    2026-08-01 fix — tokenizer mismatch (provider-aware):
      Token counts were computed with tiktoken's cl100k_base encoding
      (OpenAI's tokenizer), but this project actually runs on two
      different backends depending on environment (see
      app/services/llm/provider.py):
        - local dev: Ollama, serving qwen2.5 (0.5b for routing/grading
          agents, 7b for the answer agent)
        - production: Groq's hosted API, serving llama-3.3-70b-versatile
          (Render free tier has no GPU/RAM for qwen2.5:7b)
      Neither is OpenAI, so cl100k_base was never the right unit for
      either environment.

      Counting now goes through app/utils/tokenization.py, which picks
      Qwen2.5's real (ungated) tokenizer for the Ollama path, and a
      conservative chars-per-token estimate for the Groq/Llama path
      (that tokenizer is gated on HuggingFace -- see that module's
      docstring for the full reasoning). Loaded as a process-wide
      singleton there, NOT per-TextChunker-instance -- this class used
      to load its own AutoTokenizer, which re-hit disk on every single
      document upload since DocumentProcessor (and this chunker) is
      constructed fresh per background upload task.
    """

    def __init__(
        self,
        max_tokens: int = 150,
        overlap_tokens: int = 50,
        max_tokens_table: int = 1000,
        use_groq: Optional[bool] = None,
    ):
        # Mirrors LLMProvider's own selection logic (provider.py) so the
        # chunker's notion of "which backend is this for" never drifts
        # out of sync with which backend is actually serving answers.
        self.use_groq = using_groq(use_groq)

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.max_tokens_table = max_tokens_table

        self.separators = [
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
        ]

        self._table_title_re = re.compile(r'(?m)^Table\s+\d+:')

    def count_tokens(self, text: str) -> int:
        return _shared_count_tokens(text, use_groq=self.use_groq)

    def chunk(self, text: str) -> List[dict]:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text).strip()

        segments = self._extract_table_blocks(text)

        result = []
        for seg_type, seg_text in segments:
            if not seg_text.strip():
                continue
            if seg_type == "table":
                tokens = self.count_tokens(seg_text)
                if tokens <= self.max_tokens_table:
                    result.append({"text": seg_text.strip(), "tokens": tokens})
                else:
                    raw = self._split_recursive(seg_text, self.separators)
                    merged = self._merge_with_overlap(raw)
                    result.extend(
                        {"text": c.strip(), "tokens": self.count_tokens(c)}
                        for c in merged if c.strip()
                    )
            else:
                raw = self._split_recursive(seg_text, self.separators)
                merged = self._merge_with_overlap(raw)
                result.extend(
                    {"text": c.strip(), "tokens": self.count_tokens(c)}
                    for c in merged if c.strip()
                )

        print(f"[CHUNKER DEBUG] segments: {len(segments)}, final chunks: {len(result)}")
        for i, ch in enumerate(result):
            print(f"[CHUNKER DEBUG] chunk {i+1} ({ch['tokens']} tokens): {ch['text'][:80]}")
        return result

    def _extract_table_blocks(self, text: str):
        """
        Splits text into ('prose', text) / ('table', text) segments.
        Each table's scan window is bounded by [this title's start, next
        title's start) so consecutive tables can never merge.

        Within a window: lines are accepted unconditionally (as caption
        text) until the first real data row is seen. After that, the
        stricter _is_table_line check decides where the table ends.
        """
        matches = list(self._table_title_re.finditer(text))
        if not matches:
            return [("prose", text)]

        segments = []
        cursor = 0
        for i, m in enumerate(matches):
            start = m.start()
            if start < cursor:
                continue
            if start > cursor:
                segments.append(("prose", text[cursor:start]))

            window_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            window = text[start:window_end]
            lines = window.split("\n")

            table_lines = []
            consumed_len = 0
            blank_run = 0
            seen_data_row = False

            for line in lines:
                if not line.strip():
                    blank_run += 1
                    if blank_run > 1 and seen_data_row:
                        break
                    table_lines.append(line)
                    consumed_len += len(line) + 1
                    continue
                blank_run = 0

                if not seen_data_row:
                    # Still in title/caption -- accept unconditionally.
                    table_lines.append(line)
                    consumed_len += len(line) + 1
                    if self._is_data_row(line):
                        seen_data_row = True
                    continue

                if self._is_table_line(line):
                    table_lines.append(line)
                    consumed_len += len(line) + 1
                else:
                    break

            table_text = "\n".join(table_lines).strip()
            segments.append(("table", table_text))
            cursor = start + consumed_len

        if cursor < len(text):
            segments.append(("prose", text[cursor:]))

        return segments

    @staticmethod
    def _is_data_row(line: str) -> bool:
        """A line is a genuine data row if it's mostly digits/%/./,/-/etc."""
        stripped = line.strip()
        if not stripped:
            return False
        non_table_chars = re.sub(r'[\d%\.,\-/±()\[\]\s]', '', stripped)
        density = len(non_table_chars) / max(len(stripped), 1)
        return density < 0.35

    @staticmethod
    def _is_table_line(line: str) -> bool:
        """Used once inside data rows: accepts data-heavy OR short label lines."""
        stripped = line.strip()
        if not stripped:
            return False
        non_table_chars = re.sub(r'[\d%\.,\-/±()\[\]\s]', '', stripped)
        density = len(non_table_chars) / max(len(stripped), 1)
        if density < 0.35:
            return True
        if len(stripped) < 45 and not stripped.endswith('.'):
            return True
        return False

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not text.strip():
            return []
        if not separators:
            return self._hard_split_by_tokens(text)

        separator = separators[0]
        remaining = separators[1:]
        splits = text.split(separator)

        result = []
        for idx, raw_split in enumerate(splits):
            piece = raw_split.strip()
            if not piece:
                continue
            trailing = separator if idx < len(splits) - 1 else ""
            if self.count_tokens(piece) > self.max_tokens:
                sub_pieces = self._split_recursive(piece, remaining)
                if sub_pieces and trailing:
                    sub_pieces[-1] = sub_pieces[-1] + trailing
                result.extend(sub_pieces)
            else:
                result.append(piece + trailing)

        return result

    def _hard_split_by_tokens(self, text: str) -> List[str]:
        if self.use_groq:
            # No real tokenizer available on this path (see class
            # docstring / app/utils/tokenization.py) -- fall back to
            # splitting by the same chars-per-token estimate used in
            # count_tokens, so this stays consistent with whatever
            # count_tokens() would report for each resulting piece.
            char_window = self.max_tokens * 4  # matches tokenization._CHARS_PER_TOKEN
            return [
                text[i:i + char_window]
                for i in range(0, len(text), char_window)
            ]
        from app.utils.tokenization import get_qwen_tokenizer
        tokenizer = get_qwen_tokenizer()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return [
            tokenizer.decode(tokens[i:i + self.max_tokens])
            for i in range(0, len(tokens), self.max_tokens)
        ]

    def _merge_with_overlap(self, splits: List[str]) -> List[str]:
        chunks = []
        current = []
        current_tokens = 0
        overlap_buffer = []

        for split in splits:
            split_tokens = self.count_tokens(split)
            if current_tokens + split_tokens > self.max_tokens:
                if current:
                    chunks.append("".join(current))
                    overlap_buffer = self._get_overlap_buffer(current)
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
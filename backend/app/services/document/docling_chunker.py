"""
docling_chunker.py

Structure-aware chunker built on Docling's parsed document, replacing
the regex-based table/section handling in chunker.py for PDFs.

Two problems this fixes (found via eval_rag.py root-cause analysis,
2026-07-09):

1. Table fragmentation: the old chunker split dense numeric tables by
   token count alone, producing bare number streams with no column
   labels (e.g. "39.4" with no indication it's TriviaQA S->S accuracy).
   Docling's TableFormer model gives us real row/column structure, so
   here each table row is serialized as "column_name=value" pairs --
   the number and its label can never be separated by a chunk boundary
   again.

2. Section-boundary bleeding: the old token-based splitter would
   happily stitch the tail of one section to the head of the next
   (confirmed: a chunk mixing "backchannel injection" content with the
   start of "4.3 Implementation Details" diluted the embedding enough
   to drop a directly-relevant chunk out of the top-20 vector search
   results). This chunker treats every section_header as a hard
   boundary -- prose is only merged within a section, never across.
"""

from typing import List, Optional
import tiktoken


class DoclingChunker:
    def __init__(self, model: str = "cl100k_base", max_tokens: int = 150, overlap_tokens: int = 50):
        self.encoding = tiktoken.get_encoding(model)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chunk(self, doc) -> List[dict]:
        """
        doc: a docling Document (result.document from DocumentConverter.convert()).
        Returns list of {"text": str, "tokens": int} -- same shape the
        rest of the ingestion pipeline (embedder, Qdrant storage) already
        expects, so DocumentProcessor needs only a branch at the call site,
        not a rewrite.
        """
        result = []
        current_heading = ""
        prose_buffer: List[str] = []
        prose_tokens = 0

        def flush_prose():
            nonlocal prose_buffer, prose_tokens
            if not prose_buffer:
                return
            text = " ".join(prose_buffer).strip()
            if text:
                # Prepend the section heading as a short context header --
                # anchors the embedding to the right section identity even
                # when the chunk's own content is topically mixed.
                full_text = f"[Section: {current_heading}]\n{text}" if current_heading else text
                result.append({"text": full_text, "tokens": self.count_tokens(full_text)})
            prose_buffer = []
            prose_tokens = 0

        for item, _level in doc.iterate_items():
            label = getattr(item, "label", None)
            label_str = str(label)

            if label_str == "DocItemLabel.SECTION_HEADER" or label_str == "section_header":
                flush_prose()
                current_heading = getattr(item, "text", "") or current_heading
                continue

            if label_str == "DocItemLabel.TABLE" or label_str == "table":
                flush_prose()
                table_chunks = self._chunk_table(item, doc, current_heading)
                result.extend(table_chunks)
                continue

            text = getattr(item, "text", "")
            if not text or not text.strip():
                continue

            text = text.strip()
            t_tokens = self.count_tokens(text)

            if prose_tokens + t_tokens > self.max_tokens and prose_buffer:
                flush_prose()

            prose_buffer.append(text)
            prose_tokens += t_tokens

        flush_prose()
        return result

    def _chunk_table(self, table_item, doc, heading: str) -> List[dict]:
        """
        Serialize each table row as explicit column=value pairs so a
        number can never be retrieved without its column label attached.
        One row (or a small group of rows for very wide/short tables) per
        chunk, with the table caption prepended for context.
        """
        try:
            df = table_item.export_to_dataframe(doc=doc)
        except TypeError:
            # older docling versions: export_to_dataframe() takes no args
            df = table_item.export_to_dataframe()

        caption = ""
        try:
            texts = table_item.captions
            if texts:
                caption = " ".join(getattr(c, "text", "") for c in texts if hasattr(c, "text"))
        except Exception:
            pass

        chunks = []
        columns = [str(c) for c in df.columns]
        row_label_col = columns[0] if columns else None

        for _, row in df.iterrows():
            row_label = str(row[row_label_col]) if row_label_col else ""
            pairs = [f"{col}={row[col]}" for col in columns[1:]] if row_label_col else \
                     [f"{col}={row[col]}" for col in columns]

            header_line = f"[Section: {heading}]" if heading else ""
            caption_line = f"Table: {caption}" if caption else ""
            row_line = f"Row [{row_label}]: " + ", ".join(pairs) if row_label else ", ".join(pairs)

            text = "\n".join(filter(None, [header_line, caption_line, row_line]))
            chunks.append({"text": text, "tokens": self.count_tokens(text)})

        return chunks

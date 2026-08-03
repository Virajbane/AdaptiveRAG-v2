"""
docling_chunker.py

Structure-aware chunker built on Docling's parsed document, replacing
the regex-based table/section handling in chunker.py for PDFs.

Four problems this fixes (found via eval_rag.py root-cause analysis):

1. Table fragmentation (2026-07-09): the old chunker split dense numeric
   tables by token count alone, producing bare number streams with no
   column labels (e.g. "39.4" with no indication it's TriviaQA S->S
   accuracy). Docling's TableFormer model gives us real row/column
   structure, so here each table row is serialized as "column_name=value"
   pairs -- the number and its label can never be separated by a chunk
   boundary again.

2. Section-boundary bleeding (2026-07-09): the old token-based splitter
   would happily stitch the tail of one section to the head of the next
   (confirmed: a chunk mixing "backchannel injection" content with the
   start of "4.3 Implementation Details" diluted the embedding enough
   to drop a directly-relevant chunk out of the top-20 vector search
   results). This chunker treats every section_header as a hard
   boundary -- prose is only merged within a section, never across.

3. Figure/chart content silently dropped (2026-07-14, Bug 3): PICTURE
   items had no branch here at all -- they fell through to the generic
   `text = getattr(item, "text", "")` path, which is empty for an image,
   so they hit the `if not text: continue` guard and were dropped
   entirely. Confirmed via diagnostic: a UTMOS chart value never
   appeared in ANY retrieved candidate, in any form -- not a ranking
   problem, the number was never captured into a chunk at all. Fixed by
   adding a PICTURE branch below that pulls the VLM-generated
   description (see docling_parser.py's do_picture_description /
   PictureDescriptionApiOptions config) and chunks it the same way a
   table row is chunked -- anchored to its section heading so it's
   retrievable the same way.

4. Duplicate column name collision (2026-08-03): wide comparison tables
   (e.g. Table 1: LlamaQ S->T/S->S, TriviaQA S->T/S->S, ..., Avg S->T/
   S->S) repeat the same leaf column name once per benchmark group. If
   Docling's export_to_dataframe() doesn't preserve that grouping as a
   MultiIndex, df.columns ends up with real duplicates (e.g. "S->T"
   three times over). The old code indexed each row by NAME
   (row[col]), and pandas returns EVERY same-named column as a Series
   when the name is duplicated -- not a single scalar. So "Lat.=826"
   (Lychee-FD, FullDuplexBench 1.5) silently became a garbled
   multi-value block whenever "Lat." also existed for FullDuplexBench
   1.0, and a downstream reader (verification pass or the LLM itself)
   could pick the wrong line out of it. Confirmed as the source of the
   Q16 Lat./Stop. swap and the Q11-14 "couldn't verify" failures on
   Table 1. Fixed below by indexing values POSITIONALLY instead of by
   name, so duplicate names can never collide.
"""

from typing import List, Optional
from app.utils.tokenization import count_tokens as _shared_count_tokens, using_groq


class DoclingChunker:
    """
    2026-08-01 fix — tokenizer mismatch: see chunker.py's matching
    docstring and app/utils/tokenization.py for the full reasoning.
    Counting is routed through the shared, process-wide tokenizer
    there instead of loading a private AutoTokenizer per instance --
    DoclingChunker (like TextChunker) is constructed fresh per
    background upload task, so a per-instance load would re-hit disk
    on every single PDF upload.
    """

    def __init__(
        self,
        max_tokens: int = 150,
        overlap_tokens: int = 50,
        use_groq: Optional[bool] = None,
    ):
        self.use_groq = using_groq(use_groq)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def count_tokens(self, text: str) -> int:
        return _shared_count_tokens(text, use_groq=self.use_groq)

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

            if label_str == "DocItemLabel.PICTURE" or label_str == "picture":
                flush_prose()
                picture_chunk = self._chunk_picture(item, current_heading)
                if picture_chunk:
                    result.append(picture_chunk)
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

    def _chunk_picture(self, picture_item, heading: str) -> Optional[dict]:
        """
        Turns a figure's VLM-generated description (produced during
        parsing -- see docling_parser.py's do_picture_description config)
        into a real, retrievable chunk. Without this, PICTURE items have
        no text at all and were previously silently dropped.

        ASSUMPTION TO VERIFY: picture_item.annotations is assumed to be a
        list of annotation objects, where a PictureDescriptionData-style
        annotation exposes its generated text via a `.text` attribute.
        Exact attribute name can differ by docling version -- if this
        comes back empty even though do_picture_description is enabled
        and Ollama logs show the vision model actually being called,
        paste the picture_item's actual attributes/annotations structure
        and this gets corrected against your real installed version.

        Returns None (produces no chunk) if do_picture_description was
        disabled or the description came back empty -- same "don't
        silently fabricate content" principle as the rest of the
        pipeline; a figure with no description is better logged/skipped
        than chunked as an empty string.
        """
        description_text = ""
        annotations = getattr(picture_item, "annotations", None) or []
        for annotation in annotations:
            candidate = getattr(annotation, "text", None)
            if candidate:
                description_text = candidate.strip()
                break

        if not description_text:
            return None

        caption = ""
        try:
            texts = getattr(picture_item, "captions", None) or []
            caption = " ".join(getattr(c, "text", "") for c in texts if hasattr(c, "text"))
        except Exception:
            pass

        header_line = f"[Section: {heading}]" if heading else ""
        caption_line = f"Figure: {caption}" if caption else "Figure:"
        text = "\n".join(filter(None, [header_line, caption_line, description_text]))

        return {"text": text, "tokens": self.count_tokens(text)}

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

        # 2026-08-03 fix — duplicate column name collision (see module
        # docstring, item 4). Wide comparison tables repeat the same leaf
        # column name once per benchmark group (e.g. "S->T"/"S->S" under
        # LlamaQ, TriviaQA, and Avg; or "Lat."/"Stop." under
        # FullDuplexBench 1.0 and 1.5). If Docling's dataframe export
        # doesn't preserve that grouping as a MultiIndex, df.columns ends
        # up with real duplicate names. Indexing a row by NAME (row[col])
        # then returns EVERY same-named column as a pandas Series instead
        # of one scalar -- silently corrupting the row's serialized text
        # (e.g. "Lat.=826" became a garbled multi-value block whenever
        # more than one "Lat." column existed).
        #
        # Fix: read every value POSITIONALLY (list(row)) so duplicate
        # names can never collide -- value at index i always corresponds
        # to columns[i], regardless of how many other columns share that
        # name. Disambiguate the display label for any repeat so the
        # chunk text stays legible: "S->T", then "S->T (2)", "S->T (3)"
        # for later occurrences, rather than looking like a silent
        # overwrite.
        seen: dict = {}
        disambiguated = []
        for c in columns:
            seen[c] = seen.get(c, 0) + 1
            disambiguated.append(c if seen[c] == 1 else f"{c} ({seen[c]})")

        row_label_idx = 0 if disambiguated else None

        for _, row in df.iterrows():
            values = list(row)  # positional -- immune to duplicate names
            row_label = str(values[row_label_idx]) if row_label_idx is not None else ""
            value_cols = disambiguated[1:] if row_label_idx is not None else disambiguated
            value_vals = values[1:] if row_label_idx is not None else values

            pairs = [f"{col}={val}" for col, val in zip(value_cols, value_vals)]

            header_line = f"[Section: {heading}]" if heading else ""
            caption_line = f"Table: {caption}" if caption else ""
            row_line = f"Row [{row_label}]: " + ", ".join(pairs) if row_label else ", ".join(pairs)

            text = "\n".join(filter(None, [header_line, caption_line, row_line]))
            chunks.append({"text": text, "tokens": self.count_tokens(text)})

        return chunks
"""
docling_parser.py
...
"""

import logging

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)

from app.services.document.pdf_batcher import split_pdf_into_batches, cleanup_batches
from app.config.settings import settings

logger = logging.getLogger(__name__)


# 2026-07-14 fix (Bug 3, figure-value extraction): the converter previously
# had NO PdfPipelineOptions configured at all, so do_picture_description
# defaulted to off -- every figure (UTMOS chart, gradient-influence
# heatmap, layer-ablation curve, etc.) was parsed as a bare PICTURE item
# with no text, which DoclingChunker then silently dropped (no branch for
# that label). Confirmed via eval_rag.py diagnostic: the Step-Audio-2
# UTMOS value (4.44) never appeared in ANY retrieved candidate, in any
# form -- not a ranking problem, the number was never captured at all.
#
# Fix: run every extracted picture through a local vision model (Ollama,
# same endpoint already used for embeddings/generation) with a prompt
# specifically asking for a literal label:value transcription of any
# chart/graph data -- not a generic "describe this image" caption, which
# would produce prose that doesn't actually preserve exact numbers.
#
# ASSUMPTION TO VERIFY: PictureDescriptionApiOptions's exact field names
# (url / params / prompt / timeout) and the request shape it POSTs
# (assumed here to be an OpenAI-compatible vision chat payload, which is
# why this targets Ollama's /v1/chat/completions endpoint rather than its
# native /api/generate) can differ by installed docling version. If
# ingestion raises a TypeError/AttributeError on these fields, paste the
# traceback and this gets corrected against your actual installed version
# rather than guessed twice.
_PICTURE_DESCRIPTION_PROMPT = (
    "This image is a figure from a research paper (bar chart, line chart, "
    "heatmap, or diagram). Transcribe every visible data label and its "
    "exact numeric value as a plain list, one per line, in the form "
    "'label: value'. Include axis labels and legend entries if they "
    "identify which value belongs to which series. Do not describe the "
    "visual style -- only transcribe the labels and numbers exactly as "
    "shown."
)


def _build_pipeline_options() -> PdfPipelineOptions:
    options = PdfPipelineOptions()

    if settings.ENABLE_PICTURE_DESCRIPTION:
        # generate_picture_images=True: the picture-description model needs
        # an actual rendered image to look at, not just the page's parsed
        # layout structure -- this is what makes that image available to it.
        options.generate_picture_images = True
        options.do_picture_description = True
        options.picture_description_options = PictureDescriptionApiOptions(
            url=f"{settings.OLLAMA_BASE_URL}/v1/chat/completions",
            params={"model": settings.PICTURE_DESCRIPTION_MODEL},
            prompt=_PICTURE_DESCRIPTION_PROMPT,
            timeout=120,  # CPU-only local inference on an image can be slow;
                          # generous timeout rather than a false failure.
        )

    return options


class DoclingPDFParser:
    _converter = None  # lazy singleton -- avoid re-loading layout/table models per call

    @classmethod
    def _get_converter(cls) -> DocumentConverter:
        if cls._converter is None:
            pipeline_options = _build_pipeline_options()
            cls._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        return cls._converter

    @staticmethod
    def parse(file_path: str) -> dict:
        """
        Returns:
            {
                "document": DoclingDocument,     # pass THIS to DoclingChunker.chunk()
                "status": ConversionStatus,      # SUCCESS, PARTIAL_SUCCESS
                "page_errors": [str, ...],       # error messages; empty list if none
            }

        BREAKING CHANGE from previous version: this used to return the bare
        DoclingDocument. Callers must now do parse(...)["document"] to get
        what they previously got directly.

        Why: converter.convert() defaulted to raises_on_error=True, and this
        function only ever returned result.document -- so result.status and
        result.errors were discarded even on a PARTIAL_SUCCESS conversion
        (e.g. one page hit a bad_alloc/OCR failure but the rest parsed fine).
        Those pages' content never becomes chunks, so it never shows up in
        chunk-level failed_chunk_indices either -- it's a silent gap
        upstream of chunking entirely. This surfaces it instead.
        """
        converter = DoclingPDFParser._get_converter()
        # raises_on_error=False: let FAILURE be reported as data (status +
        # errors) rather than an exception, consistent with how
        # embed_batch/store_vectors report failure as data rather than
        # raising -- we decide what to do with it below, not in convert().
        result = converter.convert(file_path, raises_on_error=False)
        if not hasattr(result, "document"):
            result = next(iter(result))

        page_errors = [e.error_message for e in result.errors]

        if result.status == ConversionStatus.FAILURE:
            # Total failure -- no usable document at all. This is the one
            # case that should still raise: there's nothing to chunk, no
            # partial content to preserve, same logic as embed_batch's
            # "if not successful_chunks: raise" case.
            raise ValueError(
                f"Docling failed to convert {file_path}: "
                f"{'; '.join(page_errors) if page_errors else 'no error detail available'}"
            )

        return {
            "document": result.document,
            "status": result.status,
            "page_errors": page_errors,
        }

    @staticmethod
    def _retry_batch_at_page_level(batch, converter) -> dict:
        """
        2026-07-14 fix (Bug 1 -- silent page loss on pages 19-22): when a
        multi-page batch fails outright (std::bad_alloc or FAILURE
        status), the OLD behavior lost every page in that batch, even if
        only one page was actually the memory-hog. E.g. a batch_size=5
        batch covering pages 18-22 would lose all 5 pages even though the
        original crash logs showed the failure was specific to individual
        pages (19, 20, 21, 22 each logged their own bad_alloc separately).

        This re-splits the FAILED BATCH's own mini-PDF (batch.file_path,
        already just a handful of pages) down to single-page mini-PDFs
        and retries each page alone -- isolating exactly which page(s)
        are unparseable instead of writing off the whole batch.

        ASSUMPTION TO VERIFY: split_pdf_into_batches(batch.file_path, ...)
        is assumed to return start_page/end_page 0-indexed RELATIVE TO
        THE FILE PASSED IN (i.e. relative to this already-small mini-PDF),
        the same way the top-level call in parse_in_batches works
        relative to the original file. That's why `batch.start_page +
        sub.start_page` is used below to recover the page's ABSOLUTE
        position in the original document. If pdf_batcher.py instead
        always returns page numbers absolute to the original file
        regardless of what it's splitting, drop that offset.

        Returns:
            {
                "documents": [{"document": DoclingDocument,
                                "start_page": int, "end_page": int}, ...],
                "still_failed_pages": [int, ...],  # 0-indexed, ABSOLUTE
                                                     # (original document)
            }
        """
        sub_batches = split_pdf_into_batches(batch.file_path, batch_size=1)
        succeeded: list[dict] = []
        still_failed_pages: list[int] = []

        try:
            for sub in sub_batches:
                absolute_page = batch.start_page + sub.start_page
                try:
                    result = converter.convert(sub.file_path, raises_on_error=False)
                    if not hasattr(result, "document"):
                        result = next(iter(result))

                    if result.status == ConversionStatus.FAILURE:
                        still_failed_pages.append(absolute_page)
                        continue

                    succeeded.append(
                        {
                            "document": result.document,
                            "start_page": absolute_page,
                            "end_page": absolute_page + 1,
                        }
                    )

                except Exception as e:
                    # Same page failed even in complete isolation -- a
                    # single-page PDF is about as small as we can make
                    # the input, so this is treated as a real, page-level
                    # failure rather than retried further.
                    logger.error(
                        "Page %d still failed at single-page isolation: %s",
                        absolute_page + 1, e,
                    )
                    still_failed_pages.append(absolute_page)
                    continue
        finally:
            cleanup_batches(sub_batches)

        return {"documents": succeeded, "still_failed_pages": still_failed_pages}

    @staticmethod
    def _fallback_extract_page_text(file_path: str, page_number: int) -> str | None:
        """
        2026-07-14 fix (Bug 1, fix option #4 from the eval report): last
        resort plain-text extraction for a page Docling could not process
        even in single-page isolation. Trades structure-aware parsing
        (tables, reading order, headings, figure captions) for just
        getting SOME text instead of losing the page's content entirely
        -- e.g. a page that's genuinely too memory-heavy for Docling's
        layout model to lay out, but whose raw text PyMuPDF can still
        pull with negligible memory.

        page_number is 0-indexed, ABSOLUTE (position in the original
        document), matching still_failed_pages above.

        Returns None (not an empty string) if PyMuPDF itself isn't
        installed, the page is out of range, or extraction produced no
        usable text -- callers should treat None as "still genuinely
        lost" and surface that, not silently drop it.
        """
        try:
            import fitz  # PyMuPDF -- confirm this is in requirements
        except ImportError:
            logger.error(
                "PyMuPDF (fitz) not installed -- cannot run fallback text "
                "extraction for page %d. Add `pymupdf` to requirements.",
                page_number + 1,
            )
            return None

        try:
            doc = fitz.open(file_path)
            if page_number < 0 or page_number >= len(doc):
                doc.close()
                return None
            text = doc[page_number].get_text().strip()
            doc.close()
            return text or None
        except Exception as e:
            logger.error(
                "PyMuPDF fallback extraction failed for page %d: %s",
                page_number + 1, e,
            )
            return None

    @staticmethod
    def parse_in_batches(file_path: str, batch_size: int = 5) -> dict:
        """
        Splits `file_path` into page-range mini-PDFs (see pdf_batcher.py) and
        runs DocumentConverter.convert() on EACH ONE SEPARATELY, so a
        bad_alloc on one batch only costs that batch's pages, not the whole
        document -- same "isolate the failure, don't let it take down
        everything else" pattern as embed_batch's per-chunk retry.

        2026-07-14 fix (Bug 1): a batch-level failure no longer means
        every page in that batch is lost. It now goes through two more
        recovery stages before giving up on a page:
          1. Retry the failed batch at single-page granularity
             (_retry_batch_at_page_level) -- isolates whether the WHOLE
             batch was unparseable or just one bad page within it.
          2. For any page still failing alone, fall back to PyMuPDF plain
             text extraction (_fallback_extract_page_text) -- degraded
             but present, instead of silently absent.
        A page only ends up in `pages_fully_lost` if BOTH of those fail.

        Design note -- deliberately does NOT merge DoclingDocument objects.
        Docling has no supported API for splicing multiple DoclingDocuments
        into one (their internal texts/tables/pictures/groups/page-ref
        structure isn't designed for it), and hand-rolling that merge is a
        good way to silently corrupt cross-references later. Instead this
        returns a LIST of (document, start_page) pairs -- chunk each one
        separately with your existing DoclingChunker.chunk() call, then
        concatenate the resulting chunk lists. Merging AFTER chunking is
        safe because chunks are just text + metadata by that point.

        Returns:
            {
                "documents": [
                    {"document": DoclingDocument, "start_page": int, "end_page": int},
                    ...
                ],  # only batches/pages that succeeded via Docling; empty
                    # list if EVERY page failed (see raise below)
                "status": ConversionStatus,  # SUCCESS if no page_errors at all,
                                              # PARTIAL_SUCCESS if some but not all
                                              # batches/pages had issues
                "page_errors": [str, ...],   # every error/recovery message,
                                              # prefixed with the page(s) it
                                              # came from
                "fallback_pages": {int: str},  # 0-indexed absolute page ->
                                                # plain text, for pages Docling
                                                # couldn't parse even alone but
                                                # PyMuPDF recovered
                "pages_fully_lost": [int, ...],  # 0-indexed absolute pages
                                                   # where NEITHER Docling
                                                   # (even isolated) NOR the
                                                   # PyMuPDF fallback produced
                                                   # anything -- genuinely
                                                   # missing content
            }

        Raises:
            ValueError if every single batch failed (no usable content at all) --
            same "nothing to chunk" case parse() raises on.
        """
        converter = DoclingPDFParser._get_converter()
        batches = split_pdf_into_batches(file_path, batch_size=batch_size)

        succeeded: list[dict] = []
        page_errors: list[str] = []
        fallback_pages: dict = {}
        pages_fully_lost: list[int] = []

        def _recover_failed_batch(batch) -> None:
            """Runs the retry-then-fallback pipeline for one failed batch
            and folds the results into the outer succeeded/page_errors/
            fallback_pages/pages_fully_lost accumulators."""
            is_single_page = (batch.end_page - batch.start_page) <= 1

            if is_single_page:
                # Already as small as it gets -- skip the redundant
                # re-split-and-retry step and go straight to recording
                # it as still-failed at page level.
                still_failed_pages = [batch.start_page]
            else:
                recovered = DoclingPDFParser._retry_batch_at_page_level(batch, converter)
                succeeded.extend(recovered["documents"])
                still_failed_pages = recovered["still_failed_pages"]
                if recovered["documents"]:
                    recovered_page_nums = [d["start_page"] + 1 for d in recovered["documents"]]
                    page_errors.append(
                        f"[pages {batch.start_page + 1}-{batch.end_page}] batch failed, "
                        f"but recovered {len(recovered['documents'])} page(s) individually "
                        f"({recovered_page_nums})"
                    )

            for p in still_failed_pages:
                fallback_text = DoclingPDFParser._fallback_extract_page_text(file_path, p)
                if fallback_text:
                    fallback_pages[p] = fallback_text
                    page_errors.append(
                        f"[page {p + 1}] Docling failed even in single-page isolation; "
                        f"recovered via PyMuPDF plain-text fallback (structure lost, "
                        f"text preserved)"
                    )
                else:
                    pages_fully_lost.append(p)
                    page_errors.append(
                        f"[page {p + 1}] Docling failed even in isolation, and PyMuPDF "
                        f"fallback produced no usable text -- CONTENT LOST for this page"
                    )

        try:
            for batch in batches:
                try:
                    result = converter.convert(batch.file_path, raises_on_error=False)
                    if not hasattr(result, "document"):
                        result = next(iter(result))

                    # Offset in-batch error messages back to original-document
                    # page numbers (batch.start_page is 0-indexed into the
                    # ORIGINAL file, e.g. batch 3 of size 5 starts at page 10).
                    batch_errors = [
                        f"[pages {batch.start_page + 1}-{batch.end_page}] {e.error_message}"
                        for e in result.errors
                    ]

                    if result.status == ConversionStatus.FAILURE:
                        # 2026-07-14: previously just recorded this range as
                        # lost and moved on. Now attempt page-level recovery
                        # before writing the whole batch off.
                        logger.warning(
                            "Batch pages %d-%d failed outright -- retrying at "
                            "single-page level before giving up on the range.",
                            batch.start_page + 1, batch.end_page,
                        )
                        if not batch_errors:
                            page_errors.append(
                                f"[pages {batch.start_page + 1}-{batch.end_page}] "
                                f"Docling conversion failed, no error detail available"
                            )
                        else:
                            page_errors.extend(batch_errors)
                        _recover_failed_batch(batch)
                        continue

                    page_errors.extend(batch_errors)
                    succeeded.append(
                        {
                            "document": result.document,
                            "start_page": batch.start_page,
                            "end_page": batch.end_page,
                        }
                    )

                except Exception as e:
                    # Catches things convert() itself raises despite
                    # raises_on_error=False -- e.g. a hard std::bad_alloc
                    # crash/segfault surfaced as a Python exception rather
                    # than a clean FAILURE status. This is the whole reason
                    # batching exists: isolate it to this batch's page range.
                    #
                    # 2026-07-14: this is exactly the original bug's failure
                    # mode (std::bad_alloc on pages 19-22). Previously the
                    # whole batch was written off here; now it goes through
                    # the same page-level retry + fallback recovery path as
                    # a clean FAILURE status above.
                    logger.error(
                        "Batch pages %d-%d raised during convert(): %s -- "
                        "retrying at single-page level.",
                        batch.start_page + 1, batch.end_page, e,
                    )
                    page_errors.append(
                        f"[pages {batch.start_page + 1}-{batch.end_page}] "
                        f"Docling raised: {e}"
                    )
                    _recover_failed_batch(batch)
                    continue
        finally:
            # Always clean up temp mini-PDFs, even if something above raised
            # or the loop was interrupted partway through.
            cleanup_batches(batches)

        if not succeeded and not fallback_pages:
            raise ValueError(
                f"Docling failed to convert every batch of {file_path}, and "
                f"the PyMuPDF fallback recovered nothing either: "
                f"{'; '.join(page_errors) if page_errors else 'no error detail available'}"
            )

        status = (
            ConversionStatus.SUCCESS if not page_errors else ConversionStatus.PARTIAL_SUCCESS
        )

        return {
            "documents": succeeded,
            "status": status,
            "page_errors": page_errors,
            "fallback_pages": fallback_pages,
            "pages_fully_lost": pages_fully_lost,
        }

    @staticmethod
    def to_plain_text(doc, max_chars: int = 2500) -> str:
        """
        Unchanged -- still takes the raw DoclingDocument, not the new dict.
        Caller must unwrap parse()'s return before passing here.
        """
        return doc.export_to_markdown()[:max_chars]
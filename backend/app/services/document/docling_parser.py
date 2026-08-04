"""
docling_parser.py

2026-08-03 fix — picture images rendered too small to read (see
_build_pipeline_options below): confirmed via debug_vision_model_direct.py
that a real figure (the UTMOS bar chart on page 8) was being rendered as a
67x19 pixel PNG before ever reaching the vision model. At that resolution
no vision model can read axis labels or bar values -- this explains BOTH
failure modes seen in production: some figures got no annotation at all
(model found nothing legible), and others produced plausible-looking but
meaningless numeric output (model guessing at a blurry image instead of
transcribing real data). This was never a prompt or model-choice problem;
Docling's default rendering scale for extracted pictures is close to
screen resolution, which is fine for a thumbnail preview but not for OCR/
VLM transcription of dense chart data. Fixed by setting images_scale
explicitly (see below) so pictures render large enough to actually read.

2026-08-05 fix — repeated per-page logo/watermark images causing massive
parse-time bloat. Confirmed against a real production PDF (22 pages): a
single small header logo image, byte-identical across all 22 pages, was
being detected as a distinct Picture item on every page and sent through
the full do_picture_description VLM round-trip each time -- N wasted
calls per document (one per page) before a single real figure was even
considered. Log evidence from that run: batches with zero real figures
processed at ~1.2-1.5 sec/page (pure layout/OCR, no VLM call), while
batches hitting the logo ran at 37-190+ sec/page, with several individual
calls hitting the full timeout ceiling and returning nothing.

This is NOT specific to that one document -- any PDF template that
stamps a repeating header/footer/watermark image (letterhead logos,
"CONFIDENTIAL" watermarks, conference/journal branding, page-corner
icons) hits the same failure mode, so the fix below is intentionally
generic: it hashes whatever images actually appear in whatever PDF is
being parsed and only touches images that repeat across a large fraction
of THAT document's own page count. It makes no assumption about what a
logo looks like, where it sits, or how many distinct repeating images
exist in a given file (a document with both a header logo AND a
different footer watermark is handled the same way, automatically, since
detection is per-hash not "find one thing"). Documents with no repeated
images at all pass through completely unchanged. See
_strip_repeated_page_images below and its own docstring for the exact
matching logic and configurable thresholds (now sourced from settings,
not hardcoded, so this can be tuned per corpus without a code change).
Also raised the API timeout ceiling further (see
_PICTURE_DESCRIPTION_TIMEOUT below) since several calls in the same
production run still hit 300s even with keep_alive already in place --
that headroom is now affordable because logo-stripping removes the
majority of call volume on any document that has this pattern.
"""

import hashlib
import logging
import os
from collections import defaultdict

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

# 2026-08-03 fix -- see module docstring. Docling's own default for
# extracted-picture resolution is tuned for thumbnail-sized previews,
# not for OCR/VLM transcription of dense chart data. Confirmed directly:
# a real bar chart (Figure 4, page 8) rendered at only 67x19 pixels under
# the default, which is well below what any vision model can read text
# or bar heights from. images_scale is a multiplier on top of Docling's
# base rendering size (roughly 72 DPI-equivalent at scale=1.0); 4.0 gets
# dense figures into a legible range at the cost of somewhat slower
# parsing and slightly larger memory use per page -- worth it since the
# alternative was silently unusable images. If figures are still
# illegible after this change, try 6.0 next; there's no universal "right"
# value, since it depends on how small the source figure is on the page.
_PICTURE_IMAGES_SCALE = 4.0

# 2026-08-05 fix -- raised from 300s. Production log showed three
# individual picture-description calls still hitting the full 300s
# ceiling and returning nothing, even with keep_alive="30m" already
# active (see below) -- keep_alive prevents *cold-load* timeouts, but
# doesn't guarantee every inference finishes in under 300s on this
# hardware (CPU-bound Ollama, small VRAM budget). 480s gives real
# headroom for a genuinely dense figure. This is affordable now that
# logo-stripping (see _strip_repeated_page_images) removes the bulk of
# call volume -- previously every one of these slow calls was competing
# with ~1 logo call per page for the same 300s budget.
#
# FOLLOW-UP NOT YET DONE: shrinking the image actually sent to the VLM
# (separate from images_scale, which must stay high for legibility in
# the exported document) would attack *why* calls are slow, not just
# how long we wait -- fewer vision-tower tokens on a CPU box should cut
# per-call latency substantially. Not implemented here because it
# requires confirming whether PictureDescriptionApiOptions exposes any
# resize hook in your installed docling version, or whether that would
# require disabling do_picture_description and posting to Ollama
# manually with a resized image. Grep your installed docling version
# (`python -c "import docling; print(docling.__version__)"` and inspect
# PictureDescriptionApiOptions's fields) and paste that back for this to
# be implemented correctly rather than guessed.
_PICTURE_DESCRIPTION_TIMEOUT = 480

# 2026-08-05 fix -- see module docstring. Thresholds for treating a
# repeated embedded image as a logo/watermark rather than real content.
# These are read from settings (falling back to the defaults below) so
# ops can tune aggressiveness per-corpus without a code change --
# min_page_fraction is deliberately conservative by default (0.5) because
# real recurring diagrams essentially never repeat byte-for-byte across
# half a document, since even a legitimately reused figure would differ
# slightly from re-scaling/re-compression per render. min_page_floor
# guards short documents where the fraction alone would be too
# aggressive (e.g. a 4-page doc with the same image on 2 pages could
# plausibly be intentional, not a logo).
_LOGO_STRIP_MIN_PAGE_FRACTION = getattr(settings, "LOGO_STRIP_MIN_PAGE_FRACTION", 0.5)
_LOGO_STRIP_MIN_PAGE_FLOOR = getattr(settings, "LOGO_STRIP_MIN_PAGE_FLOOR", 3)
# Master on/off switch -- default True, but lets ops disable this whole
# pre-pass instantly (e.g. if it ever misfires on a specific corpus)
# without touching code, since it's a heuristic layered on top of
# Docling's own parsing, not something Docling itself understands.
_LOGO_STRIP_ENABLED = getattr(settings, "LOGO_STRIP_ENABLED", True)


def _build_pipeline_options() -> PdfPipelineOptions:
    options = PdfPipelineOptions()

    # 2026-08-03 fix: must be set regardless of do_picture_description,
    # since it controls the resolution of the rendered image that gets
    # produced for any picture item, not just what's sent to the VLM.
    options.images_scale = _PICTURE_IMAGES_SCALE

    if settings.ENABLE_PICTURE_DESCRIPTION:
        # 2026-07-15 fix: Docling has its OWN separate safety gate for any
        # remote call a pipeline step might make -- setting
        # do_picture_description=True alone is not enough. Confirmed
        # directly: every single page failed conversion with "Connections
        # to remote services is only allowed when set explicitly.
        # pipeline_options.enable_remote_services=True.", which meant the
        # WHOLE document (not just figures) fell back to plain PyMuPDF
        # text extraction -- losing table structure and section
        # boundaries too, not just figure descriptions. This flag must be
        # set explicitly even though moondream is running locally --
        # Docling doesn't distinguish "local" from "remote" here, it just
        # sees an HTTP call leaving the process.
        options.enable_remote_services = True

        # generate_picture_images=True: the picture-description model needs
        # an actual rendered image to look at, not just the page's parsed
        # layout structure -- this is what makes that image available to it.
        # Combined with images_scale above, this is now a legible image
        # rather than a thumbnail-sized one.
        options.generate_picture_images = True
        options.do_picture_description = True
        options.picture_description_options = PictureDescriptionApiOptions(
            url=f"{settings.OLLAMA_BASE_URL}/v1/chat/completions",
            # 2026-08-04 fix: "keep_alive" stops Ollama unloading the vision
            # model from VRAM between calls. Without it, every picture-
            # description request was a cold load on this 2GB-VRAM GPU --
            # confirmed in production: three separate 120s timeouts inside
            # ONE 5-page batch, each ~2 minutes apart, which is the signature
            # of the model being unloaded after every single call and never
            # finishing reload before timeout. This silently dropped the
            # picture's caption with no retry -- conversion "succeeded" but
            # figures came through captionless.
            params={
                "model": settings.PICTURE_DESCRIPTION_MODEL,
                "keep_alive": "30m",
            },
            prompt=_PICTURE_DESCRIPTION_PROMPT,
            # 2026-08-05 fix: raised from 120s -> 300s -> 480s. See
            # _PICTURE_DESCRIPTION_TIMEOUT comment above for why 300s was
            # still insufficient in production.
            timeout=_PICTURE_DESCRIPTION_TIMEOUT,
        )

    return options


# 2026-08-05 fix -- see module docstring. Generic pre-pass with PyMuPDF,
# run before Docling ever opens the file, for ANY PDF -- not tuned to any
# one document. Hashes every embedded raster image's raw bytes per page;
# any hash that repeats across a large fraction of THAT document's own
# pages is treated as a logo/watermark and white-redacted in a scratch
# copy. Docling's layout model then has nothing to cluster as a Picture
# in that region, so it's never even a candidate for description.
#
# Handles multiple distinct repeating images automatically (e.g. a
# document with both a header logo and a separately-imaged footer
# watermark) -- detection works per-hash, not "find the one logo", so
# there's no assumption about how many repeating images exist or that
# there's exactly one. A document with no repeated images at all (the
# common case) returns unchanged with zero PyMuPDF write cost.
#
# KNOWN LIMITATION (by design, not yet handled): this only catches
# byte-identical raster images via exact MD5. It will NOT catch:
#   - vector-drawn logos (paths/shapes, no embedded image object)
#   - a logo that's re-compressed slightly differently per page (some
#     PDF generators re-encode each page's resources independently even
#     when the source image is "the same")
# If a future document has a repeating logo that survives this pass
# untouched, that's the signature to look for -- confirm via
# `page.get_image_info(xrefs=True)` whether the logo even appears as a
# raster image object at all before assuming this function is broken.
#
# SAFETY: this entire pre-pass is a heuristic layered on top of Docling,
# not a Docling feature itself -- it must never be the reason a valid PDF
# fails to ingest. Any unexpected error here is caught, logged, and
# treated as "no logo found" (i.e. falls back to the original file)
# rather than propagating and blocking the parse.
#
# ASSUMPTION TO VERIFY: relies on Page.get_image_info(xrefs=True) and
# Document.extract_image(xref), plus add_redact_annot/apply_redactions
# for the blanking step -- all present in the PyMuPDF version installed
# in this container as of this fix. If your production environment pins
# an older PyMuPDF and this starts silently no-op'ing (check logs for
# the warning below), paste `import pymupdf; pymupdf.__version__` (or
# `fitz.__doc__`) and this gets corrected against your actual installed
# version.
def _strip_repeated_page_images(
    file_path: str,
    min_page_fraction: float = _LOGO_STRIP_MIN_PAGE_FRACTION,
    min_page_floor: int = _LOGO_STRIP_MIN_PAGE_FLOOR,
) -> str:
    """
    Returns a path to hand to Docling's converter. If no repeated images
    were found (or the document has too few pages for "repeats across
    pages" to mean anything, or logo-stripping is disabled via settings,
    or anything unexpected goes wrong), returns file_path UNCHANGED and
    no scratch file is created -- this function is designed to fail open.
    If a scratch copy WAS made, the caller is responsible for deleting it
    after conversion (it is NOT auto-cleaned here, matching how batch
    scratch files are already handled by cleanup_batches elsewhere in
    this module).
    """
    if not _LOGO_STRIP_ENABLED:
        return file_path

    try:
        return _strip_repeated_page_images_impl(file_path, min_page_fraction, min_page_floor)
    except Exception as e:
        # Fail open: a bug or an unexpected PDF structure in this
        # heuristic pre-pass must never block ingestion of an otherwise
        # valid document. Worst case without this fix: some documents
        # are slower to parse (the original problem). Worst case WITH a
        # silent crash here instead of this guard: documents that
        # ingested fine yesterday stop ingesting at all today. The
        # former is clearly the safer failure mode.
        logger.warning(
            "Logo-strip pre-pass raised an unexpected error on %s (%s) "
            "-- falling back to the original file unmodified. This is "
            "non-fatal; conversion will proceed, just without the "
            "logo-stripping speedup for this document.",
            file_path, e,
        )
        return file_path


def _strip_repeated_page_images_impl(
    file_path: str,
    min_page_fraction: float,
    min_page_floor: int,
) -> str:
    import fitz  # PyMuPDF -- already a dependency, used elsewhere in this file

    doc = fitz.open(file_path)
    try:
        n_pages = len(doc)
        if n_pages < min_page_floor:
            return file_path  # too few pages for "repeats across pages" to mean anything

        hash_pages = defaultdict(set)
        occurrences = []  # (page_num, hash, bbox)

        for page_num in range(n_pages):
            page = doc[page_num]
            try:
                image_infos = page.get_image_info(xrefs=True)
            except Exception as e:
                logger.warning(
                    "Logo-strip pre-pass: get_image_info failed on page %d "
                    "of %s (%s) -- skipping logo detection for this page only.",
                    page_num + 1, file_path, e,
                )
                continue

            for img in image_infos:
                xref = img.get("xref")
                if not xref:
                    continue
                try:
                    raw = doc.extract_image(xref)["image"]
                except Exception as e:
                    logger.warning(
                        "Logo-strip pre-pass: extract_image failed for xref "
                        "%s on page %d of %s (%s) -- skipping this image.",
                        xref, page_num + 1, file_path, e,
                    )
                    continue
                h = hashlib.md5(raw).hexdigest()
                hash_pages[h].add(page_num)
                occurrences.append((page_num, h, img["bbox"]))

        threshold = max(min_page_floor, int(n_pages * min_page_fraction))
        # NOTE: this is a set comprehension, not a single "the logo hash"
        # variable -- any number of distinct repeating images (header
        # logo, footer watermark, etc., each a different hash) that each
        # individually clear the threshold get included and stripped in
        # the same pass. No assumption that there is exactly one.
        logo_hashes = {h for h, pages in hash_pages.items() if len(pages) >= threshold}

        if not logo_hashes:
            return file_path

        logger.info(
            "Logo-strip pre-pass on %s: detected %d distinct repeated "
            "image(s) across >=%d/%d pages -- treating as logo/"
            "watermark, blanking before Docling conversion (this "
            "replaces one picture-description VLM call per occurrence "
            "with zero).",
            file_path, len(logo_hashes), threshold, n_pages,
        )

        redacted_regions = 0
        for page_num, h, bbox in occurrences:
            if h in logo_hashes:
                doc[page_num].add_redact_annot(fitz.Rect(bbox), fill=(1, 1, 1))
                redacted_regions += 1
        for page in doc:
            page.apply_redactions()

        scratch_path = f"{file_path}.delogo.pdf"
        doc.save(scratch_path)
        logger.info(
            "Logo-strip pre-pass: wrote de-logoed scratch copy to %s "
            "(%d region(s) redacted across %d distinct image(s)).",
            scratch_path, redacted_regions, len(logo_hashes),
        )
        return scratch_path
    finally:
        doc.close()


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
        """
        converter = DoclingPDFParser._get_converter()

        # 2026-08-05 fix: strip repeated logo/watermark images before
        # Docling ever opens the file -- see module docstring and
        # _strip_repeated_page_images above. NOTE: this is a module-level
        # function, not a method on this class -- call it bare, not via
        # DoclingPDFParser.<name>.
        clean_path = _strip_repeated_page_images(file_path)
        made_scratch_copy = clean_path != file_path

        try:
            result = converter.convert(clean_path, raises_on_error=False)
            if not hasattr(result, "document"):
                result = next(iter(result))

            page_errors = [e.error_message for e in result.errors]

            if result.status == ConversionStatus.FAILURE:
                raise ValueError(
                    f"Docling failed to convert {file_path}: "
                    f"{'; '.join(page_errors) if page_errors else 'no error detail available'}"
                )

            return {
                "document": result.document,
                "status": result.status,
                "page_errors": page_errors,
            }
        finally:
            if made_scratch_copy:
                try:
                    os.remove(clean_path)
                except OSError as e:
                    logger.warning(
                        "Failed to clean up logo-strip scratch file %s: %s",
                        clean_path, e,
                    )

    @staticmethod
    def _retry_batch_at_page_level(batch, converter) -> dict:
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
        converter = DoclingPDFParser._get_converter()

        # 2026-08-05 fix: strip repeated logo/watermark images ONCE up
        # front, before splitting into batches -- so every batch (and
        # every single-page retry) already sees the cleaned file, rather
        # than re-detecting/re-stripping per batch. NOTE: module-level
        # function, not a method on this class -- call it bare.
        clean_path = _strip_repeated_page_images(file_path)
        made_scratch_copy = clean_path != file_path

        batches = split_pdf_into_batches(clean_path, batch_size=batch_size)

        succeeded: list[dict] = []
        page_errors: list[str] = []
        fallback_pages: dict = {}
        pages_fully_lost: list[int] = []

        def _recover_failed_batch(batch) -> None:
            is_single_page = (batch.end_page - batch.start_page) <= 1

            if is_single_page:
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
                # NOTE: fallback text extraction intentionally reads from
                # clean_path (the de-logoed copy, or the original if no
                # logo was stripped) so page numbering stays consistent
                # with everything else in this method.
                fallback_text = DoclingPDFParser._fallback_extract_page_text(clean_path, p)
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

                    batch_errors = [
                        f"[pages {batch.start_page + 1}-{batch.end_page}] {e.error_message}"
                        for e in result.errors
                    ]

                    if result.status == ConversionStatus.FAILURE:
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
            cleanup_batches(batches)
            if made_scratch_copy:
                try:
                    os.remove(clean_path)
                except OSError as e:
                    logger.warning(
                        "Failed to clean up logo-strip scratch file %s: %s",
                        clean_path, e,
                    )

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
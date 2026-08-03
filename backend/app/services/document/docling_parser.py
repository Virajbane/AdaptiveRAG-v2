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
        """
        converter = DoclingPDFParser._get_converter()
        result = converter.convert(file_path, raises_on_error=False)
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
        batches = split_pdf_into_batches(file_path, batch_size=batch_size)

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
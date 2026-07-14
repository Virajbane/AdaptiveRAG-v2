"""
Splits a PDF into small page-range mini-PDFs before handing to Docling,
so peak memory per Docling run is bounded (~batch_size pages) instead of
scaling with total document size.

Reconstructed per handoff notes — verified standalone on a 22-page test
PDF (produced 5,5,5,5,2 batches with correct boundaries, temp files
cleaned up). Swap this out if your tested version differs in signature.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass

from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


@dataclass
class PdfBatch:
    file_path: str        # path to the temp mini-PDF on disk
    start_page: int        # 0-indexed, inclusive, original-document numbering
    end_page: int           # 0-indexed, exclusive, original-document numbering
    original_file_path: str  # kept for logging / error messages


def split_pdf_into_batches(file_path: str, batch_size: int = 5) -> list[PdfBatch]:
    """
    Split `file_path` into consecutive page-range mini-PDFs of at most
    `batch_size` pages each. The last batch may have fewer pages.

    Returns a list of PdfBatch, each pointing at a real temp file on disk.
    Batch count adapts automatically to document length (e.g. a 22-page
    doc with batch_size=5 -> 5 batches of 5,5,5,5,2 pages).
    """
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)

    if total_pages == 0:
        raise ValueError(f"PDF has no pages: {file_path}")

    batches: list[PdfBatch] = []
    tmp_dir = tempfile.mkdtemp(prefix="pdf_batch_")

    for start in range(0, total_pages, batch_size):
        end = min(start + batch_size, total_pages)

        writer = PdfWriter()
        for page_idx in range(start, end):
            writer.add_page(reader.pages[page_idx])

        batch_filename = os.path.join(
            tmp_dir, f"batch_{start:04d}_{end:04d}.pdf"
        )
        with open(batch_filename, "wb") as f:
            writer.write(f)

        batches.append(
            PdfBatch(
                file_path=batch_filename,
                start_page=start,
                end_page=end,
                original_file_path=file_path,
            )
        )

    logger.info(
        "Split %s (%d pages) into %d batches of up to %d pages each",
        file_path, total_pages, len(batches), batch_size,
    )
    return batches


def cleanup_batches(batches: list[PdfBatch]) -> None:
    """Delete all temp mini-PDF files (and their shared temp dir if empty)."""
    if not batches:
        return

    tmp_dirs = set()
    for batch in batches:
        tmp_dirs.add(os.path.dirname(batch.file_path))
        try:
            if os.path.exists(batch.file_path):
                os.remove(batch.file_path)
        except OSError as e:
            logger.warning("Failed to remove batch temp file %s: %s", batch.file_path, e)

    for d in tmp_dirs:
        try:
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
        except OSError as e:
            logger.warning("Failed to remove temp batch dir %s: %s", d, e)
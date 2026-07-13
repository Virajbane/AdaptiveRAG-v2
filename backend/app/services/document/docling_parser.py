"""
docling_parser.py
...
"""

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import ConversionStatus


class DoclingPDFParser:
    _converter = None  # lazy singleton -- avoid re-loading layout/table models per call

    @classmethod
    def _get_converter(cls) -> DocumentConverter:
        if cls._converter is None:
            cls._converter = DocumentConverter()
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
    def to_plain_text(doc, max_chars: int = 2500) -> str:
        """
        Unchanged -- still takes the raw DoclingDocument, not the new dict.
        Caller must unwrap parse()'s return before passing here.
        """
        return doc.export_to_markdown()[:max_chars]
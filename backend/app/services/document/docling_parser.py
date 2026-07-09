"""
docling_parser.py

Docling-based PDF parsing, replacing PyMuPDF + regex table/hyphenation
hacks in parser.py's PDFParser for the PDF path only. DOCX/TXT/CSV
continue through the existing DocumentParser unchanged.

Returns the raw docling Document object (not a string) -- chunking
happens directly against this structured object via DoclingChunker,
not against flattened text. The one place raw text is still needed is
MetadataExtractor (title/author extraction), so we also expose a
plain-text export for that single use.
"""

from docling.document_converter import DocumentConverter


class DoclingPDFParser:
    _converter = None  # lazy singleton -- avoid re-loading layout/table models per call

    @classmethod
    def _get_converter(cls) -> DocumentConverter:
        if cls._converter is None:
            cls._converter = DocumentConverter()
        return cls._converter

    @staticmethod
    def parse(file_path: str):
        """
        Returns the docling Document object. Pass this directly to
        DoclingChunker.chunk(doc) -- do not stringify it first, that
        would throw away the table/heading structure this whole
        migration exists to preserve.
        """
        converter = DoclingPDFParser._get_converter()
        result = converter.convert(file_path)
        return result.document

    @staticmethod
    def to_plain_text(doc, max_chars: int = 2500) -> str:
        """
        For MetadataExtractor only, which needs a plain string of the
        opening text, not the structured document. Markdown export is
        good enough for an LLM prompt -- headers/bold markers don't
        confuse title/author extraction and this avoids writing a
        second plain-text exporter.
        """
        return doc.export_to_markdown()[:max_chars]

"""
test_docling.py

Standalone check: does Docling correctly extract table structure and
respect heading boundaries on our actual Lychee-FD paper (or any PDF)?

First run will download Docling's layout + table-structure models from
Hugging Face (similar one-time download to BGE reranker) -- expect a
delay on first execution only.

Run:
    python test_docling.py path/to/your/paper.pdf
"""

import sys

from docling.document_converter import DocumentConverter


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_docling.py path/to/paper.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print(f"Converting {pdf_path} ... (first run downloads models, be patient)")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    print("\n" + "=" * 60)
    print(f"Tables detected: {len(doc.tables)}")
    print("=" * 60)
    for i, table in enumerate(doc.tables):
        print(f"\n--- Table {i+1} ---")
        try:
            df = table.export_to_dataframe()
            print(df)
        except Exception as e:
            print(f"(could not export as dataframe: {e})")

    print("\n" + "=" * 60)
    print("Markdown export (first 3000 chars) -- check heading structure")
    print("=" * 60)
    md = doc.export_to_markdown()
    print(md[:3000])

    print("\n" + "=" * 60)
    print("Item-level structure (first 40 items) -- check heading labels present")
    print("=" * 60)
    for i, (item, level) in enumerate(doc.iterate_items()):
        if i >= 40:
            break
        label = getattr(item, "label", type(item).__name__)
        text_preview = getattr(item, "text", "")[:60] if hasattr(item, "text") else ""
        print(f"[{i}] level={level} label={label} text={text_preview!r}")


if __name__ == "__main__":
    main()
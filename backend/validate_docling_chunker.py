"""
validate_docling_chunker.py

Run this against the real paper to confirm docling_chunker.py actually
fixes the two golden-set failures found via eval_rag.py:
  - "39.4" (TriviaQA S->S accuracy, previously a bare number with no
     column label)
  - "140K" (dataset size, previously merged into a chunk that also
     contained unrelated CosyVoice content AND the start of the next
     section "4.3 Implementation Details")

Run:
    python validate_docling_chunker.py "2607.06540v1.pdf"
"""

import sys
from docling.document_converter import DocumentConverter
from docling_chunker import DoclingChunker


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_docling_chunker.py path/to/paper.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print(f"Converting {pdf_path} ...")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    print("Chunking with DoclingChunker ...")
    chunker = DoclingChunker()
    chunks = chunker.chunk(doc)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Avg tokens/chunk: {sum(c['tokens'] for c in chunks) // len(chunks)}")

    targets = ["39.4", "140K"]
    for t in targets:
        print(f"\n{'=' * 60}")
        print(f"Chunks containing {t!r}")
        print("=" * 60)
        found = False
        for i, c in enumerate(chunks):
            if t in c["text"]:
                found = True
                print(f"\n--- chunk {i} ({c['tokens']} tokens) ---")
                print(c["text"][:600])
        if not found:
            print("  NOT FOUND -- investigate")

    # Sanity check: make sure no chunk mixes two different section headers
    # (the exact bug that caused chunk 55 in the old chunker)
    print(f"\n{'=' * 60}")
    print("Section-boundary check -- any chunk should reference exactly")
    print("one [Section: ...] header, never two")
    print("=" * 60)
    import re
    bad = 0
    for i, c in enumerate(chunks):
        sections = re.findall(r"\[Section: (.*?)\]", c["text"])
        if len(set(sections)) > 1:
            bad += 1
            print(f"  chunk {i}: mixes sections {set(sections)}")
    if bad == 0:
        print("  OK -- no chunk mixes multiple sections")


if __name__ == "__main__":
    main()
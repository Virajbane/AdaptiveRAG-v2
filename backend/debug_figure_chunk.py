"""
debug_figure_chunk.py

Answers one question: did Figure 4 (the UTMOS bar chart) actually get
turned into a retrievable chunk, and if so, what does it say?

Run it like:
    python debug_figure_chunk.py /path/to/lychee_fd_paper.pdf

What it does:
1. Parses the PDF through the SAME DoclingPDFParser your real ingestion
   pipeline uses (so this reflects reality, not a simplified re-implementation).
2. Runs the SAME DoclingChunker.chunk() your real pipeline uses.
3. Prints every chunk that came from a PICTURE item, in full.
4. Separately searches ALL chunks (picture, table, prose) for "UTMOS" and
   prints those too -- this catches the "the chart's numbers actually
   live in a sentence of prose, not the chart itself" possibility.
5. For every PICTURE item Docling found, prints its raw `.annotations`
   structure -- so if do_picture_description is enabled but chunk_picture
   in docling_chunker.py silently returns None, you can see exactly why
   (this checks the "ASSUMPTION TO VERIFY" flagged in that file: whether
   annotation.text is really the right attribute for your installed
   Docling version).

Read the output like this:
- No PICTURE chunk AND no "UTMOS" chunk containing all 5 models
  -> Reason A: the chart never got read. Check the annotations dump
     at the bottom -- it'll show you why (empty list = VLM call never
     ran or produced nothing; populated list with a different attribute
     name than `.text` = just a one-line fix in _chunk_picture).
- A PICTURE chunk EXISTS and correctly lists Moshi: 3.31
  -> Reason B: the data is fine, the LLM is the one mixing up labels.
     That's a generation-side fix (grounding instruction), not a
     chunking fix.
- A "UTMOS" hit comes from a PROSE chunk (not a PICTURE chunk) mentioning
  only Lychee-FD's own score
  -> Confirms the "prose restates our own number, model grabs it for
     any UTMOS question" theory directly.
"""

import sys

sys.path.insert(0, "backend")  # adjust if you run this from a different cwd

from app.services.document.docling_parser import DoclingPDFParser
from app.services.document.docling_chunker import DoclingChunker


def main(pdf_path: str):
    print(f"Parsing {pdf_path} ...\n")
    parsed = DoclingPDFParser.parse(pdf_path)
    doc = parsed["document"]

    if parsed["page_errors"]:
        print("--- Page errors during parse ---")
        for e in parsed["page_errors"]:
            print(" ", e)
        print()

    # --- Raw picture item inspection (checks the ASSUMPTION TO VERIFY) ---
    print("=" * 70)
    print("RAW PICTURE ITEMS FOUND BY DOCLING")
    print("=" * 70)
    picture_count = 0
    for item, _level in doc.iterate_items():
        label_str = str(getattr(item, "label", None))
        if label_str not in ("DocItemLabel.PICTURE", "picture"):
            continue
        picture_count += 1
        print(f"\n--- Picture #{picture_count} ---")
        annotations = getattr(item, "annotations", None)
        print(f"  annotations attribute present: {annotations is not None}")
        print(f"  annotations value: {annotations!r}")
        if annotations:
            for i, ann in enumerate(annotations):
                print(f"  annotation[{i}] type: {type(ann)}")
                print(f"  annotation[{i}] dir(): {[a for a in dir(ann) if not a.startswith('_')]}")
                print(f"  annotation[{i}] .text (if present): {getattr(ann, 'text', '<no .text attr>')!r}")

    if picture_count == 0:
        print("  No PICTURE items found at all -- Docling isn't detecting "
              "any figures on this document. That's a parsing-level issue, "
              "not a chunking one.")

    # --- Real chunking pass, same as production ---
    print("\n" + "=" * 70)
    print("CHUNKS PRODUCED BY DoclingChunker.chunk()")
    print("=" * 70)
    chunker = DoclingChunker()
    chunks = chunker.chunk(doc)
    print(f"Total chunks: {len(chunks)}\n")

    # Chunks that came from a Figure (we can't tag them at this point without
    # re-instrumenting chunk(), so instead we search by shape: our
    # _chunk_picture format always starts a line with "Figure:")
    figure_chunks = [c for c in chunks if "\nFigure:" in c["text"] or c["text"].startswith("Figure:")]
    print(f"Chunks that look like figure chunks: {len(figure_chunks)}")
    for c in figure_chunks:
        print("\n--- FIGURE CHUNK ---")
        print(c["text"])

    print("\n" + "=" * 70)
    print('ALL CHUNKS MENTIONING "UTMOS" (any type)')
    print("=" * 70)
    utmos_hits = [c for c in chunks if "UTMOS" in c["text"]]
    print(f"Found {len(utmos_hits)} chunk(s):\n")
    for c in utmos_hits:
        print("--- CHUNK ---")
        print(c["text"])
        print()

    if not figure_chunks and not utmos_hits:
        print("Nothing at all mentions UTMOS or looks like a figure chunk. "
              "The data may be getting lost even earlier (Docling not "
              "finding the picture on the page, or filtering it out during "
              "batching) -- check the picture_count above.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_figure_chunk.py /path/to/paper.pdf")
        sys.exit(1)
    main(sys.argv[1])
"""
debug_vision_model_direct.py

Isolates ONE step: "does the vision model actually describe a chart
image correctly, when nothing else in the pipeline can interfere?"

This bypasses do_picture_description entirely and talks to Ollama
directly, using:
  - the SAME endpoint (settings.OLLAMA_BASE_URL)
  - the SAME model (settings.PICTURE_DESCRIPTION_MODEL)
  - the SAME prompt (_PICTURE_DESCRIPTION_PROMPT from docling_parser.py)
so a bad result here means the problem is genuinely in the model/prompt/
image, not in how docling_parser.py wires things together.

Run it like:
    python debug_vision_model_direct.py "path\\to\\paper.pdf" --page 8

2026-08-03 fix (two bugs found via direct inspection):

Bug 1 -- only the FIRST picture found on the target page was ever sent
to the model. Page 8 has more than one PICTURE item (a logo/header
graphic in addition to the actual chart), and iteration order put the
logo first every time -- so every "result" so far was the model
describing a logo, never the chart. Fixed by sending EVERY picture
found on the page, each one clearly labeled by filename, so you can
see which result corresponds to which image.

Bug 2 (the real root cause hiding behind Bug 1) -- the request payload
used Ollama's NATIVE /api/chat shape (top-level "images": [...] on the
message) while POSTing to the OpenAI-compatible /v1/chat/completions
endpoint. That endpoint expects an OpenAI-style vision content block
list ("content": [{"type":"text",...}, {"type":"image_url",...}]).
Sent the old way, most OpenAI-compat shims silently ignore the
unrecognized "images" field -- meaning the model was answering with
NO image input at all, regardless of resolution. This explains
plausible-but-wrong / coordinate-looking output independent of image
size. Fixed to build a proper image_url content block with a base64
data URI.

Read the output like this:
- If a saved PNG in ./debug_images/ does NOT look like the expected
  chart -> that particular image is the wrong picture item (e.g. a
  logo) -- check the OTHER saved images from the same page for the
  real chart, and identify it by filename before trusting its result.
- If the correct chart PNG looks fine but its Ollama response is still
  garbage -> now the bug is genuinely the model/prompt/resolution.
  Try swapping PICTURE_DESCRIPTION_MODEL for a known-good vision model
  (e.g. llava, qwen2-vl) on the SAME image.
- If the correct chart's response IS a clean "label: value" list ->
  the direct call works; compare this script's payload against what
  docling_parser.py's PictureDescriptionApiOptions actually sends, to
  spot any remaining mismatch.
"""

import sys
import argparse
import base64
import json
import os
import io

import requests

sys.path.insert(0, "backend")  # adjust if you run this from a different cwd

from app.config.settings import settings
from app.services.document.docling_parser import DoclingPDFParser

_PROMPT = (
    "This image is a figure from a research paper (bar chart, line chart, "
    "heatmap, or diagram). Transcribe every visible data label and its "
    "exact numeric value as a plain list, one per line, in the form "
    "'label: value'. Include axis labels and legend entries if they "
    "identify which value belongs to which series. Do not describe the "
    "visual style -- only transcribe the labels and numbers exactly as "
    "shown."
)


def _call_ollama_vision(image_bytes: bytes) -> dict:
    """POSTs to the OpenAI-compatible /v1/chat/completions endpoint using
    the content-block shape it actually expects: content is a LIST mixing
    a text block and an image_url block (base64 data URI), not a
    top-level 'images' field on the message."""
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": settings.PICTURE_DESCRIPTION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}"
                        },
                    },
                ],
            }
        ],
        "stream": False,
    }

    url = f"{settings.OLLAMA_BASE_URL}/v1/chat/completions"
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def main(pdf_path: str, target_page: int):
    print(f"Parsing {pdf_path} (this re-runs the full Docling pass, "
          f"including do_picture_description if enabled)...\n")
    parsed = DoclingPDFParser.parse(pdf_path)
    doc = parsed["document"]

    os.makedirs("debug_images", exist_ok=True)

    saved_images = []  # list of (fname, image_bytes) -- ALL pictures on the page

    for item, _level in doc.iterate_items():
        label_str = str(getattr(item, "label", None))
        if label_str not in ("DocItemLabel.PICTURE", "picture"):
            continue

        page_no = None
        try:
            prov = getattr(item, "prov", None)
            if prov:
                page_no = prov[0].page_no  # 1-indexed in most docling versions
        except Exception:
            pass

        if target_page is not None and page_no != target_page:
            continue

        try:
            pil_image = item.get_image(doc)
        except Exception as e:
            print(f"  Picture on page {page_no}: could not get rendered "
                  f"image ({type(e).__name__}: {e}). This means "
                  f"generate_picture_images isn't producing an image for "
                  f"this item -- that's the bug, before the vision model "
                  f"is even reached.")
            continue

        if pil_image is None:
            print(f"  Picture on page {page_no}: get_image() returned None "
                  f"-- no rendered image available for this item.")
            continue

        idx = len(saved_images) + 1
        fname = f"debug_images/page{page_no}_pic{idx}.png"
        pil_image.save(fname)
        print(f"  Saved {fname} ({pil_image.size[0]}x{pil_image.size[1]}) "
              f"-- open this file and confirm which figure this actually is.")

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        saved_images.append((fname, buf.getvalue()))

    if not saved_images:
        print(f"\nNo picture images found on page {target_page} "
              f"(or page numbers aren't exposed the way this script "
              f"expects -- check the 'prov' attribute on a PictureItem "
              f"if this looks wrong). Try running without --page to dump "
              f"every picture in the document instead.")
        return

    print(f"\nFound {len(saved_images)} picture(s) on page {target_page}. "
          f"Sending EACH ONE to Ollama ({settings.OLLAMA_BASE_URL}, "
          f"model={settings.PICTURE_DESCRIPTION_MODEL}) with the production "
          f"prompt, so you can match each result to its filename above.\n")

    for fname, image_bytes in saved_images:
        print(f"=== {fname} ===")
        try:
            result = _call_ollama_vision(image_bytes)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Request failed: {type(e).__name__}: {e}")
            print("If this is a 4xx/shape error, double-check your Ollama "
                  "version actually serves an OpenAI-compatible "
                  "/v1/chat/completions route with image_url content "
                  "blocks -- older Ollama versions may only support the "
                  "native /api/chat + 'images' field shape, in which case "
                  "swap the URL and payload in _call_ollama_vision to match.")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--page", type=int, default=None,
                         help="1-indexed page number to isolate (e.g. 8 for "
                              "the Figure 4 UTMOS chart's page). Omit to "
                              "dump every picture in the document.")
    args = parser.parse_args()
    main(args.pdf_path, args.page)
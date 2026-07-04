import json
import re
from typing import Optional


class MetadataExtractor:
    """
    Extracts document-level metadata (title, authors, affiliations) once
    at ingestion time via a single LLM call on the opening text.

    Why this exists: metadata questions ("what is the title", "who are
    the authors") have near-zero lexical/semantic overlap with the
    metadata text itself — the title never contains the word "title".
    No amount of embedding-model tuning fixes that; retrieval isn't the
    right tool for this class of question. Extract once, store
    separately, answer directly — skip retrieval entirely.
    """

    def __init__(self, llm):
        # Use the main/deep-reasoning llm, not fast_llm — same lesson
        # learned from the AnswerAgent wiring bug (2026-07-04).
        self.llm = llm

    async def extract(self, full_text: str) -> Optional[dict]:
        """
        Returns {"title": str, "authors": [str], "affiliations": [str]}
        or None if extraction fails — caller should treat None as
        "no metadata available, fall back to normal retrieval" rather
        than raising, since this is a best-effort enhancement, not a
        required step.
        """
        opening_text = full_text[:2500]

        prompt = f"""Extract metadata from the opening of this document. Output only JSON, no explanation.

Text:
{opening_text}

Rules:
- title: the full document/paper title, exactly as written, no truncation
- authors: list of author names only, no affiliations markers/numbers
- affiliations: list of institutions/organizations mentioned as author affiliations
- If a field genuinely isn't present in this text, use null (title) or [] (authors/affiliations)
- Do not guess or invent values not present in the text

{{"title": "...", "authors": ["..."], "affiliations": ["..."]}}

Start with {{"""

        try:
            response = await self.llm.generate(prompt)
            metadata = self._parse_json(response)

            if not metadata or not metadata.get("title"):
                print("[METADATA] Extraction produced no usable title, skipping")
                return None

            print(f"[METADATA] Extracted title: {metadata.get('title')!r}")
            print(f"[METADATA] Extracted authors: {metadata.get('authors')}")
            print(f"[METADATA] Extracted affiliations: {metadata.get('affiliations')}")
            return metadata

        except Exception as e:
            print(f"[METADATA] Extraction failed, continuing without it: {e}")
            return None

    def _parse_json(self, response: str) -> Optional[dict]:
        text = response.strip()
        if not text.startswith("{"):
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            text = match.group(0)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
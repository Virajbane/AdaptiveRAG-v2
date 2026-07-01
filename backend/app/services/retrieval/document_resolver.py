"""
backend/app/services/retrieval/document_resolver.py

Resolves an optional document_id filter from the question text, by
fuzzy-matching against the user's uploaded filenames.

Why this exists: chunks already carry `doc_id` in both Qdrant and the
BM25 index, but nothing ever filtered on it — search always pooled
every document a user has uploaded. When a question references a
specific file ("the resume", "the rag2.0 pdf"), that pooling lets
chunks from unrelated documents drown out the right one (confirmed by
production logs: a 26-chunk tracker doc out-scored a 4-chunk resume
doc on a resume-specific query).

This stays a soft signal, not a hard requirement: if no confident
match is found, return None and callers should search across all of
the user's documents exactly as before (no behavior change for
generic questions that don't name a file).
"""

import difflib
import re
from typing import Optional

from app.db.mongodb.queries import DocumentQueries

# Below this similarity score, treat it as "no real match" rather than
# force a possibly-wrong filter onto retrieval.
MATCH_THRESHOLD = 0.45


def _normalize(name: str) -> str:
    """Strip extension, lowercase, split on any non-alphanumeric run.

    Filenames often contain parentheses, dots, spaces mixed together
    (e.g. "Viraj Resume(JUNE 2026).pdf") — splitting only on
    underscore/hyphen left punctuation glued to words ("resume(june"),
    which then failed to match the same word appearing cleanly in the
    question. Splitting on ANY non-alnum run avoids that.
    """
    name = re.sub(r"\.\w+$", "", name)  # drop final extension (.pdf/.docx/etc)
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return name.strip()


async def resolve_document_filter(question: str, user_id: str, db) -> Optional[str]:
    """
    Returns a doc_id string if the question confidently references one
    of the user's uploaded documents, otherwise None.
    """
    if db is None:
        print("[DOC_RESOLVER] db is None, skipping filter")
        return None

    try:
        doc_queries = DocumentQueries(db)
        docs = await doc_queries.list_documents(user_id)
    except Exception as e:
        print(f"[DOC_RESOLVER] Failed to list documents, skipping filter: {e}")
        return None

    print(f"[DOC_RESOLVER] user_id={user_id} found {len(docs)} documents: "
          f"{[d.get('filename') for d in docs]}")

    if not docs:
        print("[DOC_RESOLVER] No documents for this user, skipping filter")
        return None

    question_lower = question.lower()

    best_score = 0.0
    best_doc_id = None

    for doc in docs:
        filename = doc.get("filename", "")
        if not filename:
            continue

        norm_name = _normalize(filename)
        name_tokens = norm_name.split()

        # Token-overlap signal: how many filename words appear in the question.
        # Catches cases like "the resume pdf" matching "Viraj_Resume.pdf"
        # even though full sequence-match similarity would be low.
        token_hits = sum(1 for tok in name_tokens if len(tok) > 2 and tok in question_lower)
        token_score = token_hits / len(name_tokens) if name_tokens else 0.0

        # Sequence-similarity signal: catches close-but-not-exact matches.
        seq_score = difflib.SequenceMatcher(None, norm_name, question_lower).ratio()

        score = max(token_score, seq_score)

        if score > best_score:
            best_score = score
            best_doc_id = str(doc.get("_id"))

    if best_score >= MATCH_THRESHOLD:
        print(f"[DOC_RESOLVER] Matched document_id={best_doc_id} (score={best_score:.2f})")
        return best_doc_id

    print(f"[DOC_RESOLVER] Best match score={best_score:.2f} below threshold "
          f"({MATCH_THRESHOLD}), no filter applied")
    return None
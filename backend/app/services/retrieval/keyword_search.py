from rank_bm25 import BM25Plus
from typing import List, Optional
import re

class KeywordSearchEngine:
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.doc_ids = []

    def build_index(self, chunks: List[dict]):
        tokenized_chunks = []
        self.documents = []
        self.doc_ids = []

        for chunk in chunks:
            tokens = self._tokenize(chunk['text'])
            tokenized_chunks.append(tokens)
            self.documents.append(chunk['text'])
            self.doc_ids.append({
                'doc_id': chunk['doc_id'],
                'chunk_index': chunk['chunk_index'],
                'text': chunk['text']
            })

        # FIX (2026-07-14) -- BM25Plus computes avg doc length internally
        # as sum(doc_lengths) / len(corpus); called with an EMPTY corpus
        # (e.g. removing a user's last remaining document leaves
        # self.user_chunks[user_id] == []) that's a division by zero.
        # Guard here, in the one place every caller (index_document,
        # remove_document, rebuild_from_chunks) already funnels through,
        # rather than patching each call site separately. self.bm25
        # stays None, which search() already checks for and handles by
        # returning [] -- correct behavior for "this user has no
        # indexed documents right now."
        if not tokenized_chunks:
            self.bm25 = None
            return

        self.bm25 = BM25Plus(tokenized_chunks)

    def search(self, query: str, top_k: int = 10, document_id: Optional[str] = None) -> List[dict]:
        if not self.bm25:
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # If scoped to a document, only rank chunks from that document.
        # Filtering BEFORE taking top_k (not after) so a doc-scoped search
        # still returns top_k results from within that doc, rather than
        # top_k overall results that happen to survive a post-filter.
        candidate_indices = range(len(scores))
        if document_id:
            candidate_indices = [
                i for i in candidate_indices
                if self.doc_ids[i]['doc_id'] == document_id
            ]

        top_indices = sorted(
            candidate_indices,
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'doc_id': self.doc_ids[idx]['doc_id'],
                    'chunk_index': self.doc_ids[idx]['chunk_index'],
                    'text': self.documents[idx],
                    'score': float(scores[idx]),
                    'search_type': 'keyword'
                })
        return results

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        return re.findall(r'\w+', text)


class KeywordSearchManager:
    """
    2026-07-04 fix — replace-on-reindex:
      index_document() previously only ever called
      self.user_chunks[user_id].extend(chunks), which meant re-ingesting
      the SAME document (e.g. after a chunker bugfix, or a normal
      re-upload) added a second, parallel copy of its chunks alongside
      the old ones instead of replacing them. BM25 has no versioning
      concept, so both copies scored and surfaced in results
      simultaneously.

      This was silently present before, but became visible once the
      chunker's punctuation-preservation fix changed chunk text/ranking
      enough to surface the duplicate in the observed eval's top-5.
      Confirmed directly in eval logs: the identical chunk text appeared
      twice in one result set under two different doc_ids - one
      resolving to a real filename via RetrieverAgent._attach_filenames'
      Mongo lookup, and one (the orphaned old doc_id, since superseded/
      deleted in Mongo on re-upload) resolving to "Unknown document".
      The filename-lookup code was working correctly; the bug was BM25
      holding a stale doc_id that no longer had a matching Mongo record
      at all.

      Fix: index_document() now removes any existing chunks whose
      doc_id matches one of the doc_ids being (re-)indexed before
      extending, so re-indexing a document replaces it rather than
      duplicating it. remove_document() is also added for explicit use
      by a document-delete endpoint, so it can keep BM25 in sync with
      Qdrant (QdrantVectorDB.delete_document_vectors) and Mongo instead
      of leaving orphaned chunks behind on deletion too.

    2026-07-14 fix — empty-corpus division by zero:
      remove_document() (and, in principle, index_document()) can leave
      user_chunks[user_id] == [] once a user's LAST remaining document
      is removed. Rebuilding BM25Plus on an empty corpus divides by
      zero internally (avg doc length = sum/len(corpus)). Fixed at the
      source in KeywordSearchEngine.build_index() -- see comment there.
    """

    def __init__(self):
        self.user_indexes = {}
        self.user_chunks = {}

    async def index_document(self, user_id: str, chunks: List[dict]):
        if not chunks:
            return

        if user_id not in self.user_chunks:
            self.user_chunks[user_id] = []

        # Replace, don't append: drop any existing chunks for the
        # doc_id(s) being indexed now, so re-ingesting a document
        # (same doc_id re-processed, or a new doc_id superseding an
        # old one for "the same" document from the user's perspective)
        # can't leave a stale duplicate copy sitting in the index.
        doc_ids_being_indexed = {c['doc_id'] for c in chunks}
        self.user_chunks[user_id] = [
            c for c in self.user_chunks[user_id]
            if c['doc_id'] not in doc_ids_being_indexed
        ]
        self.user_chunks[user_id].extend(chunks)

        if user_id not in self.user_indexes:
            self.user_indexes[user_id] = KeywordSearchEngine()

        self.user_indexes[user_id].build_index(self.user_chunks[user_id])

    async def remove_document(self, user_id: str, doc_id: str):
        """
        Remove all chunks for a specific document from this user's BM25
        index. Call this from the document-delete endpoint alongside
        QdrantVectorDB.delete_document_vectors(), so a deleted document
        doesn't leave orphaned chunks behind in BM25 whose doc_id no
        longer has a Mongo record — those chunks would keep surfacing
        in hybrid search results and resolve to "Unknown document" once
        their filename lookup fails, exactly like the accumulation bug
        above.

        Safe to call when this removes the user's LAST remaining
        document -- build_index() handles the resulting empty corpus
        without dividing by zero (see 2026-07-14 fix note above).
        """
        if user_id not in self.user_chunks:
            return

        self.user_chunks[user_id] = [
            c for c in self.user_chunks[user_id] if c['doc_id'] != doc_id
        ]
        self.user_indexes[user_id] = KeywordSearchEngine()
        self.user_indexes[user_id].build_index(self.user_chunks[user_id])

    async def rebuild_from_chunks(self, user_id: str, chunks: List[dict]):
        """Rebuild full index from scratch — used on startup."""
        self.user_chunks[user_id] = chunks
        self.user_indexes[user_id] = KeywordSearchEngine()
        self.user_indexes[user_id].build_index(chunks)

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        document_id: Optional[str] = None,
    ) -> List[dict]:
        if user_id not in self.user_indexes:
            return []
        return self.user_indexes[user_id].search(query, top_k, document_id=document_id)


keyword_manager = KeywordSearchManager()
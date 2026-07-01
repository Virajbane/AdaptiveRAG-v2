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
    def __init__(self):
        self.user_indexes = {}
        self.user_chunks = {}

    async def index_document(self, user_id: str, chunks: List[dict]):
        if user_id not in self.user_chunks:
            self.user_chunks[user_id] = []
        self.user_chunks[user_id].extend(chunks)

        if user_id not in self.user_indexes:
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
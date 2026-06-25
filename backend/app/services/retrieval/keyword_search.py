from rank_bm25 import BM25Plus
from typing import List
import re

class KeywordSearchEngine:
    """BM25 keyword search implementation"""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []  # Store original docs for retrieval
        self.doc_ids = []    # Store corresponding doc IDs
    
    def build_index(self, chunks: List[dict]):
        """
        Build BM25 index from document chunks
        
        Args:
            chunks: List of dicts with 'doc_id', 'text', 'chunk_index'
        """
        # Tokenize all chunks
        tokenized_chunks = []
        self.documents = []
        self.doc_ids = []
        
        for chunk in chunks:
            # Simple tokenization (split on whitespace and lowercase)
            tokens = self._tokenize(chunk['text'])
            tokenized_chunks.append(tokens)
            self.documents.append(chunk['text'])
            self.doc_ids.append({
                'doc_id': chunk['doc_id'],
                'chunk_index': chunk['chunk_index'],
                'text': chunk['text']
            })
        
        # Build BM25 index (BM25Plus: IDF floor via delta, so small
        # corpora don't collapse exact matches to a zero/negative score)
        self.bm25 = BM25Plus(tokenized_chunks)
    
    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """
        Search using BM25
        
        Args:
            query: Search query string
            top_k: Number of results to return
        
        Returns:
            List of results with scores
        """
        if not self.bm25:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include genuinely positive relevance
                results.append({
                    'doc_id': self.doc_ids[idx]['doc_id'],
                    'chunk_index': self.doc_ids[idx]['chunk_index'],
                    'text': self.documents[idx],
                    'score': float(scores[idx]),
                    'search_type': 'keyword'
                })
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and split
        tokens = re.findall(r'\w+', text)
        return tokens
    
# Add this class to build and maintain keyword index

class KeywordSearchManager:
    """Manage BM25 indexes per user"""
    
    def __init__(self):
        self.user_indexes = {}  # user_id -> BM25 index
        self.user_chunks = {}   # user_id -> all chunks seen so far
    
    async def index_document(
        self,
        user_id: str,
        chunks: List[dict]
    ):
        """
        Add document chunks to user's keyword index.
        Accumulates chunks across multiple documents instead of
        wiping out previous documents' chunks.
        """
        # Keep a running list of every chunk this user has uploaded
        if user_id not in self.user_chunks:
            self.user_chunks[user_id] = []
        
        self.user_chunks[user_id].extend(chunks)
        
        # Rebuild the BM25 index from the full accumulated chunk list
        if user_id not in self.user_indexes:
            self.user_indexes[user_id] = KeywordSearchEngine()
        
        self.user_indexes[user_id].build_index(self.user_chunks[user_id])
    
    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 10
    ) -> List[dict]:
        """
        Search user's keyword index
        """
        if user_id not in self.user_indexes:
            return []
        
        return self.user_indexes[user_id].search(query, top_k)

# Global manager instance
keyword_manager = KeywordSearchManager()
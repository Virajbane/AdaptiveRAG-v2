import re
from typing import List
import tiktoken

class TextChunker:
    """Split text into chunks of ~512 tokens"""
    
    # Token counter (using tiktoken)
    def __init__(self, model: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(model)
        self.chunk_size = 512
        self.overlap = 50  # tokens overlap between chunks
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def chunk(self, text: str) -> List[dict]:
        """
        Split text into chunks of ~512 tokens
        with 50-token overlap for context preservation
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split by sentences first (preserve meaning)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence exceeds chunk size
            if current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "tokens": current_tokens
                    })
                
                # Start new chunk with overlap
                # Take last few sentences for context (50 tokens)
                if chunks:
                    overlap_text = self._get_overlap(current_chunk)
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                
                current_tokens = self.count_tokens(current_chunk)
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "tokens": self.count_tokens(current_chunk.strip())
            })
        
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """Get last ~50 tokens from text for overlap"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        overlap_sentences = []
        token_count = 0
        
        # Work backwards through sentences
        for sentence in reversed(sentences):
            tokens = self.count_tokens(sentence)
            if token_count + tokens <= self.overlap:
                overlap_sentences.insert(0, sentence)
                token_count += tokens
            else:
                break
        
        return " ".join(overlap_sentences)
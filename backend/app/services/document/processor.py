from app.services.document.parser import DocumentParser
from app.services.document.chunker import TextChunker
from app.services.document.embedder import EmbeddingGenerator
from app.db.qdrant.client import QdrantVectorDB

class DocumentProcessor:
    """Orchestrate document processing pipeline"""
    
    def __init__(self, db):
        self.db = db
        self.parser = DocumentParser()
        self.chunker = TextChunker()
        self.embedder = EmbeddingGenerator()
        self.vector_db = QdrantVectorDB()
    
    async def process(
        self,
        file_path: str,
        file_type: str,
        user_id: str,
        doc_id: str
    ) -> dict:
        """
        Process document end-to-end:
        1. Parse
        2. Chunk
        3. Embed
        4. Store vectors
        """
        
        # 1. Parse document
        print(f"Parsing {file_type} document...")
        text = self.parser.parse(file_path, file_type)
        
        if not text.strip():
            raise ValueError("Document is empty after parsing")
        
        # 2. Chunk text
        print(f"Chunking text...")
        chunks = self.chunker.chunk(text)
        
        if not chunks:
            raise ValueError("No chunks generated from document")
        
        for i, chunk in enumerate(chunks):
            chunk["doc_id"] = doc_id
            chunk["chunk_index"] = i

        # Index chunks for keyword (BM25) search
        from app.services.retrieval.keyword_search import keyword_manager
        await keyword_manager.index_document(user_id, chunks)
        
        # 3. Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        chunk_texts = [c["text"] for c in chunks]
        embeddings = await self.embedder.embed_batch(chunk_texts)
        
        # 4. Store in Qdrant
        print(f"Storing vectors in Qdrant...")
        vectors_stored = await self.vector_db.store_vectors(
            doc_id=doc_id,
            user_id=user_id,
            chunks=chunks,
            embeddings=embeddings
        )
        
        return {
            "chunk_count": len(chunks),
            "avg_tokens": sum(c["tokens"] for c in chunks) // len(chunks),
            "total_tokens": sum(c["tokens"] for c in chunks),
            "vectors_stored": vectors_stored
        }
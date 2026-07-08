from app.services.document.parser import DocumentParser
from app.services.document.chunker import TextChunker
from app.services.document.embedder import EmbeddingGenerator
from app.services.document.metadata_extractor import MetadataExtractor
from app.db.qdrant.client import QdrantVectorDB
from bson import ObjectId
from app.config.settings import settings
class DocumentProcessor:
    """Orchestrate document processing pipeline"""

    def __init__(self, db, llm):
        self.db = db
        self.parser = DocumentParser()
        self.chunker = TextChunker()
        self.embedder = EmbeddingGenerator(ollama_url=settings.OLLAMA_BASE_URL)
        self.vector_db = QdrantVectorDB()
        self.metadata_extractor = MetadataExtractor(llm)  # pass main llm, not fast_llm

    async def process(
        self,
        file_path: str,
        file_type: str,
        user_id: str,
        doc_id: str
    ) -> dict:

        print(f"Parsing {file_type} document...")
        text = self.parser.parse(file_path, file_type)

        if not text.strip():
            raise ValueError("Document is empty after parsing")

        # Extract metadata BEFORE chunking — needs the raw, unsegmented
        # opening text, not a 150-token chunk of it.
        print("Extracting document metadata...")
        metadata = await self.metadata_extractor.extract(text)
        if metadata:
            update_result = await self.db.documents.update_one(
                {"_id": ObjectId(doc_id)},
                {"$set": {"metadata": metadata}}
            )
            if update_result.matched_count == 0:
                print(f"[METADATA] WARNING: update_one matched 0 documents "
                      f"for doc_id={doc_id!r} — metadata not saved")
            else:
                print(f"[METADATA] Saved metadata for doc_id={doc_id}")

        print(f"Chunking text...")
        chunks = self.chunker.chunk(text)

        if not chunks:
            raise ValueError("No chunks generated from document")

        for i, chunk in enumerate(chunks):
            chunk["doc_id"] = doc_id
            chunk["chunk_index"] = i

        from app.services.retrieval.keyword_search import keyword_manager
        await keyword_manager.index_document(user_id, chunks)

        print(f"Generating embeddings for {len(chunks)} chunks...")
        chunk_texts = [c["text"] for c in chunks]
        embeddings = await self.embedder.embed_batch(chunk_texts)

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
            "vectors_stored": vectors_stored,
            "metadata": metadata,
        }
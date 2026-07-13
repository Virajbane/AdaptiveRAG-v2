from app.services.document.parser import DocumentParser
from app.services.document.chunker import TextChunker
from app.services.document.docling_parser import DoclingPDFParser
from app.services.document.docling_chunker import DoclingChunker
from app.services.document.embedder import EmbeddingGenerator
from app.services.document.metadata_extractor import MetadataExtractor
from app.db.qdrant.client import QdrantVectorDB
from bson import ObjectId
from app.config.settings import settings


class DocumentProcessor:
    def __init__(self, db, llm):
        self.db = db
        self.parser = DocumentParser()
        self.chunker = TextChunker()
        self.docling_chunker = DoclingChunker()
        self.embedder = EmbeddingGenerator(ollama_url=settings.OLLAMA_BASE_URL)
        self.vector_db = QdrantVectorDB()
        self.metadata_extractor = MetadataExtractor(llm)

    async def process(
        self,
        file_path: str,
        file_type: str,
        user_id: str,
        doc_id: str
    ) -> dict:

        file_type = file_type.lower()
        use_docling = file_type == "pdf"

        if use_docling:
            print("Parsing PDF via Docling (structure-aware)...")
            doc = DoclingPDFParser.parse(file_path)
            opening_text = DoclingPDFParser.to_plain_text(doc)
        else:
            print(f"Parsing {file_type} document...")
            text = self.parser.parse(file_path, file_type)
            if not text.strip():
                raise ValueError("Document is empty after parsing")
            opening_text = text[:2500]

        print("Extracting document metadata...")
        metadata = await self.metadata_extractor.extract(opening_text)
        if metadata:
            update_result = await self.db.documents.update_one(
                {"_id": ObjectId(doc_id)},
                {"$set": {"metadata": metadata}}
            )
            if update_result.matched_count == 0:
                print(f"[METADATA] WARNING: update_one matched 0 documents "
                      f"for doc_id={doc_id!r} — metadata not saved")

        print("Chunking...")
        if use_docling:
            chunks = self.docling_chunker.chunk(doc)
        else:
            chunks = self.chunker.chunk(text)

        if not chunks:
            raise ValueError("No chunks generated from document")

        for i, chunk in enumerate(chunks):
            chunk["doc_id"] = doc_id
            chunk["chunk_index"] = i

        # Keyword index still gets ALL chunks -- BM25 indexing is local/
        # in-memory and doesn't have the same partial-failure profile as
        # embedding/Qdrant. If this becomes unreliable too, it gets its
        # own stage later -- not folding it in blindly here.
        from app.services.retrieval.keyword_search import keyword_manager
        await keyword_manager.index_document(user_id, chunks)

        print(f"Generating embeddings for {len(chunks)} chunks...")
        chunk_texts = [c["text"] for c in chunks]
        embed_result = await self.embedder.embed_batch(chunk_texts)

        # Only pass chunks/embeddings that actually succeeded into
        # store_vectors -- keeps the two lists aligned by construction,
        # rather than relying on store_vectors to sort it out.
        successful_chunks = []
        successful_embeddings = []
        embed_failed_indices = set(embed_result["failed_indices"])

        for chunk, embedding in zip(chunks, embed_result["embeddings"]):
            if embedding is not None:
                successful_chunks.append(chunk)
                successful_embeddings.append(embedding)

        if not successful_chunks:
            # Every single chunk failed to embed -- this is a genuine
            # total failure, not a partial one. Raise so the document
            # gets marked "failed", not "processed_with_gaps" with zero
            # actual content indexed.
            raise ValueError(
                f"All {len(chunks)} chunks failed to embed -- "
                f"see [EMBED FAILED] logs above for details"
            )

        print(f"Storing {len(successful_chunks)} vectors in Qdrant "
              f"({len(embed_failed_indices)} skipped due to embed failure)...")
        store_result = await self.vector_db.store_vectors(
            doc_id=doc_id,
            user_id=user_id,
            chunks=successful_chunks,
            embeddings=successful_embeddings
        )

        # Combine failure sources: chunks that never got embedded, plus
        # chunks that embedded fine but failed to upsert to Qdrant.
        # These are DIFFERENT chunk_index sets (store_result's indices
        # are positions within successful_chunks, not the original
        # chunk list) -- need to map back to original indices.
        qdrant_failed_original_indices = [
            successful_chunks[i]["chunk_index"]
            for i in store_result["failed_chunk_indices"]
        ]

        all_failed_indices = sorted(embed_failed_indices | set(qdrant_failed_original_indices))
        total_chunks = len(chunks)
        total_failed = len(all_failed_indices)
        total_succeeded = total_chunks - total_failed

        return {
            "chunk_count": total_chunks,
            "chunks_stored": total_succeeded,
            "chunks_failed": total_failed,
            "failed_chunk_indices": all_failed_indices,
            "avg_tokens": sum(c["tokens"] for c in chunks) // len(chunks),
            "total_tokens": sum(c["tokens"] for c in chunks),
            "metadata": metadata,
        }
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

        page_errors = []  # only ever populated for the Docling/PDF path

        if use_docling:
            print("Parsing PDF via Docling (structure-aware)...")
            # FIXED: DoclingPDFParser.parse() now returns a dict
            # ({"document", "status", "page_errors"}), not the bare
            # DoclingDocument -- must unwrap before using it.
            parse_result = DoclingPDFParser.parse(file_path)
            doc = parse_result["document"]
            page_errors = parse_result["page_errors"]
            if page_errors:
                # Not yet folded into chunks_failed/status decision --
                # logged for now so it's at least visible, same as
                # before. See open Stage 5 question: should this become
                # part of processed_with_gaps, or its own field. Not
                # decided yet, so not wired into the return dict below.
                print(f"[DOCLING] {len(page_errors)} page-level error(s) "
                      f"for doc_id={doc_id}: {page_errors}")
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

        from app.services.retrieval.keyword_search import keyword_manager
        await keyword_manager.index_document(user_id, chunks)

        print(f"Generating embeddings for {len(chunks)} chunks...")
        chunk_texts = [c["text"] for c in chunks]
        embed_result = await self.embedder.embed_batch(chunk_texts)

        successful_chunks = []
        successful_embeddings = []
        embed_failed_indices = set(embed_result["failed_indices"])

        for chunk, embedding in zip(chunks, embed_result["embeddings"]):
            if embedding is not None:
                successful_chunks.append(chunk)
                successful_embeddings.append(embedding)

        if not successful_chunks:
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

        # FIXED: store_vectors now returns failed_chunk_indices already in
        # terms of ORIGINAL chunk_index (it uses chunk["chunk_index"]
        # internally now, not loop position). Re-mapping via
        # successful_chunks[i]["chunk_index"] here would be WRONG a second
        # time over -- that re-mapping was only correct when
        # store_vectors's indices were positions-within-successful_chunks.
        # Just union the two sets of real indices directly now.
        qdrant_failed_original_indices = store_result["failed_chunk_indices"]

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
            "docling_page_errors": page_errors,  # not yet used by the route/schema;
                                                   # exposed here so it's available
                                                   # once Stage 5's open question is settled
        }
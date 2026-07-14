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

        page_errors = []       # only ever populated for the Docling/PDF path
        fallback_pages = {}    # 0-indexed page -> plain text, recovered via
                                # PyMuPDF when Docling failed even at
                                # single-page isolation (see docling_parser.py)
        pages_fully_lost = []  # 0-indexed pages where NEITHER Docling nor
                                # the PyMuPDF fallback produced anything

        if use_docling:
            # DOCLING_BATCH_SIZE: configurable so it can be tuned per
            # environment without a code change if memory conditions
            # differ (e.g. smaller containers may need batch_size=3).
            batch_size = settings.DOCLING_BATCH_SIZE
            print(f"Parsing PDF via Docling in batches of {batch_size} "
                  f"pages (bounds peak memory per Docling run)...")

            # parse_in_batches splits the PDF into page-range mini-PDFs and
            # runs DocumentConverter.convert() on each SEPARATELY, so a
            # bad_alloc on one batch only costs that batch's pages, not the
            # whole document. Returns {"documents": [...], "status",
            # "page_errors", "fallback_pages", "pages_fully_lost"} --
            # "documents" (plural) because DoclingDocument objects from
            # separate batches are NOT merged (no supported Docling API for
            # that); each is chunked independently below and the resulting
            # CHUNKS are what get concatenated instead.
            #
            # 2026-07-14 fix (Bug 1): parse_in_batches now retries a failed
            # batch at single-page granularity, and falls back to PyMuPDF
            # plain-text extraction for any page that still fails alone --
            # see docling_parser.py for the full recovery chain. Only pages
            # that fail BOTH recovery stages end up in pages_fully_lost.
            parse_result = DoclingPDFParser.parse_in_batches(file_path, batch_size=batch_size)
            batch_docs = parse_result["documents"]
            page_errors = parse_result["page_errors"]
            fallback_pages = parse_result.get("fallback_pages", {})
            pages_fully_lost = parse_result.get("pages_fully_lost", [])

            if page_errors:
                print(f"[DOCLING] {len(page_errors)} page-level issue(s) "
                      f"for doc_id={doc_id}: {page_errors}")

            # Metadata extraction only needs the START of the document --
            # use the first successful batch, not all of them concatenated
            # (that would blow past to_plain_text's 2500-char window for
            # no benefit).
            #
            # 2026-07-14 fix: batch_docs (Docling successes) can now be
            # EMPTY even on a document that isn't a total loss, if every
            # page that survived did so only via the PyMuPDF fallback
            # (structure-aware parsing failed everywhere, plain text
            # didn't). Previously batch_docs[0] would raise IndexError in
            # that case. Fall back to the first recovered fallback page's
            # plain text for metadata extraction instead.
            if batch_docs:
                opening_text = DoclingPDFParser.to_plain_text(batch_docs[0]["document"])
            elif fallback_pages:
                first_fallback_page = sorted(fallback_pages.keys())[0]
                opening_text = fallback_pages[first_fallback_page][:2500]
            else:
                # parse_in_batches only returns successfully if SOMETHING
                # came back (Docling docs or fallback text) -- this branch
                # should be unreachable, but fail loudly instead of
                # silently proceeding with no metadata source if it ever
                # is reached.
                raise ValueError(
                    f"No usable content recovered for doc_id={doc_id} -- "
                    f"parse_in_batches returned no documents and no "
                    f"fallback_pages, which should not happen without it "
                    f"having already raised."
                )
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
            # Chunk each batch's DoclingDocument SEPARATELY (this is why we
            # didn't merge the documents themselves), then flatten into one
            # list. chunk_index gets reassigned globally right after this
            # block anyway, so per-batch chunks starting back at their own
            # index 0 internally is fine -- that gets overwritten below.
            # NOTE -- confirmed via docling_chunker.py: chunks are
            # {"text", "tokens"} only, no page field, so there's no
            # page-offset bug to fix here.
            #
            # KNOWN TRADEOFF (not yet resolved, flagging rather than
            # silently patching): DoclingChunker's current_heading and
            # prose_buffer are local to one chunk() call, and it treats
            # every section_header as a hard flush boundary -- that's the
            # fix for "section-boundary bleeding" described in its
            # docstring. Chunking each batch separately means a FORCED
            # flush_prose() at every batch_size-page boundary too, whether
            # or not a section actually ended there, and current_heading
            # resets to "" going into the next batch until that batch's own
            # first section_header appears. A section or table spanning a
            # batch boundary (e.g. batch_size=5, section runs page 4-6)
            # can end up split with a missing/wrong heading on the
            # continuation -- the same flavor of quality regression
            # eval_rag.py caught in the old chunker, just relocated from
            # token-count boundaries to batch-page boundaries. Doesn't
            # crash or drop chunks; worth revisiting (e.g. small page
            # overlap between batches) if evals show it matters in
            # practice.
            chunks = []
            for batch_doc in batch_docs:
                batch_chunks = self.docling_chunker.chunk(batch_doc["document"])
                chunks.extend(batch_chunks)

            # 2026-07-14 fix (Bug 1, fix option #4): pages that Docling
            # couldn't parse even in single-page isolation, but PyMuPDF
            # recovered as plain text, still need to become searchable
            # chunks -- otherwise recovering the text was pointless, it'd
            # never reach Qdrant/BM25 either. Reuse the same generic
            # TextChunker used for non-PDF documents, since these pages
            # have no DoclingDocument structure to chunk against. Tagged
            # with source/page so degraded-quality chunks are identifiable
            # later (e.g. if eval results show they retrieve worse than
            # Docling-parsed chunks, that's now visible rather than mixed
            # in indistinguishably).
            if fallback_pages:
                print(f"[DOCLING] Chunking {len(fallback_pages)} page(s) "
                      f"recovered via PyMuPDF fallback (structure lost, "
                      f"text preserved): pages "
                      f"{sorted(p + 1 for p in fallback_pages)}")
                for page_num, page_text in sorted(fallback_pages.items()):
                    fallback_chunks = self.chunker.chunk(page_text)
                    for c in fallback_chunks:
                        c["source"] = "pymupdf_fallback"
                        c["page"] = page_num + 1  # 1-indexed, human-facing
                    chunks.extend(fallback_chunks)

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

        # 2026-07-14 fix (Bug 1, fix option #3 -- post-ingestion sanity
        # check): previously docling_page_errors was collected but never
        # actually used to decide anything or surface a real warning --
        # a document could finish "successfully" while missing entire
        # sections and nothing downstream would know. pages_fully_lost is
        # now the explicit, human-facing signal for that: it's non-empty
        # ONLY when a page failed Docling, failed single-page retry, AND
        # failed the PyMuPDF fallback -- i.e. content that is genuinely,
        # unrecoverably missing from the index. Surfaced loudly here, and
        # returned in the response so the API/UI layer can choose to warn
        # the user (e.g. "this document was indexed with N page(s) missing").
        if pages_fully_lost:
            human_pages = sorted(p + 1 for p in pages_fully_lost)
            print(f"[INGESTION WARNING] doc_id={doc_id}: {len(human_pages)} "
                  f"page(s) could not be recovered by Docling OR the "
                  f"PyMuPDF fallback -- this content is NOT in the index: "
                  f"pages {human_pages}")

        return {
            "chunk_count": total_chunks,
            "chunks_stored": total_succeeded,
            "chunks_failed": total_failed,
            "failed_chunk_indices": all_failed_indices,
            "avg_tokens": sum(c["tokens"] for c in chunks) // len(chunks),
            "total_tokens": sum(c["tokens"] for c in chunks),
            "metadata": metadata,
            "docling_page_errors": page_errors,
            "processed_with_gaps": bool(pages_fully_lost),
            "pages_fully_lost": sorted(p + 1 for p in pages_fully_lost),  # 1-indexed, human-facing
            "pages_recovered_via_fallback": sorted(p + 1 for p in fallback_pages),  # 1-indexed
        }
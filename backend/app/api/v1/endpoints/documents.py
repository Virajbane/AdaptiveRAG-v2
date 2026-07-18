from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
import os
import tempfile
import uuid
from datetime import datetime
from app.middleware.auth import get_current_user
from app.db.mongodb.queries import DocumentQueries
from app.db.mongodb.client import get_db
from app.db.qdrant.client import QdrantVectorDB
from app.services.retrieval.keyword_search import keyword_manager




qdrant = QdrantVectorDB()

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_TYPES = {"pdf", "docx", "txt", "csv"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Use system temp dir — works on both Windows and Linux
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "rag_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


async def _purge_document(doc_id: str, user_id: str, db, doc_queries: "DocumentQueries", storage_path: str | None):
    """
    Fully remove a document's Qdrant vectors, BM25 chunks, on-disk file
    (if present), and Mongo record. Shared by DELETE /{doc_id} and by
    upload_document's replace-on-reupload path, so both stay in sync
    instead of drifting into two slightly different cleanup routines.

    Each step wrapped independently -- a delete failing partway should
    never leave searchable content for a doc that's about to be
    superseded or removed.
    """
    try:
        await qdrant.delete_document_vectors(doc_id=doc_id, user_id=user_id)
    except Exception as e:
        print(f"⚠️ Failed to delete Qdrant vectors for {doc_id}: {e}")

    try:
        await keyword_manager.remove_document(user_id=user_id, doc_id=doc_id)
    except Exception as e:
        print(f"⚠️ Failed to remove BM25 chunks for {doc_id}: {e}")

    if storage_path and os.path.exists(storage_path):
        try:
            os.remove(storage_path)
        except Exception as e:
            print(f"⚠️ Failed to delete file {storage_path}: {e}")

    await doc_queries.delete_document(doc_id, user_id)


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
    background_tasks: BackgroundTasks = None,
    db=Depends(get_db)
):
    """
    Upload a document for processing. Accepts: PDF, DOCX, TXT, CSV. Max size: 50MB.

    REPLACE-ON-REUPLOAD (2026-07-14): uploading a file with the SAME
    filename as an existing document for this user now REPLACES it --
    old Qdrant vectors, BM25 chunks, temp file, and Mongo record are
    purged first, then this upload proceeds as a normal fresh document.

    Why: without this, a same-name re-upload got a brand new doc_id,
    so none of the existing "replace, don't duplicate" logic (which is
    keyed on doc_id matching -- see keyword_search.py's 2026-07-04 fix
    and QdrantVectorDB's point IDs) ever applied. The old and new
    copies both stayed indexed side by side under different doc_ids,
    silently doubling up retrieval results for what the user experienced
    as "the same document."
    """

    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is required")

    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type .{file_ext} not allowed. Allowed: {', '.join(ALLOWED_TYPES)}"
        )

    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max size: 50MB")

    doc_queries = DocumentQueries(db)

    # Replace-on-reupload: purge any existing document with this exact
    # filename for this user before creating the new one.
    existing = await db.documents.find_one({"user_id": user_id, "filename": file.filename})
    if existing:
        old_doc_id = str(existing["_id"])
        print(f"↻ Replacing existing document '{file.filename}' (old doc_id={old_doc_id}) with new upload")
        await _purge_document(
            doc_id=old_doc_id,
            user_id=user_id,
            db=db,
            doc_queries=doc_queries,
            storage_path=existing.get("storage_path"),
        )

    # RACE FIX (2026-07-14): was f"{user_id}_{file.filename}" -- collided
    # whenever two uploads (same OR different intent, e.g. a double-click,
    # two browser tabs, or a re-upload landing while the prior one is
    # still mid-Docling-convert) shared a filename, since the path was
    # deterministic from filename alone. A uuid4 component makes every
    # upload's temp path unique regardless of filename or timing, so two
    # in-flight uploads can never overwrite each other's file on disk.
    unique_suffix = uuid.uuid4().hex[:12]
    temp_file_path = os.path.join(UPLOAD_DIR, f"{user_id}_{unique_suffix}_{file.filename}")
    with open(temp_file_path, "wb") as f:
        f.write(file_content)

    # Create document record in MongoDB
    doc_id = await doc_queries.create_document(
        user_id=user_id,
        filename=file.filename,
        file_type=file_ext,
        file_size_bytes=len(file_content),
        storage_path=temp_file_path,
    )

    # Pass the live db object directly into the background task — NOT re-imported
    background_tasks.add_task(
        process_document,
        doc_id=doc_id,
        user_id=user_id,
        file_path=temp_file_path,
        file_type=file_ext,
        db=db,
    )

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "status": "processing",
        "message": "Document queued for processing",
    }


async def process_document(doc_id: str, user_id: str, file_path: str, file_type: str, db):
    """Background task: Process uploaded document."""
    from app.services.document.processor import DocumentProcessor
    from app.services.llm.provider import LLMProvider

    doc_queries = DocumentQueries(db)

    try:
        await doc_queries.mark_processing_started(doc_id)
    except Exception as e:
        print(f"⚠️ Failed to stamp started_at for {doc_id}: {e}")

    try:
        llm = LLMProvider()
        processor = DocumentProcessor(db, llm)
        result = await processor.process(file_path, file_type, user_id, doc_id)

        has_chunk_gaps = result["chunks_failed"] > 0
        has_page_errors = bool(result["docling_page_errors"])

        if not has_chunk_gaps and not has_page_errors:
            final_status = "processed"
        elif result["chunks_stored"] > 0:
            final_status = "processed_with_gaps"
        else:
            final_status = "failed"

        await doc_queries.update_document_status(
            doc_id=doc_id,
            status=final_status,
            chunks_info={
                "count": result["chunk_count"],
                "average_tokens": result["avg_tokens"],
                "total_tokens": result["total_tokens"],
                "overlap_tokens": 50,
                "stored_count": result["chunks_stored"],
            },
            chunks_failed=result["chunks_failed"],
            failed_chunk_indices=result["failed_chunk_indices"],
            docling_page_errors=result["docling_page_errors"],
            pages_fully_lost=result["pages_fully_lost"],   # NEW -- was computed
                                                            # by processor.process()
                                                            # but never persisted,
                                                            # so the ingestion gate
                                                            # had nothing real to
                                                            # query. See eval
                                                            # bug report §2.1.
            user_id=user_id,
        )

        log_msg = f"✅ Document {doc_id} processed: {result['chunks_stored']}/{result['chunk_count']} chunks stored"
        if result["chunks_failed"]:
            log_msg += f", {result['chunks_failed']} chunks failed"
        if result["docling_page_errors"]:
            log_msg += f", {len(result['docling_page_errors'])} page-level parse error(s)"
        print(log_msg)

        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await doc_queries.update_document_status(
                doc_id=doc_id,
                status="failed",
                error=str(e),
            )
        except Exception as inner:
            print(f"Failed to update document status: {inner}")
        print(f"❌ Error processing document {doc_id}: {e}")


@router.get("")
async def list_documents(
    user_id: str = Depends(get_current_user),
    db=Depends(get_db)
):
    """List all documents for the current user."""
    doc_queries = DocumentQueries(db)
    docs = await doc_queries.list_documents(user_id)

    documents = []
    for doc in docs:
        documents.append({
            "_id": str(doc["_id"]),
            "filename": doc["filename"],
            "file_type": doc["file_type"],
            "file_size_bytes": doc["file_size_bytes"],
            "status": doc["status"],
            "chunks": doc.get("chunks", {"count": 0}),
            "chunks_failed": doc.get("chunks_failed", 0),                    # NEW
            "failed_chunk_indices": doc.get("failed_chunk_indices", []),     # NEW
            "docling_page_errors": doc.get("docling_page_errors", []),       # NEW
            "processing_error": doc.get("processing_error"),                 # NEW —
                                                                               # was already
                                                                               # being saved
                                                                               # on "failed"
                                                                               # status, just
                                                                               # never returned
            "created_at": doc["created_at"],
        })

    return {"documents": documents}


@router.post("/{doc_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_document(
    doc_id: str,
    user_id: str = Depends(get_current_user),
    background_tasks: BackgroundTasks = None,
    db=Depends(get_db)
):
    """
    Re-queue a document that's either:
      - status="failed" (processing raised an exception last time), or
      - status="processing" but stale (started_at is old enough that the
        background task almost certainly died mid-flight — see
        DocumentQueries.get_stale_processing_documents)

    Reuses the original file from storage_path, which is only ever
    deleted on a SUCCESSFUL process_document() run — so a failed or
    crashed job's file is still on disk to reprocess from, no
    re-upload needed.

    NOTE: no BM25/Qdrant cleanup needed here before reprocessing — the
    retried job reuses the SAME doc_id, and DocumentProcessor.process()
    -> keyword_manager.index_document() now replaces any existing
    chunks for that doc_id rather than appending (see keyword_search.py
    2026-07-04 fix), so this path was already safe once that fix
    landed.
    """
    doc_queries = DocumentQueries(db)
    doc = await doc_queries.get_document(doc_id, user_id)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc["status"] not in ("failed", "processing", "processed_with_gaps"):
        raise HTTPException(
            status_code=400,
            detail=f"Document status is '{doc['status']}' — only 'failed', "
                   f"'processed_with_gaps', or stuck 'processing' documents can be retried.",
        )

    if doc["status"] == "processing":
        # Only allow retrying a "processing" doc if it actually looks
        # stale -- a processed_with_gaps or failed doc has no such
        # ambiguity (it's definitely not still running), so this check
        # stays scoped to "processing" only.
        stale_docs = await doc_queries.get_stale_processing_documents(user_id, timeout_minutes=10)
        stale_ids = {str(d["_id"]) for d in stale_docs}
        if doc_id not in stale_ids:
            raise HTTPException(
                status_code=409,
                detail="Document is still actively processing — not stale enough to retry yet.",
            )

    file_path = doc["storage_path"]
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=410,
            detail="Original uploaded file is no longer available — please re-upload.",
        )

    background_tasks.add_task(
        process_document,
        doc_id=doc_id,
        user_id=user_id,
        file_path=file_path,
        file_type=doc["file_type"],
        db=db,
    )

    return {
        "doc_id": doc_id,
        "filename": doc["filename"],
        "status": "processing",
        "message": "Document re-queued for processing",
    }

@router.delete("/{doc_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    doc_id: str,
    user_id: str = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Delete a document: removes the Qdrant vectors, the BM25 keyword-
    index chunks, the Mongo record, and the on-disk file if it still
    exists (e.g. a failed job that was never cleaned up — see
    process_document's note on why failed files are kept around for
    /retry).

    2026-07-04 fix: this previously cleaned up Qdrant and Mongo but
    never touched KeywordSearchManager's in-memory BM25 index at all —
    the same class of bug just fixed for re-ingestion (see
    keyword_search.py), except unpatched on this path. A deleted
    document's chunks stayed in BM25 indefinitely, would keep surfacing
    in hybrid search results, and would resolve to "Unknown document"
    once their doc_id no longer matched a Mongo record (identical
    symptom/root cause to the re-ingestion duplication bug).

    2026-07-14: cleanup steps now shared with upload_document's
    replace-on-reupload path via _purge_document(), so both stay in
    sync instead of two near-duplicate cleanup routines drifting apart.
    """
    doc_queries = DocumentQueries(db)
    doc = await doc_queries.get_document(doc_id, user_id)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    await _purge_document(
        doc_id=doc_id,
        user_id=user_id,
        db=db,
        doc_queries=doc_queries,
        storage_path=doc.get("storage_path"),
    )

    return {"doc_id": doc_id, "status": "deleted"}
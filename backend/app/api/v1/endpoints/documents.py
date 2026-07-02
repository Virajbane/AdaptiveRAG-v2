from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
import os
import tempfile
from datetime import datetime
from app.middleware.auth import get_current_user
from app.db.mongodb.queries import DocumentQueries
from app.db.mongodb.client import get_db
from app.db.qdrant.client import QdrantVectorDB

qdrant = QdrantVectorDB()

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_TYPES = {"pdf", "docx", "txt", "csv"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Use system temp dir — works on both Windows and Linux
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "rag_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
    background_tasks: BackgroundTasks = None,
    db=Depends(get_db)
):
    """Upload a document for processing. Accepts: PDF, DOCX, TXT, CSV. Max size: 50MB."""

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

    # Save file to temp dir (cross-platform)
    temp_file_path = os.path.join(UPLOAD_DIR, f"{user_id}_{file.filename}")
    with open(temp_file_path, "wb") as f:
        f.write(file_content)

    # Create document record in MongoDB
    doc_queries = DocumentQueries(db)
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

    doc_queries = DocumentQueries(db)

    # Stamp started_at the moment real work begins — this is what stale-job
    # detection (get_stale_processing_documents) keys off of. If this
    # background task dies before finishing (process restart/crash), the
    # document stays at status="processing" with a started_at that quickly
    # falls behind "now" by more than processing normally takes — making
    # the stuck job detectable instead of indistinguishable from a job
    # that's still legitimately in progress.
    try:
        await doc_queries.mark_processing_started(doc_id)
    except Exception as e:
        print(f"⚠️ Failed to stamp started_at for {doc_id}: {e}")

    try:
        processor = DocumentProcessor(db)
        result = await processor.process(file_path, file_type, user_id, doc_id)

        await doc_queries.update_document_status(
            doc_id=doc_id,
            status="processed",
            chunks_info={
                "count": result["chunk_count"],
                "average_tokens": result["avg_tokens"],
                "total_tokens": result["total_tokens"],
                "overlap_tokens": 50,
            },
        )

        print(f"✅ Document {doc_id} processed: {result['chunk_count']} chunks")

        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as e:
        import traceback
        traceback.print_exc()  # Print full stack trace so we can see what's failing
        try:
            await doc_queries.update_document_status(
                doc_id=doc_id,
                status="failed",
                error=str(e),
            )
        except Exception as inner:
            print(f"Failed to update document status: {inner}")
        print(f"❌ Error processing document {doc_id}: {e}")
        # NOTE: file is deliberately NOT deleted on failure — this is what
        # lets the /retry endpoint reprocess without asking the user to
        # re-upload. It's only ever cleaned up on success (above) or by
        # a future cleanup job for genuinely abandoned failed uploads.


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
    """
    doc_queries = DocumentQueries(db)
    doc = await doc_queries.get_document(doc_id, user_id)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc["status"] not in ("failed", "processing"):
        raise HTTPException(
            status_code=400,
            detail=f"Document status is '{doc['status']}' — only 'failed' or "
                   f"stuck 'processing' documents can be retried.",
        )

    if doc["status"] == "processing":
        # Only allow retrying a "processing" doc if it actually looks
        # stale — otherwise this could race a job that's genuinely still
        # running and kick off a duplicate processing run for the same
        # document.
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
    Delete a document: removes the Qdrant vectors, the Mongo record, and
    the on-disk file if it still exists (e.g. a failed job that was never
    cleaned up — see process_document's note on why failed files are kept
    around for /retry).
    """
    doc_queries = DocumentQueries(db)
    doc = await doc_queries.get_document(doc_id, user_id)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove Qdrant vectors first — a delete failing midway should never
    # leave searchable vectors for a doc Mongo no longer lists.
    try:
        await qdrant.delete_document_vectors(doc_id=doc_id, user_id=user_id)
    except Exception as e:
        print(f"⚠️ Failed to delete Qdrant vectors for {doc_id}: {e}")

    storage_path = doc.get("storage_path")
    if storage_path and os.path.exists(storage_path):
        try:
            os.remove(storage_path)
        except Exception as e:
            print(f"⚠️ Failed to delete file {storage_path}: {e}")

    deleted = await doc_queries.delete_document(doc_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"doc_id": doc_id, "status": "deleted"}
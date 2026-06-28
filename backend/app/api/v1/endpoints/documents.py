from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
import os
import tempfile
from datetime import datetime
from app.middleware.auth import get_current_user
from app.db.mongodb.queries import DocumentQueries
from app.db.mongodb.client import get_db

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
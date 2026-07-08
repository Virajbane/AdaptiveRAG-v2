from dotenv import load_dotenv
load_dotenv()
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.mongodb.client import connect_to_mongo, close_mongo_connection, get_db
from app.services.memory.redis_client import redis_client
from app.services.memory.long_term import LongTermMemory
from app.services.memory.manager import MemoryManager
import app.services.memory.manager as mm_module
from app.services.tools.web_search import init_web_search
from app.config.settings import settings
from app.utils.health_checks import get_system_health
from app.middleware.logging import LoggingMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.middleware.audit_log import AuditLoggingMiddleware
from app.middleware.timeout import TimeoutMiddleware


app = FastAPI(
    title="RAG 2.0 System API",
    description="Enterprise-grade Adaptive RAG System",
    version="1.0.0"
)

allowed_origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(TimeoutMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(LoggingMiddleware)


@app.get("/health/detailed")
async def detailed_health_check():
    return await get_system_health()


@app.on_event("startup")
async def startup():
    print("\n🚀 Starting RAG 2.0 System...")

    await connect_to_mongo()
    print("✅ MongoDB connected")

    await redis_client.connect()
    print("✅ Redis connected")

    init_web_search(settings.TAVILY_API_KEY)
    print("✅ Tools initialized")

    db = await get_db()
    print(f"✅ Got database: {db}")

    if db is not None:
        try:
            long_term_mem = LongTermMemory(db)
            memory_mgr = MemoryManager(long_term_mem)
            mm_module.memory_manager = memory_mgr
            mm_module.long_term_memory = long_term_mem
            print("✅ Memory system initialized")
        except Exception as e:
            print(f"❌ Memory init failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ db is still None after get_db()!")

    # Rebuild BM25 keyword indexes from Qdrant on every startup
    await _rebuild_bm25_indexes()


async def _rebuild_bm25_indexes():
    """Rebuild in-memory BM25 indexes from Qdrant so keyword search
    works immediately after restart without re-uploading documents."""
    try:
        from app.db.qdrant.client import QdrantVectorDB
        from app.services.retrieval.keyword_search import keyword_manager
        import asyncio

        qdrant = QdrantVectorDB()
        all_points = []
        offset = None
        loop = asyncio.get_event_loop()

        while True:
            result = await loop.run_in_executor(
                None,
                lambda o=offset: qdrant.client.scroll(
                    collection_name="documents_embeddings",
                    limit=100,
                    offset=o,
                    with_payload=True,
                    with_vectors=False,
                )
            )
            points, next_offset = result
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        if not all_points:
            print("ℹ️  No vectors in Qdrant — BM25 index empty (upload docs first)")
            return

        user_chunks: dict = {}
        for point in all_points:
            uid = point.payload.get("user_id")
            if not uid:
                continue
            chunk_text = point.payload.get("chunk_text", "")
            if not chunk_text:
                continue
            user_chunks.setdefault(uid, []).append({
                "doc_id": point.payload.get("doc_id", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "text": chunk_text,
            })

        for uid, chunks in user_chunks.items():
            await keyword_manager.rebuild_from_chunks(uid, chunks)
            print(f"✅ BM25 rebuilt for user {uid[:8]}...: {len(chunks)} chunks")

    except Exception as e:
        print(f"⚠️  BM25 rebuild failed (non-fatal): {e}")
        import traceback
        traceback.print_exc()


from app.api.v1.router import router
app.include_router(router)


@app.on_event("shutdown")
async def shutdown():
    await close_mongo_connection()
    await redis_client.disconnect()
    print("✅ Services disconnected\n")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "RAG 2.0 System API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
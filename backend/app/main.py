from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.mongodb.client import connect_to_mongo, close_mongo_connection, get_db
from app.services.memory.redis_client import redis_client
from app.services.memory.long_term import LongTermMemory
from app.services.memory.manager import MemoryManager
import app.services.memory.manager as mm_module
from app.services.tools.web_search import init_web_search
from app.config.settings import settings
    
   

app = FastAPI(
    title="RAG 2.0 System API",
    description="Enterprise-grade Adaptive RAG System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    print("\n🚀 Starting RAG 2.0 System...")
    
    await connect_to_mongo()
    print("✅ MongoDB connected")
    
    await redis_client.connect()
    print("✅ Redis connected")

    init_web_search(settings.TAVILY_API_KEY)
    print("✅ Tools initialized")
    
    # Get db AFTER connecting
    db = await get_db()
    print(f"✅ Got database: {db}")
    
    # Initialize memory
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
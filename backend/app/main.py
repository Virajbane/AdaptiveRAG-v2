from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import router
from app.db.mongodb.client import connect_to_mongo, close_mongo_connection
# Add this import
from app.api.v1.endpoints.documents import router as documents_router

# Add this line with the other routers


app = FastAPI(
    title="RAG 2.0 System API",
    description="Enterprise-grade Adaptive RAG System",
    version="1.0.0"
)

# CORS must come BEFORE routers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(documents_router)

@app.on_event("startup")
async def startup():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown():
    await close_mongo_connection()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "services": {"api": "up"}}

@app.get("/")
async def root():
    return {"message": "RAG 2.0 System API", "version": "1.0.0", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
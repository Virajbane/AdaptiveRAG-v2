from motor.motor_asyncio import AsyncIOMotorClient
from app.config.settings import settings

client: AsyncIOMotorClient = None
db = None

async def connect_to_mongo():
    global client, db

    client = AsyncIOMotorClient(settings.MONGODB_URL)

    db = client["rag_db"]

    print("Connected to MongoDB: rag_db")


async def close_mongo_connection():
    global client

    if client:
        client.close()
        print("MongoDB connection closed")


async def get_db():
    return db
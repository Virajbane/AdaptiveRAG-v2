from motor.motor_asyncio import AsyncIOMotorClient

client: AsyncIOMotorClient = None
db = None

async def connect_to_mongo():
    global client, db
    client = AsyncIOMotorClient(
        "mongodb://admin:password123@localhost:27017/rag_db?authSource=admin"
    )
    db = client["rag_db"]
    print("Connected to MongoDB: rag_db")

async def close_mongo_connection():
    global client
    if client:
        client.close()
        print("MongoDB connection closed")
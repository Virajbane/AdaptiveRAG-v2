import asyncio
from app.db.mongodb.client import connect_to_mongo, get_db

async def check():
    await connect_to_mongo()
    db = await get_db()
    cursor = db.documents.find({}, {"filename": 1, "metadata": 1})
    async for doc in cursor:
        print(doc)

if __name__ == "__main__":
    asyncio.run(check())
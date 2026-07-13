"""
find_doc_owner.py

Run this from your backend folder (same place you run eval_rag.py) with
your venv activated:

    python find_doc_owner.py

Paste your REAL MongoDB connection string (with username/password, the
same one your backend .env file uses) into MONGO_URI below before running.
"""

from pymongo import MongoClient
from bson import ObjectId

# TODO: paste your real connection string here, e.g.
# "mongodb://someuser:somepassword@localhost:27017/rag_db?authSource=admin"
MONGO_URI = "mongodb://admin:DRqT6GcPmh--Wks4WLau3w@localhost:27017/rag_db?authSource=admin"

DOC_ID = "6a549f0f62642be451689205"

client = MongoClient(MONGO_URI)
db = client["rag_db"]

print("Collections in rag_db:", db.list_collection_names())

doc = db["documents"].find_one({"_id": ObjectId(DOC_ID)})

if doc:
    print("\nFOUND document:")
    print("  user_id:", doc.get("user_id"))
    print("  filename:", doc.get("filename", doc.get("title", "n/a")))
else:
    print(f"\nNo document found with _id={DOC_ID} in 'documents' collection.")
    print("Try checking the collection name above is right, or that DOC_ID matches exactly.")
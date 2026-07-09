# diag_qdrant.py
from app.config.settings import settings  # adjust import path if your config file lives elsewhere
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

# 1. Total points in the collection
count = client.count(collection_name="documents_embeddings", exact=True)
print("Total points:", count)

# 2. Sample raw payloads to see ACTUAL user_id field/value stored
points, _ = client.scroll(
    collection_name="documents_embeddings",
    limit=10,
    with_payload=True,
    with_vectors=False,
)

# add to diag_qdrant.py, replace the sample-print loop with:
points, _ = client.scroll(
    collection_name="documents_embeddings",
    limit=1,
    with_payload=True,
    with_vectors=False,
)
import json
print(json.dumps(points[0].payload, indent=2, default=str))
print("\n-- Sample payloads --")
for p in points:
    print(p.payload.get("user_id"), "|", p.payload.get("document_id"), "|", str(p.payload.get("text", ""))[:60])

# 3. Check the specific doc from the golden set
filtered, _ = client.scroll(
    collection_name="documents_embeddings",
    scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value="6a4f3ceb757ca528ed374b2d"))]),
    limit=5,
    with_payload=True,
)
print(f"\n-- Points for document_id 6a4f3ceb757ca528ed374b2d: {len(filtered)} --")
for p in filtered:
    print(p.payload.get("user_id"))
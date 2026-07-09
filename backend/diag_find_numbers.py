# diag_find_numbers.py
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.config.settings import settings  # adjust to your confirmed working import

client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

# scroll through all points for this doc and grep locally for target substrings
points, _ = client.scroll(
    collection_name="documents_embeddings",
    scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value="6a4f3ceb757ca528ed374b2d"))]),
    limit=300,
    with_payload=True,
)
print(f"Total chunks for this doc: {len(points)}")

targets = ["39.4", "140K", "140,000", "140 K"]
for p in points:
    text = p.payload.get("chunk_text", "")
    for t in targets:
        if t in text:
            print(f"\n--- FOUND '{t}' in chunk_index={p.payload.get('chunk_index')} ---")
            print(text[:300])
            
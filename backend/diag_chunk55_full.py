# diag_chunk55_full.py
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.config.settings import settings

client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

points, _ = client.scroll(
    collection_name="documents_embeddings",
    scroll_filter=Filter(must=[
        FieldCondition(key="doc_id", match=MatchValue(value="6a4f3ceb757ca528ed374b2d")),
        FieldCondition(key="chunk_index", match=MatchValue(value=55)),
    ]),
    limit=1,
    with_payload=True,
)
p = points[0]
text = p.payload.get("chunk_text", "")
print(f"Chunk length (chars): {len(text)}")
print(f"Token count (payload): {p.payload.get('tokens')}")
print("--- FULL TEXT ---")
print(text)
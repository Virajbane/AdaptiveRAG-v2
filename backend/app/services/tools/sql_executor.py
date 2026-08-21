from typing import Dict, List, Optional, Tuple
from app.db.mongodb import client as mongo_client

class SQLExecutorTool:
    """
    Execute SQL-like queries on databases

    Currently supports MongoDB aggregation

    2026-08-22 FIX: Previously did `from app.db.mongodb.client import db`,
    which snapshots `db` (None) at import time -- before
    connect_to_mongo() runs at FastAPI startup and reassigns client.py's
    OWN module-level `db`. That reassignment never reached this module's
    local name, so every query hit `None[collection]` ->
    'NoneType' object is not subscriptable, on every call, deterministically
    (not a race condition -- a permanently stale reference). Fixed by
    importing the client MODULE and reading `mongo_client.db` at call
    time, the same pattern already used correctly for web_search_tool in
    registry.py's get_tools().

    2026-08-22 IMPROVEMENT: Added "users" to the collection whitelist so
    database questions about user counts/lists can be answered. find on
    "users" additionally enforces a safe field projection server-side
    (never returns password_hash) regardless of what projection is
    requested by the caller -- this is a second, independent guard on
    top of tool_agent.py only ever requesting safe fields, so a bug or
    future caller change in tool_agent.py can't leak credentials.
    """

    ALLOWED_COLLECTIONS = ["documents", "chat_sessions", "memory_long_term", "users"]

    # Fields that may NEVER be returned, regardless of what projection
    # a caller requests. Enforced last, after any caller-supplied
    # projection, so it can't be bypassed.
    _FORBIDDEN_FIELDS: Dict[str, Tuple[str, ...]] = {
        "users": ("password_hash",),
    }

    def _sanitize_projection(self, collection: str, projection: Optional[Dict]) -> Optional[Dict]:
        forbidden = self._FORBIDDEN_FIELDS.get(collection)
        if not forbidden:
            return projection
        if not projection:
            # No projection requested at all -- Mongo would return every
            # field, including forbidden ones. Explicitly exclude them.
            return {f: 0 for f in forbidden}
        # A projection was requested. If it's an inclusion projection
        # (any value == 1) forbidden fields are already excluded by
        # omission -- but strip them defensively in case one snuck in.
        sanitized = dict(projection)
        for f in forbidden:
            sanitized.pop(f, None)
            # If this was an exclusion-style projection (values == 0),
            # make sure the forbidden field is excluded too.
            if any(v == 0 for v in projection.values()):
                sanitized[f] = 0
        return sanitized

    async def execute_query(
        self,
        collection: str,
        query_type: str,  # "find", "count", "aggregate"
        query: Dict = None,
        projection: Dict = None,
        sort: List = None,
        **kwargs
    ) -> Dict:
        """
        Execute database query safely

        Args:
            collection: Collection name
            query_type: Type of query
            query: Query parameters
            projection: Optional field projection (find only). Sanitized
                against _FORBIDDEN_FIELDS before use.
            sort: Optional list of (field, direction) tuples (find only)

        Returns:
            Query results or error
        """

        if collection not in self.ALLOWED_COLLECTIONS:
            return {
                "error": f"Collection '{collection}' not allowed",
                "results": []
            }

        db = mongo_client.db
        if db is None:
            return {
                "error": "Database connection not available",
                "results": []
            }

        try:
            coll = db[collection]

            if query_type == "find":
                query = query or {}
                limit = kwargs.get("limit", 10)
                safe_projection = self._sanitize_projection(collection, projection)
                cursor = coll.find(query, safe_projection) if safe_projection else coll.find(query)
                if sort:
                    cursor = cursor.sort(sort)
                results = await cursor.limit(limit).to_list(length=limit)
                return {
                    "results": results,
                    "count": len(results),
                    "error": None
                }

            elif query_type == "count":
                query = query or {}
                count = await coll.count_documents(query)
                return {
                    "results": {"count": count},
                    "error": None
                }

            elif query_type == "aggregate":
                pipeline = query or []
                results = await coll.aggregate(pipeline).to_list(length=None)
                return {
                    "results": results,
                    "count": len(results),
                    "error": None
                }

            else:
                return {
                    "error": f"Unknown query type: {query_type}",
                    "results": []
                }

        except Exception as e:
            return {
                "error": str(e),
                "results": []
            }

# Global instance
sql_executor_tool = SQLExecutorTool()
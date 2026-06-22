from typing import Dict, List
from app.db.mongodb.client import db

class SQLExecutorTool:
    """
    Execute SQL-like queries on databases
    
    Currently supports MongoDB aggregation
    """
    
    async def execute_query(
        self,
        collection: str,
        query_type: str,  # "find", "count", "aggregate"
        query: Dict = None,
        **kwargs
    ) -> Dict:
        """
        Execute database query safely
        
        Args:
            collection: Collection name
            query_type: Type of query
            query: Query parameters
        
        Returns:
            Query results or error
        """
        
        # Whitelist allowed collections
        ALLOWED_COLLECTIONS = ["documents", "chat_sessions", "memory_long_term"]
        
        if collection not in ALLOWED_COLLECTIONS:
            return {
                "error": f"Collection '{collection}' not allowed",
                "results": []
            }
        
        try:
            coll = db[collection]
            
            if query_type == "find":
                query = query or {}
                limit = kwargs.get("limit", 10)
                results = await coll.find(query).limit(limit).to_list(length=limit)
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
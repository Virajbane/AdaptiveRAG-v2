from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.services.retrieval.hybrid_search import HybridSearchEngine
from app.services.retrieval.document_resolver import resolve_document_filter
from bson import ObjectId
from bson.errors import InvalidId
import time

class RetrieverAgent(BaseAgent):
    """
    Retriever Agent: Searches documents

    - Resolves whether the question names a specific uploaded document
    - Performs hybrid search (vector + keyword), scoped to that document
      if one was confidently resolved
    - Retrieves top documents
    - Attaches each chunk's source filename (one batched Mongo lookup)
      so downstream AnswerAgent / the frontend source cards can display
      the real document name instead of a generic "Source N" fallback
    - Passes context to other agents

    NOTE: Runs in PARALLEL with PlannerAgent, so it always searches.
    The orchestrator decides whether to use the results based on
    planner's sources_needed AFTER both complete.
    """

    def __init__(self, llm=None, db=None):
        super().__init__(llm)
        self.db = db   # Mongo handle, needed to look up filenames for
                        # document-scoped retrieval AND for the filename
                        # enrichment step below. May be None if not wired
                        # through — both degrade gracefully.

    async def _attach_filenames(self, results: list[dict]) -> None:
        """
        Batch-look up filenames for every unique doc_id in the result set
        and attach them in place as result['filename']. One query total,
        regardless of how many chunks were retrieved.

        Mutates `results` in place; no-ops gracefully if self.db is
        unavailable or no chunks were returned.
        """
        if self.db is None or not results:
            return

        unique_doc_ids = {doc["doc_id"] for doc in results}

        object_ids = []
        for doc_id in unique_doc_ids:
            try:
                object_ids.append(ObjectId(doc_id))
            except (InvalidId, TypeError):
                # doc_id wasn't a valid ObjectId string — skip rather than
                # crash retrieval over a metadata-enrichment problem.
                print(f"[RETRIEVER] Skipping filename lookup for malformed doc_id: {doc_id!r}")

        if not object_ids:
            return

        cursor = self.db.documents.find(
            {"_id": {"$in": object_ids}},
            {"filename": 1},
        )
        filename_map = {str(doc["_id"]): doc["filename"] async for doc in cursor}

        for doc in results:
            doc["filename"] = filename_map.get(doc["doc_id"], "Unknown document")

    async def _execute(self, state: AgentState) -> AgentState:
        """Search for relevant documents — always runs, results used conditionally."""

        try:
            start_time = time.time()
            question = state.rewritten_question or state.question

            document_id = await resolve_document_filter(question, state.user_id, self.db)

            search_engine = HybridSearchEngine()
            results = await search_engine.search(
                query=question,
                user_id=state.user_id,
                top_k=5,
                document_id=document_id,
            )

            await self._attach_filenames(results)

            state.retrieved_docs = results
            state.search_time_ms = (time.time() - start_time) * 1000

            print(f"[RETRIEVER] document_id filter: {document_id}")
            print(f"[RETRIEVER] Found {len(results)} documents")
            for i, doc in enumerate(results, 1):
                # rerank_score is the BGE cross-encoder score and is what
                # actually determines final ranking (when reranking ran).
                # combined_score is the RRF fusion score from BM25+vector -
                # it's rank-based and tightly clustered by design (RRF_K=60),
                # so it will always look "flat" at 2 decimal places even
                # when fusion is working correctly. Don't use it to judge
                # whether retrieval quality is good or bad.
                if 'rerank_score' in doc:
                    score_label = "rerank"
                    score_value = doc['rerank_score']
                else:
                    # Reranker was unavailable/skipped - fall back to RRF score,
                    # but flag it so it's obvious which scoring path was used.
                    score_label = "rrf (no rerank)"
                    score_value = doc.get('combined_score', 0.0)

                print(f"  {i}. [{score_label}] Score: {score_value:.4f} | {doc.get('filename', '?')} | {doc['text'][:60]}...")

        except Exception as e:
            import traceback
            traceback.print_exc()
            state.error = f"Retriever error: {str(e)}"

        return state
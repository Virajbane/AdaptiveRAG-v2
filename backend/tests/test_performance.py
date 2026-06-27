import pytest
import pytest_asyncio
import asyncio
import time
from app.agents.orchestrator import AgentOrchestrator
from app.services.memory.redis_client import redis_client
from unittest.mock import AsyncMock, patch, MagicMock


@pytest_asyncio.fixture(autouse=True)
async def ensure_redis_connected():
    """
    test_query_cache_hit and test_cache_expiry call query_cache directly,
    with no app startup involved - ASGITransport (used in conftest.py's
    `client` fixture) does not fire FastAPI's @app.on_event("startup")
    lifecycle, and these two tests don't even use the `client` fixture
    anyway, so redis_client.connect() never ran before them.

    Without this fixture, redis_client.redis stays None, and every cache
    get/set silently no-ops (by design, for graceful degradation in
    production if Redis is briefly unavailable) - which surfaced as
    `assert None is not None` rather than a clear connection error.

    This fixture is autouse and file-scoped (applies to every test in
    this file), but is a cheap no-op for tests that don't touch Redis,
    since it only calls connect() if there isn't already a live
    connection.
    """
    if not redis_client.redis:
        await redis_client.connect()
    yield


class TestPerformance:
    """Performance tests for agent system"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance"""
        return AgentOrchestrator()
    
    @pytest.mark.asyncio
    async def test_parallel_execution_faster_than_sequential(self, orchestrator):
        """Verify parallel execution is faster than sequential"""
        
        # Mock the agents to simulate work
        with patch('app.agents.orchestrator.PlannerAgent') as mock_planner_class, \
             patch('app.agents.orchestrator.RetrieverAgent') as mock_retriever_class, \
             patch('app.agents.orchestrator.AnswerAgent') as mock_answer_class:
            
            # Setup mock planner (30ms delay)
            mock_planner = AsyncMock()
            async def mock_planner_run(state):
                await asyncio.sleep(0.03)  # 30ms
                state.plan = "test plan"
                state.sources_needed = ["documents"]
                state.confidence = 0.8
                return state
            
            mock_planner.run = mock_planner_run
            mock_planner_class.return_value = mock_planner
            
            # Setup mock retriever (15ms delay)
            mock_retriever = AsyncMock()
            async def mock_retriever_run(state):
                await asyncio.sleep(0.015)  # 15ms
                state.retrieved_docs = [
                    {"text": "doc1", "combined_score": 0.9, "doc_id": "1", "chunk_index": 0}
                ]
                return state
            
            mock_retriever.run = mock_retriever_run
            mock_retriever_class.return_value = mock_retriever
            
            # Setup mock answer (10ms delay)
            mock_answer = AsyncMock()
            async def mock_answer_run(state):
                await asyncio.sleep(0.01)  # 10ms
                state.answer = "test answer"
                state.sources = []
                state.confidence_final = 0.75
                return state
            
            mock_answer.run = mock_answer_run
            mock_answer_class.return_value = mock_answer
            
            # Create new orchestrator with mocked agents
            orch = AgentOrchestrator()
            orch.planner = mock_planner
            orch.retriever = mock_retriever
            orch.answer = mock_answer
            
            # Mock memory manager
            with patch('app.agents.orchestrator.mm_module.memory_manager', None):
                # Time the execution
                start = time.time()
                result = await orch.process(
                    question="test question",
                    user_id="test_user"
                )
                elapsed = time.time() - start
                
                # With parallel execution: max(30ms, 15ms) + 10ms = ~40ms
                # Without parallel: 30ms + 15ms + 10ms = 55ms
                assert elapsed < 0.055, f"Too slow: {elapsed}s (expected <55ms)"
                assert result["answer"] == "test answer"
                print(f"✅ Parallel execution: {elapsed*1000:.0f}ms (expected <55ms)")
    
    @pytest.mark.asyncio
    async def test_query_cache_hit(self):
        """Test that caching returns results instantly"""
        from app.services.cache.query_cache import query_cache
        
        # Clear cache first
        await query_cache.clear()
        
        question = "What is machine learning?"
        user_id = "test_user"
        result = {
            "answer": "ML is...",
            "sources": [],
            "confidence": 0.9
        }
        
        # Store result
        await query_cache.set(question, user_id, result)
        
        # Retrieve (should be fast - Redis round-trip, not instant like
        # the old in-memory dict, so the assertion threshold below is
        # relaxed accordingly)
        start = time.time()
        cached = await query_cache.get(question, user_id)
        elapsed = time.time() - start
        
        assert cached is not None
        assert cached["answer"] == result["answer"]
        # NOTE: threshold relaxed from 1ms to 50ms after the Phase 14
        # Redis migration. The old in-memory dict lookup was sub-millisecond
        # by definition (no network involved). A Redis GET is a real network
        # round-trip (even to localhost/a sidecar container), so asserting
        # <1ms here would be testing infrastructure speed, not application
        # logic, and would be flaky depending on machine/CI runner load.
        assert elapsed < 0.05, f"Cache lookup too slow: {elapsed*1000:.2f}ms"
        print(f"✅ Cache hit: {elapsed*1000:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_cache_expiry(self):
        """Test that cache entries expire"""
        from app.services.cache.query_cache import QueryCache
        
        # Create cache with 1 second TTL
        cache = QueryCache(ttl_seconds=1)
        
        question = "test"
        user_id = "user1"
        result = {"answer": "test"}
        
        # Store result
        await cache.set(question, user_id, result)
        
        # Should be available immediately
        assert await cache.get(question, user_id) is not None
        
        # Wait for expiry (Redis enforces this natively via SETEX,
        # set inside QueryCache.set() - no manual expiry bookkeeping
        # in this class anymore)
        await asyncio.sleep(1.1)
        
        # Should be gone
        assert await cache.get(question, user_id) is None
        print("✅ Cache expiry works")
    
    @pytest.mark.asyncio
    async def test_profiler_tracking(self):
        """Test that profiler accurately tracks timing"""
        from app.utils.profiler import profiler
        
        profiler.reset()
        
        # Simulate some work
        profiler.start("test_step")
        await asyncio.sleep(0.05)  # 50ms
        elapsed = profiler.end("test_step")
        
        # Should be approximately 50ms
        assert 40 < elapsed < 70, f"Timing off: {elapsed}ms"
        
        # Check stats
        stats = profiler.get_stats()
        assert "test_step" in stats
        assert stats["test_step"]["count"] == 1
        print(f"✅ Profiler accurate: {elapsed:.0f}ms")
    
    def test_timeout_middleware_exists(self):
        """Verify timeout middleware is configured"""
        from app.middleware.timeout import TimeoutMiddleware
        
        middleware = TimeoutMiddleware(app=None)
        assert "/api/v1/agents/chat" in middleware.TIMEOUT_CONFIG
        assert middleware.TIMEOUT_CONFIG["/api/v1/agents/chat"] == 120
        print("✅ Timeout middleware configured")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
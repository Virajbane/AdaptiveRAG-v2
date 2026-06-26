import pytest
import asyncio
import time
from app.agents.orchestrator import AgentOrchestrator
from unittest.mock import AsyncMock, patch, MagicMock

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
        query_cache.clear()
        
        question = "What is machine learning?"
        user_id = "test_user"
        result = {
            "answer": "ML is...",
            "sources": [],
            "confidence": 0.9
        }
        
        # Store result
        query_cache.set(question, user_id, result)
        
        # Retrieve (should be instant)
        start = time.time()
        cached = query_cache.get(question, user_id)
        elapsed = time.time() - start
        
        assert cached is not None
        assert cached["answer"] == result["answer"]
        assert elapsed < 0.001, f"Cache lookup too slow: {elapsed*1000:.2f}ms"
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
        cache.set(question, user_id, result)
        
        # Should be available immediately
        assert cache.get(question, user_id) is not None
        
        # Wait for expiry
        await asyncio.sleep(1.1)
        
        # Should be gone
        assert cache.get(question, user_id) is None
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
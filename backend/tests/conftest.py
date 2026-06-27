"""
Shared pytest fixtures for the test suite.

Because app/db/mongodb/client.py uses a module-level global `db`
(not a FastAPI Depends() dependency), we can't use
app.dependency_overrides. Instead we directly patch that global
to point at a separate test database before each test, and restore
it afterward.
"""

import pytest
import pytest_asyncio
from app.services.cache.query_cache import query_cache
from httpx import AsyncClient, ASGITransport
from motor.motor_asyncio import AsyncIOMotorClient

from app.main import app
import app.db.mongodb.client as mongo_client_module

@pytest.fixture(autouse=True)
def clear_query_cache():
    """Ensure no cached results leak between tests"""
    query_cache.clear()
    yield
    query_cache.clear()

TEST_MONGO_URL = "mongodb://admin:password123@localhost:27017/rag_db_test?authSource=admin"
TEST_DB_NAME = "rag_db_test"


@pytest_asyncio.fixture(scope="function")
async def test_db():
    """
    Creates a connection to a separate test database, points the
    app's global `db` at it for the duration of the test, then
    cleans up afterward.
    """
    test_client = AsyncIOMotorClient(TEST_MONGO_URL)
    test_database = test_client[TEST_DB_NAME]

    # Recreate the unique email index, same as production
    await test_database["users"].create_index("email", unique=True)

    # Save whatever the global was pointing at, then redirect it
    original_db = mongo_client_module.db
    mongo_client_module.db = test_database

    yield test_database

    # Cleanup: wipe every collection used during the test
    collections = await test_database.list_collection_names()
    for name in collections:
        await test_database[name].delete_many({})

    # Restore the original global so other tests/the real app aren't affected
    mongo_client_module.db = original_db
    test_client.close()


@pytest_asyncio.fixture(scope="function")
async def client(test_db, monkeypatch):
    """
    Async HTTP client wired to the FastAPI app.

    Depends on test_db so the database swap happens before any
    request is made. Also disables rate limiting for the duration
    of the test - production rate limiting (Phase 9) is untouched,
    this only patches the in-memory store so tests aren't blocked
    by hitting the same endpoints repeatedly from the same IP.
    """
    from app.middleware.rate_limit import rate_limit_store

    async def always_allow(*args, **kwargs):
        return True

    monkeypatch.setattr(rate_limit_store, "check_rate_limit", always_allow)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
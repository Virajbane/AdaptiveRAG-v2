import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _get_token(client):
    """Helper: register (ignore if exists) then login, return access_token"""
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "searchtest@example.com",
            "password": "SecurePass123!",
            "name": "Search Test"
        }
    )
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "searchtest@example.com", "password": "SecurePass123!"}
    )
    return login_resp.json()["access_token"]


def test_hybrid_search(client):
    """Test hybrid search with vector + keyword"""
    token = _get_token(client)

    response = client.post(
        "/api/v1/retrieval/search",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "query": "revenue",
            "top_k": 5
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "search_time_ms" in data
    assert data["query"] == "revenue"


def test_search_empty_query(client):
    """Test search with empty query"""
    token = _get_token(client)

    response = client.post(
        "/api/v1/retrieval/search",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": ""}
    )

    assert response.status_code == 400


def test_search_no_token(client):
    """Test search without authentication"""
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "revenue"}
    )

    assert response.status_code == 401


def test_search_user_isolation(client):
    """Test that User A cannot see User B's results - placeholder for now"""
    # Full isolation test will be expanded in Phase 5 with multiple users
    pass
"""
Unit tests for authentication: register, login, and the bug we
already found once (JWTManager -> JWTService) gets a regression
test so it can never silently reappear.
"""

import pytest

pytestmark = pytest.mark.asyncio


async def test_register_success(client):
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "newuser@example.com",
            "password": "SecurePass123!",
            "name": "New User",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "newuser@example.com"


async def test_register_weak_password_rejected(client):
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "weakpass@example.com",
            "password": "weak",
            "name": "User",
        },
    )
    assert response.status_code == 400


async def test_register_invalid_email_rejected(client):
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "not-an-email",
            "password": "SecurePass123!",
            "name": "User",
        },
    )
    assert response.status_code == 400


async def test_register_duplicate_email_rejected(client):
    payload = {
        "email": "duplicate@example.com",
        "password": "SecurePass123!",
        "name": "User",
    }
    first = await client.post("/api/v1/auth/register", json=payload)
    assert first.status_code == 200

    second = await client.post("/api/v1/auth/register", json=payload)
    assert second.status_code == 409


async def test_login_success_returns_access_token(client):
    # Register first
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "logintest@example.com",
            "password": "SecurePass123!",
            "name": "Login Test",
        },
    )

    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "logintest@example.com", "password": "SecurePass123!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "user_id" in data


async def test_login_wrong_password_rejected(client):
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "wrongpass@example.com",
            "password": "SecurePass123!",
            "name": "User",
        },
    )

    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "wrongpass@example.com", "password": "WrongPassword999!"},
    )
    assert response.status_code == 401


async def test_login_nonexistent_user_rejected(client):
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "ghost@example.com", "password": "DoesntMatter123!"},
    )
    assert response.status_code == 401


async def test_protected_endpoint_requires_token(client):
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401


async def test_protected_endpoint_with_valid_token(client):
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "protected@example.com",
            "password": "SecurePass123!",
            "name": "Protected User",
        },
    )
    login_resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "protected@example.com", "password": "SecurePass123!"},
    )
    token = login_resp.json()["access_token"]

    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


async def test_login_does_not_raise_import_error(client):
    """
    Regression test for the JWTManager/JWTService import bug.
    If the broken import ever comes back, this test fails with a
    500 instead of passing — catching it before it reaches login
    in production again.
    """
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "regression@example.com",
            "password": "SecurePass123!",
            "name": "Regression Test",
        },
    )
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "regression@example.com", "password": "SecurePass123!"},
    )
    assert response.status_code == 200, (
        f"Expected successful login, got {response.status_code}: {response.text}"
    )
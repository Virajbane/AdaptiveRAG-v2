import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.password_service import PasswordService

client = TestClient(app)

def test_register_success():
    """Test successful user registration"""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "newuser@example.com",
            "password": "SecurePass123!",
            "name": "New User"
        }
    )
    assert response.status_code == 201
    assert response.json()["email"] == "newuser@example.com"

def test_register_weak_password():
    """Test registration with weak password"""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "user@example.com",
            "password": "weak",
            "name": "User"
        }
    )
    assert response.status_code == 400

def test_register_duplicate_email():
    """Test registration with existing email"""
    # First registration
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "duplicate@example.com",
            "password": "SecurePass123!",
            "name": "User"
        }
    )
    
    # Second registration with same email
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "duplicate@example.com",
            "password": "SecurePass456!",
            "name": "Another User"
        }
    )
    assert response.status_code == 409

def test_login_success():
    """Test successful login"""
    # Register first
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "login@example.com",
            "password": "SecurePass123!",
            "name": "User"
        }
    )
    
    # Login
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "login@example.com",
            "password": "SecurePass123!"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_invalid_password():
    """Test login with wrong password"""
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "nonexistent@example.com",
            "password": "WrongPassword123!"
        }
    )
    assert response.status_code == 401

def test_protected_endpoint_without_token():
    """Test accessing protected endpoint without token"""
    response = client.get("/api/v1/users/me")
    assert response.status_code == 401

def test_protected_endpoint_with_token():
    """Test accessing protected endpoint with valid token"""
    # Register and login
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "protected@example.com",
            "password": "SecurePass123!",
            "name": "User"
        }
    )
    
    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "protected@example.com",
            "password": "SecurePass123!"
        }
    )
    token = login_response.json()["access_token"]
    
    # Access protected endpoint
    response = client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["email"] == "protected@example.com"

def test_password_service():
    """Test password hashing and verification"""
    password = "SecurePass123!"
    
    # Hash
    hashed = PasswordService.hash_password(password)
    assert hashed != password
    
    # Verify correct password
    assert PasswordService.verify_password(password, hashed)
    
    # Verify incorrect password
    assert not PasswordService.verify_password("WrongPassword", hashed)
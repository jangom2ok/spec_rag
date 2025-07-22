"""Unit tests for core auth module to improve coverage."""

from datetime import datetime, timedelta
from unittest.mock import patch

import jwt
import pytest

from app.core.auth import (
    create_access_token,
    generate_api_key,
    get_password_hash,
    validate_api_key,
    verify_password,
    verify_token,
)


class TestAuthHelpers:
    """Test auth helper functions."""

    def test_generate_api_key_different_each_time(self):
        """Test that generate_api_key returns different values."""
        key1 = generate_api_key()
        key2 = generate_api_key()
        assert key1 != key2
        # ak_test_ (8 chars) + 32 hex chars = 40 total
        assert len(key1) == 40
        assert len(key2) == 40

    @patch("app.core.auth.SECRET_KEY", "test-secret-key")
    def test_create_access_token_without_expires_delta(self):
        """Test creating access token without specifying expires_delta."""
        data = {"sub": "test@example.com"}
        token = create_access_token(data)

        # Decode and verify
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        assert payload["sub"] == "test@example.com"

    def test_get_password_hash_and_verify(self):
        """Test password hashing and verification."""
        password = "test_password_123"
        hashed = get_password_hash(password)

        assert password != hashed
        assert verify_password(password, hashed) is True
        assert verify_password("wrong_password", hashed) is False

    def test_validate_api_key_not_found(self):
        """Test validating non-existent API key."""
        result = validate_api_key("non_existent_key")
        assert result is None

    def test_generate_api_key_format(self):
        """Test generated API key format."""
        key = generate_api_key()
        assert key.startswith("ak_test_")
        assert len(key) == 40  # ak_test_ (8 chars) + 32 hex chars

    def test_validate_api_key_invalid_format(self):
        """Test validating API key with invalid format."""
        # Test key without ak_ prefix
        result = validate_api_key("invalid_key_format")
        assert result is None

        # Test key that's too short
        result = validate_api_key("ak_short")
        assert result is None

    @patch("app.core.auth.SECRET_KEY", "test-secret-key")
    def test_verify_token_invalid(self):
        """Test verify_token with invalid token."""
        # Create token with wrong secret
        data = {"sub": "test@example.com"}
        token = jwt.encode(data, "wrong-secret-key", algorithm="HS256")

        with pytest.raises(jwt.InvalidTokenError):
            verify_token(token)

    @patch("app.core.auth.SECRET_KEY", "test-secret-key")
    def test_verify_token_expired(self):
        """Test verify_token with expired token."""
        # Create an expired token
        data = {
            "sub": "test@example.com",
            "exp": datetime.utcnow() - timedelta(hours=1),
        }
        token = jwt.encode(data, "test-secret-key", algorithm="HS256")

        with pytest.raises(jwt.ExpiredSignatureError):
            verify_token(token)

    @patch("app.core.auth.SECRET_KEY", "test-secret-key")
    def test_create_access_token_with_custom_data(self):
        """Test creating access token with custom data and role-based permissions."""
        # Test with admin role
        data = {"sub": "admin@example.com", "role": "admin"}
        token = create_access_token(data)

        payload = verify_token(token)
        assert payload["sub"] == "admin@example.com"
        assert payload["role"] == "admin"
        assert payload["permissions"] == ["read", "write", "delete", "admin"]

        # Test with manager role
        data = {"sub": "manager@example.com", "role": "manager"}
        token = create_access_token(data)

        payload = verify_token(token)
        assert payload["permissions"] == ["read", "write"]

    def test_validate_api_key_valid(self):
        """Test validating a valid API key."""
        # Test with a known test key
        result = validate_api_key("ak_test_1234567890abcdef")
        assert result is not None
        assert result["user_id"] == "user123"
        assert result["permissions"] == ["read", "write"]

        # Test with readonly key
        result = validate_api_key("ak_readonly_1234567890abcdef")
        assert result is not None
        assert result["user_id"] == "readonly_user"
        assert result["permissions"] == ["read"]

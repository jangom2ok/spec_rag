"""Tests for core exceptions."""

from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    DatabaseError,
    RAGSystemError,
    ValidationError,
    VectorDatabaseError,
)


class TestExceptions:
    """Exception tests."""

    def test_rag_system_error_with_defaults(self):
        """Test RAGSystemError with default values."""
        error = RAGSystemError("Test error")
        assert str(error) == "Test error"
        assert error.error_code == "RAG_SYSTEM_ERROR"
        assert error.status_code == 500

    def test_rag_system_error_with_custom_values(self):
        """Test RAGSystemError with custom values."""
        error = RAGSystemError("Custom error", "CUSTOM_ERROR", 503)
        assert str(error) == "Custom error"
        assert error.error_code == "CUSTOM_ERROR"
        assert error.status_code == 503

    def test_database_error(self):
        """Test DatabaseError."""
        error = DatabaseError("Database connection failed")
        assert str(error) == "Database connection failed"
        assert error.error_code == "DATABASE_ERROR"
        assert error.status_code == 500

    def test_vector_database_error(self):
        """Test VectorDatabaseError."""
        error = VectorDatabaseError("Vector operation failed")
        assert str(error) == "Vector operation failed"
        assert error.error_code == "VECTOR_DATABASE_ERROR"
        assert error.status_code == 500

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid input")
        assert str(error) == "Invalid input"
        assert error.error_code == "VALIDATION_ERROR"
        assert error.status_code == 422

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Invalid credentials")
        assert str(error) == "Invalid credentials"
        assert error.error_code == "AUTHENTICATION_ERROR"
        assert error.status_code == 401

    def test_authorization_error(self):
        """Test AuthorizationError."""
        error = AuthorizationError("Access denied")
        assert str(error) == "Access denied"
        assert error.error_code == "AUTHORIZATION_ERROR"
        assert error.status_code == 403

    def test_exception_inheritance(self):
        """Test that all exceptions inherit from RAGSystemError."""
        assert issubclass(DatabaseError, RAGSystemError)
        assert issubclass(VectorDatabaseError, RAGSystemError)
        assert issubclass(ValidationError, RAGSystemError)
        assert issubclass(AuthenticationError, RAGSystemError)
        assert issubclass(AuthorizationError, RAGSystemError)

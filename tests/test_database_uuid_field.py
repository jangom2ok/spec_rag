"""Tests for database UUID field."""

import uuid
from unittest.mock import Mock

import pytest

from app.models.database import UUIDType


class TestUUIDType:
    """UUIDType tests."""

    @pytest.fixture
    def uuid_type(self):
        """Create UUIDType instance."""
        return UUIDType()

    def test_process_bind_param_with_uuid(self, uuid_type):
        """Test process_bind_param with UUID object."""
        test_uuid = uuid.uuid4()
        result = uuid_type.process_bind_param(test_uuid, Mock())
        assert result == str(test_uuid)

    def test_process_bind_param_with_string(self, uuid_type):
        """Test process_bind_param with string."""
        test_string = "test-uuid-string"
        result = uuid_type.process_bind_param(test_string, Mock())
        assert result == test_string

    def test_process_bind_param_with_none(self, uuid_type):
        """Test process_bind_param with None."""
        result = uuid_type.process_bind_param(None, Mock())
        assert result is None

    def test_process_bind_param_with_other_type(self, uuid_type):
        """Test process_bind_param with other type."""
        test_value = 12345
        result = uuid_type.process_bind_param(test_value, Mock())
        assert result == "12345"

    def test_process_result_value_with_none(self, uuid_type):
        """Test process_result_value with None."""
        result = uuid_type.process_result_value(None, Mock())
        assert result is None

    def test_process_result_value_with_string(self, uuid_type):
        """Test process_result_value with string."""
        test_string = "test-uuid-string"
        result = uuid_type.process_result_value(test_string, Mock())
        assert result == test_string

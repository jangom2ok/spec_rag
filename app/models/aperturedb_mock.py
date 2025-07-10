"""ApertureDB mock for testing and CI/CD environments"""

import os
from typing import Any


class MockClient:
    """Mock ApertureDB Client for testing"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 55555,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username or os.getenv("APERTUREDB_USERNAME", "admin")
        self.password = password or os.getenv("APERTUREDB_PASSWORD", "admin")
        self._connected = True

    def query(
        self, query: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Mock query method"""
        # Return mock responses based on query type
        if not query:
            return [], []

        first_query = query[0]
        query_type = list(first_query.keys())[0]

        if query_type == "FindDescriptorSet":
            return [{"FindDescriptorSet": {"count": 0}}], []
        elif query_type == "AddDescriptorSet":
            return [{"AddDescriptorSet": {"status": 0}}], []
        elif query_type == "AddDescriptor":
            return [{"AddDescriptor": {"status": 0}}], []
        elif query_type == "FindDescriptor":
            return [{"FindDescriptor": {"returned": 0, "entities": []}}], []
        elif query_type == "DeleteDescriptor":
            return [{"DeleteDescriptor": {"status": 0}}], []
        elif query_type == "AddEntity":
            return [{"AddEntity": {"status": 0}}], []
        elif query_type == "GetStatus":
            return [{"GetStatus": {"status": "ready"}}], []
        else:
            return [{}], []


class DBError(Exception):
    """Mock ApertureDB exception"""

    pass


class Utils:
    """Mock ApertureDB Utils class"""

    pass


# Export mock classes
Client = MockClient
DBException = DBError  # Export with original name for compatibility

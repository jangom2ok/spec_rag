"""Extended test fixtures for mocking external dependencies.

This file provides comprehensive fixtures for mocking:
- Celery/Redis
- External APIs (Confluence, JIRA)
- Database connections (PostgreSQL, ApertureDB)
- GPU/CUDA
- NLP models
"""

from collections.abc import AsyncGenerator
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base

# ==================== Celery/Redis Fixtures ====================


@pytest.fixture
def mock_celery_app():
    """Mock Celery application."""
    with patch("app.services.embedding_tasks.celery_app") as mock_app:
        # Mock task decorator
        def mock_task_decorator(*args, **kwargs):
            def decorator(func):
                # Create a mock that behaves like a Celery task
                mock_task = Mock()
                mock_task.delay = Mock(
                    return_value=Mock(id="mock-task-id-123", state="PENDING")
                )
                mock_task.apply_async = Mock(
                    return_value=Mock(id="mock-task-id-123", state="PENDING")
                )
                return mock_task

            return decorator

        mock_app.task = mock_task_decorator
        yield mock_app


@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    with patch("redis.asyncio.from_url") as mock_redis:
        client = AsyncMock()
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock(return_value=True)
        client.delete = AsyncMock(return_value=1)
        client.exists = AsyncMock(return_value=0)
        client.expire = AsyncMock(return_value=True)
        client.ping = AsyncMock(return_value=True)
        client.close = AsyncMock()
        mock_redis.return_value = client
        yield client


@pytest.fixture
def mock_celery_task_result():
    """Mock Celery task result."""
    result = Mock()
    result.id = "mock-task-id-123"
    result.state = "SUCCESS"
    result.result = {"status": "completed", "data": {"embeddings_generated": 100}}
    result.ready = Mock(return_value=True)
    result.successful = Mock(return_value=True)
    result.failed = Mock(return_value=False)
    return result


# ==================== External API Fixtures ====================


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for external API calls."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()

        # Mock Confluence responses
        mock_client.get = AsyncMock(
            side_effect=lambda url, **kwargs: AsyncMock(
                status_code=200,
                json=AsyncMock(
                    return_value={
                        "results": [
                            {
                                "id": "123456",
                                "type": "page",
                                "title": "Test Page",
                                "body": {"storage": {"value": "<p>Test content</p>"}},
                                "_links": {
                                    "self": "https://confluence.example.com/rest/api/content/123456"
                                },
                            }
                        ],
                        "size": 1,
                        "_links": {"next": None},
                    }
                ),
            )
        )

        # Mock JIRA responses
        mock_client.post = AsyncMock(
            return_value=AsyncMock(
                status_code=200,
                json=AsyncMock(
                    return_value={
                        "issues": [
                            {
                                "id": "10001",
                                "key": "TEST-1",
                                "fields": {
                                    "summary": "Test Issue",
                                    "description": "Test description",
                                    "issuetype": {"name": "Task"},
                                    "status": {"name": "Open"},
                                    "created": "2024-01-01T00:00:00.000+0000",
                                    "updated": "2024-01-01T00:00:00.000+0000",
                                },
                            }
                        ],
                        "total": 1,
                    }
                ),
            )
        )

        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client
        yield mock_client


# ==================== Database Fixtures ====================


@pytest.fixture
def mock_aperturedb_client():
    """Mock ApertureDB client."""
    with patch("aperturedb.Client") as mock_client_class:
        mock_client = Mock()

        # Mock query responses
        mock_client.query = Mock(
            return_value=(
                [{"AddDescriptorSet": {"status": 0}}],  # response
                [],  # blobs
            )
        )

        # Mock search responses
        mock_client.find_descriptor = Mock(
            return_value={
                "descriptors": [
                    {"_id": "desc1", "distance": 0.1},
                    {"_id": "desc2", "distance": 0.2},
                ]
            }
        )

        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_asyncpg_pool():
    """Mock asyncpg connection pool."""
    with patch("asyncpg.create_pool") as mock_create_pool:
        pool = AsyncMock()
        pool.acquire = AsyncMock()
        pool.close = AsyncMock()

        # Mock connection
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=1)  # For SELECT 1
        conn.fetch = AsyncMock(return_value=[])
        conn.execute = AsyncMock()
        conn.close = AsyncMock()

        # Setup context manager
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock()

        mock_create_pool.return_value = pool
        yield pool


# ==================== GPU/Hardware Fixtures ====================


@pytest.fixture
def mock_cuda_available():
    """Mock CUDA availability."""
    with patch("torch.cuda.is_available", return_value=False):
        with patch("torch.cuda.device_count", return_value=0):
            with patch(
                "torch.cuda.get_device_name", side_effect=RuntimeError("No CUDA")
            ):
                yield


@pytest.fixture
def mock_gpu_memory():
    """Mock GPU memory utilities."""
    with patch("torch.cuda.get_device_properties") as mock_props:
        mock_props.return_value = Mock(total_memory=8589934592)  # 8GB
        with patch("torch.cuda.memory_allocated", return_value=1073741824):  # 1GB
            with patch("torch.cuda.memory_reserved", return_value=2147483648):  # 2GB
                yield


# ==================== NLP Model Fixtures ====================


@pytest.fixture
def mock_spacy_model():
    """Mock spaCy NLP model."""
    with patch("spacy.load") as mock_load:
        # Create mock NLP model
        mock_nlp = Mock()

        # Mock document processing
        def mock_process(text):
            doc = Mock()
            doc.text = text
            doc.ents = [
                Mock(text="Test Entity", label_="ORG"),
                Mock(text="John Doe", label_="PERSON"),
            ]
            doc.noun_chunks = [Mock(text="test document"), Mock(text="important data")]
            return doc

        mock_nlp.side_effect = mock_process
        mock_nlp.return_value = mock_process("test")

        mock_load.return_value = mock_nlp
        yield mock_nlp


@pytest.fixture
def mock_transformers_model():
    """Mock Transformers model for embeddings."""
    with patch("transformers.AutoModel.from_pretrained") as mock_model:
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
            # Mock model
            model = Mock()
            model.eval = Mock()
            model.to = Mock(return_value=model)

            # Mock forward pass
            import torch

            model.forward = Mock(
                return_value=Mock(
                    last_hidden_state=torch.randn(1, 10, 768),
                    pooler_output=torch.randn(1, 768),
                )
            )

            # Mock tokenizer
            tokenizer = Mock()
            tokenizer.return_value = {
                "input_ids": torch.tensor([[101, 2023, 2003, 1037, 3231, 102]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
            }
            tokenizer.pad_token_id = 0

            mock_model.return_value = model
            mock_tokenizer.return_value = tokenizer

            yield model, tokenizer


# ==================== Embedding Service Fixtures ====================


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    with patch("app.services.embedding_service.EmbeddingService") as mock_service_class:
        service = Mock()

        # Mock embedding generation
        import numpy as np

        service.generate_embeddings = AsyncMock(
            return_value={
                "dense": np.random.rand(1, 1024).astype(np.float32),
                "sparse": {"indices": [1, 5, 10], "values": [0.5, 0.3, 0.2]},
                "colbert": np.random.rand(10, 128).astype(np.float32),
            }
        )

        service.generate_query_embeddings = AsyncMock(
            return_value={
                "dense": np.random.rand(1024).astype(np.float32),
                "sparse": {"indices": [1, 5, 10], "values": [0.5, 0.3, 0.2]},
                "colbert": np.random.rand(10, 128).astype(np.float32),
            }
        )

        mock_service_class.return_value = service
        yield service


# ==================== Test Database Fixtures ====================


@pytest_asyncio.fixture
async def test_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    # Use SQLite for testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False  # type: ignore[arg-type]
    )

    async with async_session_maker() as session:
        yield session

    await engine.dispose()


# ==================== Environment Fixtures ====================


@pytest.fixture(autouse=True)
def mock_environment_variables(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("APERTUREDB_HOST", "localhost")
    monkeypatch.setenv("APERTUREDB_PORT", "55555")
    monkeypatch.setenv("DISABLE_EXTERNAL_APIS", "true")
    yield


# ==================== Utility Fixtures ====================


@pytest.fixture
def mock_datetime_now():
    """Mock datetime.now() for consistent testing."""
    fixed_time = datetime(2024, 1, 1, 12, 0, 0)
    with patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value = fixed_time
        mock_datetime.utcnow.return_value = fixed_time
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield fixed_time


@pytest.fixture
def anyio_backend():
    """Specify anyio backend for async tests."""
    return "asyncio"

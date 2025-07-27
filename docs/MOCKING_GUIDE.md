# Mocking Guide for External Dependencies

## Overview

This guide provides patterns and best practices for mocking external dependencies in the spec_rag test suite.

## Mock Fixtures Available

All fixtures are defined in `tests/conftest_extended.py`.

### 1. Celery/Redis Mocks

#### `mock_celery_app`

Mocks the Celery application and task decorators.

```python
def test_with_celery(mock_celery_app):
    # Celery tasks will be mocked automatically
    result = some_celery_task.delay(data)
    assert result.id == "mock-task-id-123"
```

#### `mock_redis_client`

Mocks Redis async client operations.

```python
async def test_with_redis(mock_redis_client):
    # Redis operations are mocked
    await mock_redis_client.set("key", "value")
    value = await mock_redis_client.get("key")
```

#### `mock_celery_task_result`

Provides a mock Celery task result object.

```python
def test_task_result(mock_celery_task_result):
    assert mock_celery_task_result.state == "SUCCESS"
    assert mock_celery_task_result.ready() is True
```

### 2. External API Mocks

#### `mock_httpx_client`

Mocks HTTP client for external API calls (Confluence, JIRA).

```python
async def test_api_call(mock_httpx_client):
    # HTTP requests will return mocked responses
    response = await client.get("https://api.example.com/data")
    assert response.status_code == 200
```

### 3. Database Mocks

#### `mock_aperturedb_client`

Mocks ApertureDB client operations.

```python
def test_vector_db(mock_aperturedb_client):
    # ApertureDB operations are mocked
    response, blobs = mock_aperturedb_client.query([{"AddDescriptorSet": {}}])
    assert response[0]["AddDescriptorSet"]["status"] == 0
```

#### `mock_asyncpg_pool`

Mocks PostgreSQL connection pool.

```python
async def test_postgres(mock_asyncpg_pool):
    # PostgreSQL operations are mocked
    async with mock_asyncpg_pool.acquire() as conn:
        result = await conn.fetchval("SELECT 1")
        assert result == 1
```

### 4. GPU/Hardware Mocks

#### `mock_cuda_available`

Mocks CUDA availability checks.

```python
def test_gpu_code(mock_cuda_available):
    # CUDA will appear as unavailable
    import torch
    assert torch.cuda.is_available() is False
```

#### `mock_gpu_memory`

Mocks GPU memory properties.

```python
def test_memory_optimization(mock_gpu_memory):
    # GPU memory checks will return mocked values
    import torch
    props = torch.cuda.get_device_properties(0)
    assert props.total_memory == 8589934592  # 8GB
```

### 5. NLP Model Mocks

#### `mock_spacy_model`

Mocks spaCy NLP models.

```python
def test_nlp_processing(mock_spacy_model):
    # spaCy model loading and processing is mocked
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Test text")
    assert len(doc.ents) > 0
```

#### `mock_transformers_model`

Mocks Transformers models for embeddings.

```python
def test_embeddings(mock_transformers_model):
    model, tokenizer = mock_transformers_model
    # Model operations are mocked
    outputs = model(input_ids)
    assert outputs.last_hidden_state.shape[0] == 1
```

### 6. Service Mocks

#### `mock_embedding_service`

Mocks the complete embedding service.

```python
async def test_embedding_generation(mock_embedding_service):
    embeddings = await mock_embedding_service.generate_embeddings(["text"])
    assert "dense" in embeddings
    assert embeddings["dense"].shape == (1, 1024)
```

### 7. Environment Mocks

#### `mock_environment_variables`

Automatically sets test environment variables (applied to all tests).

```python
# Automatically applied, sets:
# TESTING=true
# DISABLE_EXTERNAL_APIS=true
# etc.
```

## Common Mocking Patterns

### 1. Mocking External API Calls

```python
async def test_external_api_integration(mock_httpx_client):
    # Setup specific response
    mock_httpx_client.get.return_value = AsyncMock(
        status_code=200,
        json=AsyncMock(return_value={"data": "test"})
    )

    # Your test code
    result = await fetch_external_data()
    assert result["data"] == "test"
```

### 2. Mocking Database Operations

```python
async def test_database_operation(mock_asyncpg_pool):
    # Mock specific query results
    conn = AsyncMock()
    conn.fetch.return_value = [{"id": 1, "name": "Test"}]

    mock_asyncpg_pool.acquire.return_value.__aenter__.return_value = conn

    # Your test code
    results = await get_users()
    assert len(results) == 1
```

### 3. Mocking Task Queue Operations

```python
def test_async_task_submission(mock_celery_app):
    with patch("app.services.embedding_tasks.HAS_CELERY", True):
        # Mock task submission
        mock_task = Mock()
        mock_task.delay.return_value = Mock(id="task-123")

        with patch("app.services.embedding_tasks.process_document_embedding", mock_task):
            result = submit_document_task("doc-1")
            assert result.id == "task-123"
```

### 4. Mocking Hardware Dependencies

```python
def test_gpu_optimization(mock_cuda_available, mock_gpu_memory):
    # Test code that checks GPU availability
    optimizer = GPUOptimizer()

    # Should fallback to CPU
    assert optimizer.device == "cpu"
    assert optimizer.memory_limit == 8 * 1024 * 1024 * 1024  # 8GB
```

### 5. Mocking NLP Operations

```python
def test_entity_extraction(mock_spacy_model):
    # Extract entities from text
    extractor = EntityExtractor()
    entities = extractor.extract("John Doe works at Test Company")

    assert len(entities) == 2
    assert entities[0]["text"] == "Test Entity"
    assert entities[0]["label"] == "ORG"
```

## Testing Strategies

### 1. Unit Tests with Mocks

For pure unit tests, mock all external dependencies:

```python
@pytest.mark.unit
async def test_business_logic(
    mock_redis_client,
    mock_aperturedb_client,
    mock_embedding_service
):
    # Test only the business logic
    result = await process_document(doc)
    assert result.status == "processed"
```

### 2. Integration Tests with Selective Mocks

For integration tests, mock only external services:

```python
@pytest.mark.integration
async def test_api_endpoint(
    test_client,  # Real FastAPI test client
    mock_httpx_client,  # Mock external APIs
    test_db_session  # Real test database
):
    response = await test_client.post("/api/process", json=data)
    assert response.status_code == 200
```

### 3. Error Handling Tests

Test error scenarios by making mocks raise exceptions:

```python
async def test_connection_error_handling(mock_redis_client):
    # Make Redis operations fail
    mock_redis_client.ping.side_effect = Exception("Connection refused")

    # Test error handling
    health = await check_redis_health()
    assert health["status"] == "unhealthy"
```

### 4. Performance Tests

Mock slow operations to test timeouts and performance:

```python
async def test_timeout_handling(mock_httpx_client):
    # Simulate slow response
    async def slow_response(*args, **kwargs):
        await asyncio.sleep(5)  # 5 second delay
        return AsyncMock(status_code=200)

    mock_httpx_client.get.side_effect = slow_response

    # Test should timeout
    with pytest.raises(TimeoutError):
        await fetch_with_timeout(timeout=1)
```

## Best Practices

### 1. Use Fixtures Consistently

Always use the provided fixtures instead of creating ad-hoc mocks:

```python
# Good
def test_redis_operation(mock_redis_client):
    # Use the fixture

# Bad
def test_redis_operation():
    with patch("redis.asyncio.from_url"):  # Don't do this
```

### 2. Mock at the Right Level

Mock at the boundary of your application:

```python
# Good - Mock the external client
def test_feature(mock_httpx_client):
    mock_httpx_client.get.return_value = ...

# Bad - Mock internal methods
def test_feature():
    with patch("app.services.my_service._internal_method"):
```

### 3. Verify Mock Calls

Always verify that mocks were called correctly:

```python
def test_api_call(mock_httpx_client):
    await make_api_request(data)

    # Verify the call
    mock_httpx_client.post.assert_called_once_with(
        "https://api.example.com/endpoint",
        json=data,
        headers={"Authorization": "Bearer token"}
    )
```

### 4. Reset Mocks Between Tests

Fixtures automatically reset, but for manual mocks:

```python
def test_something():
    mock_obj = Mock()
    # ... use mock_obj
    mock_obj.reset_mock()  # If reusing
```

### 5. Use Realistic Mock Data

Make mock responses realistic to catch integration issues:

```python
# Good - Realistic response structure
mock_response = {
    "id": "123",
    "status": "success",
    "data": {"field": "value"},
    "timestamp": "2024-01-01T00:00:00Z"
}

# Bad - Oversimplified
mock_response = {"ok": True}
```

## Debugging Mock Issues

### 1. Check Mock Call History

```python
# See all calls made to a mock
print(mock_obj.call_args_list)
print(mock_obj.method_calls)
```

### 2. Use Mock Assertions

```python
# Various assertions available
mock_obj.assert_called()
mock_obj.assert_called_once()
mock_obj.assert_called_with(specific_args)
mock_obj.assert_not_called()
```

### 3. Debug Fixture Loading

```python
# List all fixtures available in a test
def test_debug(request):
    print(request.fixturenames)
```

### 4. Mock State Inspection

```python
# Inspect mock state
assert mock_obj.called
assert mock_obj.call_count == 2
assert mock_obj.return_value == expected_value
```

## Common Pitfalls

### 1. Forgetting Async Context

```python
# Wrong - Sync mock for async code
mock_client.get.return_value = {"data": "test"}

# Correct - Async mock
mock_client.get.return_value = AsyncMock(
    json=AsyncMock(return_value={"data": "test"})
)
```

### 2. Incorrect Patch Target

```python
# Wrong - Patching where defined
@patch("external_library.Client")

# Correct - Patching where used
@patch("app.services.my_service.Client")
```

### 3. Mock Leakage

```python
# Use fixtures or context managers to prevent leakage
with patch("module.function") as mock_func:
    # Mock is active here
    pass
# Mock is no longer active
```

### 4. Over-Mocking

Don't mock what you're testing:

```python
# Bad - Mocking the system under test
def test_my_service():
    with patch("app.services.my_service.MyService"):
        # This doesn't test anything useful
```

## Running Tests with Mocks

```bash
# Run all tests with mocks
TESTING=true pytest

# Run specific test file
TESTING=true pytest tests/test_redis_integration_fixed.py

# Run with verbose output
TESTING=true pytest -v

# Run only mocked tests (exclude integration)
TESTING=true pytest -m "not integration"
```

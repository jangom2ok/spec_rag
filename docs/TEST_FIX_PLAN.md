# Test Fix Plan for External Dependencies

## Overview

This document outlines the plan to fix failing tests due to external dependencies in the spec_rag project.

## Failing Test Categories

### 1. Celery/Redis Tests (10 tests)
**Files**: `test_embedding_tasks.py`, `test_redis_integration.py`

**Issues**:
- Tests attempting to connect to actual Redis instance
- Celery tasks trying to execute without broker
- Task status checks failing

**Solution**:
- Use `mock_celery_app` and `mock_redis_client` fixtures
- Mock task submission and result retrieval
- Use in-memory task tracking

### 2. External API Tests (11 tests)
**File**: `test_external_source_integration.py`

**Issues**:
- HTTP requests to non-existent Confluence/JIRA instances
- 404/400 errors from API endpoints
- Authentication failures

**Solution**:
- Use `mock_httpx_client` fixture
- Mock API responses with realistic data
- Test error handling with mocked error responses

### 3. Database Connection Tests (8 tests)
**File**: `test_production_database.py`

**Issues**:
- PostgreSQL connection attempts failing
- ApertureDB client initialization errors
- Connection pool creation failures

**Solution**:
- Use `mock_asyncpg_pool` and `mock_aperturedb_client` fixtures
- Mock health check responses
- Simulate connection failures for error handling tests

### 4. GPU/Hardware Tests (2 tests)
**File**: `test_embedding_optimization.py`

**Issues**:
- CUDA not available in test environment
- GPU memory checks failing

**Solution**:
- Use `mock_cuda_available` and `mock_gpu_memory` fixtures
- Mock hardware capabilities
- Test optimization logic without actual GPU

### 5. NLP Library Tests (4 tests)
**File**: `test_metadata_extractor.py`

**Issues**:
- spaCy model loading failures
- Entity extraction requiring actual NLP models

**Solution**:
- Use `mock_spacy_model` fixture
- Mock entity and keyword extraction results
- Test processing logic without model dependencies

### 6. Vector Database Tests (1 test)
**File**: `test_error_handling.py`

**Issues**:
- ApertureDB connection required for error handling tests

**Solution**:
- Use `mock_aperturedb_client` fixture
- Simulate various error conditions
- Test error recovery logic

## Implementation Strategy

### Phase 1: Setup Infrastructure
1. ✅ Create `conftest_extended.py` with all mock fixtures
2. Configure pytest to use extended fixtures
3. Add test markers for external dependency tests

### Phase 2: Fix Test Files (Priority Order)
1. **test_redis_integration.py** - Critical for async task processing
2. **test_embedding_tasks.py** - Core functionality
3. **test_external_source_integration.py** - External data ingestion
4. **test_production_database.py** - Database setup and health
5. **test_metadata_extractor.py** - Document processing
6. **test_embedding_optimization.py** - Performance optimization
7. **test_error_handling.py** - Error recovery

### Phase 3: Validation
1. Run each test file individually with mocks
2. Ensure no external connections are attempted
3. Verify test coverage remains adequate
4. Run full test suite

## Test Markers

Add these markers to categorize tests:

```python
# In pyproject.toml
[tool.pytest.ini_options]
markers = [
    "external: marks tests that require external services",
    "unit: marks unit tests that should run without external dependencies",
    "integration: marks integration tests",
    "slow: marks slow tests",
]
```

## Environment Variables

Tests should respect these environment variables:

```bash
TESTING=true                    # Enable test mode
DISABLE_EXTERNAL_APIS=true     # Disable external API calls
USE_MOCK_EMBEDDINGS=true       # Use mock embeddings
MOCK_HARDWARE=true             # Mock GPU/hardware checks
```

## Running Tests

```bash
# Run all tests with mocks
TESTING=true pytest

# Run only unit tests
pytest -m "unit"

# Run tests excluding external dependencies
pytest -m "not external"

# Run specific test file with verbose output
pytest tests/test_redis_integration.py -v

# Run with coverage
pytest --cov=app --cov-report=html
```

## Success Criteria

1. All 38 failing tests pass with mocks
2. No external connections attempted during tests
3. Test execution time reduced significantly
4. Tests can run in CI/CD environment without external services
5. Test coverage maintained or improved

## Notes

- The codebase already has mock implementations in some services (e.g., `embedding_tasks.py`)
- Leverage existing mock classes where available
- Ensure mocks are realistic to catch actual issues
- Add integration test markers for tests that genuinely need external services
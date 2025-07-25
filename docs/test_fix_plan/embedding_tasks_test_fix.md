# Embedding Tasks Test Fix

## Issue
The embedding tasks tests in `test_embedding_tasks.py` were failing in GitHub Actions with the following issues:
1. Tests were trying to access `__wrapped__` attribute on mocked Celery tasks
2. When `HAS_CELERY` is False, the mock decorator doesn't create this attribute
3. Asyncio event loop mocking was incomplete

## Root Cause
1. The mock Celery implementation in `embedding_tasks.py` doesn't add the `__wrapped__` attribute that the real Celery decorator provides
2. The tests were written to access the underlying function via `__wrapped__`, which doesn't exist in the mock
3. The asyncio event loop mocking needed both `new_event_loop` and `set_event_loop` patches

## Solution
1. **Direct Function Calls**: Modified all tests to call the Celery task functions directly instead of accessing `__wrapped__`
2. **Complete Event Loop Mocking**: Added proper patches for both `asyncio.new_event_loop` and `asyncio.set_event_loop`
3. **Mock Loop Methods**: Ensured the mock event loop has all required methods like `close()`

## Changes Made

### `/tests/test_embedding_tasks.py`
- Removed all `__wrapped__` attribute access
- Changed from `process_document_embedding_task.__wrapped__(...)` to `process_document_embedding_task(...)`
- Added `asyncio.set_event_loop` patches alongside `asyncio.new_event_loop`
- Added `mock_loop.close = Mock()` to ensure the close method exists
- Updated `update_state` mock on task objects

## Test Results
All 14 tests in `test_embedding_tasks.py` are now passing:
- TestEmbeddingTaskService: 3 tests passed
- TestEmbeddingTaskManager: 5 tests passed  
- TestCeleryTasks: 5 tests passed
- TestIntegration: 1 test passed

## Benefits
1. Tests work correctly with the mock Celery implementation
2. No dependency on Celery's internal implementation details
3. Proper asyncio event loop handling in tests
4. Tests are more maintainable and less brittle
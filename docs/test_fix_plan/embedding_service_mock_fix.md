# Embedding Service Mock Fix

## Issue
In GitHub Actions, the embedding service tests were failing with the error:
```
TypeError: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'return_dense'
```

This happened because the real BGE-M3 model was being loaded instead of using the mock, and the tokenizer was receiving unexpected keyword arguments (`return_dense`, `return_sparse`, `return_colbert_vecs`) that should only be passed to the `FlagModel.encode()` method.

## Root Cause
The mock for `FlagModel` was being applied within the fixture scope using a context manager (`with patch(...)`), but when the fixture returned the service, the patch context exited. This meant that when the actual test ran and called `initialize()`, the real FlagModel was imported instead of the mock.

## Solution
1. **Global Mock in conftest.py**: Added a global mock for `FlagModel` in the `mock_external_services` fixture that is applied with `autouse=True`. This ensures all tests use the mocked version.

2. **Removed Local Mocks**: Removed the local `mock_bge_model` fixture and the `mock_flag_model` autouse fixture from `test_embedding_service.py` to avoid conflicts.

3. **Fixed Error Test**: Updated the `test_error_handling_during_embedding` test to use a local patch with a custom error-raising mock class.

## Changes Made

### `/tests/conftest.py`
- Added `MockFlagModel` class that properly mocks the BGE-M3 model's encode method
- Added the mock to the `mock_external_services` fixture with `autouse=True`

### `/tests/test_embedding_service.py`
- Removed the `mock_bge_model` fixture
- Removed the `mock_flag_model` autouse fixture
- Simplified the `embedding_service` fixture to not use any patches
- Updated `test_error_handling_during_embedding` to use a local patch

### `/tests/test_auth_unit.py`
- Fixed the test assertion to match the actual API key format (removed extra underscore)

## Benefits
1. Consistent mocking across all tests
2. No real model downloads or initialization in CI/CD
3. Faster test execution
4. No dependency on external model files
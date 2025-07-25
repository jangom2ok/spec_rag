# Test Coverage 100% Achievement Plan

## Current Status
- **Current Coverage**: 84.97%
- **Target Coverage**: 100%
- **Gap**: 15.03%

## Overview
This document outlines the plan to fix failing tests and achieve 100% test coverage for the spec_rag project.

## Phase 1: Fix Critical Test Failures (Priority: High)

### 1.1 Fix test_100_percent_coverage.py Issues
These tests are specifically designed to cover missing lines but are failing due to incorrect assumptions.

#### Tasks:
1. **Fix EmbeddingService.has_gpu attribute error**
   - File: `tests/test_100_percent_coverage.py::TestEmbeddingServiceCoverage`
   - Issue: `'EmbeddingService' object has no attribute 'has_gpu'`
   - Solution: Check actual EmbeddingService implementation and update test

2. **Fix VectorCollection test failures**
   - File: `tests/test_100_percent_coverage.py::TestApertureDBCoverage`
   - Review VectorCollection implementation and mock properly

3. **Fix other service test failures**
   - DocumentChunker, DocumentCollector, EmbeddingTasks, etc.
   - Update tests to match actual implementation

### 1.2 Fix Authentication Test Failures
Many tests are failing due to authentication/authorization issues.

#### Tasks:
1. **Fix JWT blacklist tests**
   - Files: Multiple test files with JWT blacklist failures
   - Issue: Import errors and incorrect mocking
   - Solution: Properly mock `is_token_blacklisted` function

2. **Fix API key validation tests**
   - Ensure proper mocking of `validate_api_key` function

## Phase 2: Coverage Gap Analysis (Priority: High)

### 2.1 Modules with Low Coverage
Based on the coverage report, focus on these modules:

1. **app/services/metrics_collection.py** (70.17% → 100%)
   - Missing: 173 lines
   - Focus areas: Error handling, edge cases, metric aggregation

2. **app/database/production_config.py** (76.37% → 100%)
   - Missing: 116 lines
   - Focus areas: Database configuration edge cases

3. **app/services/search_suggestions.py** (76.67% → 100%)
   - Missing: 101 lines
   - Focus areas: Suggestion generation, personalization

4. **app/services/query_expansion.py** (77.97% → 100%)
   - Missing: 76 lines
   - Focus areas: Query expansion algorithms

5. **app/services/reranker.py** (76.81% → 100%)
   - Missing: 61 lines
   - Focus areas: Reranking algorithms

### 2.2 High Coverage Modules Needing Minor Fixes

1. **app/api/search.py** (80.12% → 100%)
   - Missing: 65 lines
   - Focus: Error paths, exception handling

2. **app/core/auth.py** (86.26% → 100%)
   - Missing: 29 lines
   - Focus: Edge cases in authentication

3. **app/services/embedding_service.py** (84.31% → 100%)
   - Missing: 24 lines
   - Focus: GPU detection, error handling

## Phase 3: Test Implementation Strategy

### 3.1 Test File Organization
Create focused test files for each coverage gap:

```
tests/coverage_100/
├── test_metrics_collection_coverage.py
├── test_production_config_coverage.py
├── test_search_suggestions_coverage.py
├── test_query_expansion_coverage.py
├── test_reranker_coverage.py
└── test_remaining_gaps.py
```

### 3.2 Testing Patterns

#### Pattern 1: Error Path Testing
```python
@pytest.mark.asyncio
async def test_error_handling():
    with patch('module.function', side_effect=Exception("Test error")):
        # Test error handling code path
```

#### Pattern 2: Edge Case Testing
```python
def test_edge_cases():
    # Test with None values
    # Test with empty collections
    # Test with invalid inputs
```

#### Pattern 3: Mock Complex Dependencies
```python
@patch('external.dependency')
def test_with_mocked_dependency(mock_dep):
    mock_dep.return_value = Mock(spec=DependencyClass)
    # Test code that uses the dependency
```

## Phase 4: Specific Coverage Targets

### 4.1 app/services/metrics_collection.py
Missing coverage in:
- Lines 335-355: Metric aggregation logic
- Lines 435-459: Export functionality
- Lines 923-939: Alert integration
- Lines 1069-1089: Prometheus export

### 4.2 app/database/production_config.py
Missing coverage in:
- Lines 625-669: Connection pooling
- Lines 696-726: Failover logic
- Lines 914-932: Migration scripts
- Lines 944-957: Backup procedures

### 4.3 app/services/search_suggestions.py
Missing coverage in:
- Lines 501-565: Personalization logic
- Lines 579-631: Trending suggestions
- Lines 637-662: Typo correction
- Lines 1115-1130: Cache management

## Phase 5: Implementation Timeline

### Week 1: Foundation Fixes
- Day 1-2: Fix all failing tests in test_100_percent_coverage.py
- Day 3-4: Fix authentication-related test failures
- Day 5: Fix remaining test failures

### Week 2: Coverage Gap Closure
- Day 1: Implement tests for metrics_collection.py
- Day 2: Implement tests for production_config.py
- Day 3: Implement tests for search_suggestions.py
- Day 4: Implement tests for query_expansion.py and reranker.py
- Day 5: Final coverage verification and cleanup

## Phase 6: Quality Assurance

### 6.1 Verification Steps
1. Run full test suite: `pytest tests/ --cov=app --cov-report=term-missing`
2. Verify no test failures
3. Confirm 100% coverage
4. Run code quality checks: `./scripts/check_code_quality.sh`

### 6.2 Documentation Updates
1. Update README with new testing instructions
2. Document any new test utilities or patterns
3. Create coverage maintenance guide

## Phase 7: Maintenance Strategy

### 7.1 CI/CD Integration
- Add coverage threshold check to CI pipeline
- Fail builds if coverage drops below 100%
- Generate coverage reports for each PR

### 7.2 Developer Guidelines
- Require tests for all new code
- Update tests when modifying existing code
- Regular coverage audits

## Appendix: Quick Reference

### Run Coverage Check
```bash
pytest tests/ --cov=app --cov-report=term-missing --cov-fail-under=100
```

### Generate HTML Coverage Report
```bash
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

### Check Specific Module Coverage
```bash
pytest tests/ --cov=app.services.metrics_collection --cov-report=term-missing
```

## Success Criteria
- [ ] All tests pass without failures
- [ ] Test coverage reaches 100%
- [ ] No code quality issues
- [ ] CI/CD pipeline updated with coverage checks
- [ ] Documentation updated

## Notes
- Focus on meaningful tests, not just line coverage
- Ensure tests are maintainable and clear
- Consider edge cases and error scenarios
- Mock external dependencies appropriately
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) system for system development documentation. The system integrates BGE-M3 hybrid search to provide high-precision search functionality to external systems.

**Core Tech Stack**: FastAPI + ApertureDB + PostgreSQL + BGE-M3 + Celery + Docker

## Common Commands

### Development Environment Setup
```bash
# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running the Application
```bash
# Start with Docker Compose (includes ApertureDB, PostgreSQL, Redis)
docker-compose up -d

# Run the FastAPI application locally
uvicorn app.main:app --reload

# Access the API documentation at http://localhost:8000/docs
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=term-missing

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Exclude slow tests

# Run specific test file
pytest tests/test_embedding_service.py
```

### Code Quality Checks
```bash
# Format code
black app/ tests/

# Lint code
ruff check app/ tests/

# Lint with auto-fix
ruff check --fix app/ tests/

# Type checking
mypy app/

# Security checks
safety check
bandit -r app/

# Run all quality checks (via pre-commit)
pre-commit run --all-files
```

### Database Operations
```bash
# Apply database migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "description"
```

## Architecture Overview

### Directory Structure
```
app/
├── api/           # FastAPI route handlers
│   ├── auth.py    # Authentication endpoints
│   ├── documents.py # Document management API
│   ├── health.py  # Health check endpoints
│   └── search.py  # Search API endpoints
├── core/          # Core application logic
│   ├── auth.py    # Authentication logic
│   ├── exceptions.py # Custom exceptions
│   └── middleware.py # FastAPI middleware
├── database/      # Database configuration
├── models/        # Data models
│   ├── database.py # SQLAlchemy models
│   └── aperturedb.py   # ApertureDB vector collections
├── repositories/  # Data access layer
├── services/      # Business logic
│   ├── embedding_service.py # BGE-M3 embedding service
│   └── embedding_tasks.py   # Celery background tasks
└── main.py        # FastAPI application factory
```

### Key Components

1. **FastAPI Application** (`app/main.py`):
   - Centralized error handling with structured JSON responses
   - CORS middleware configuration
   - Router registration for all API endpoints

2. **BGE-M3 Embedding Service** (`app/services/embedding_service.py`):
   - Supports Dense, Sparse, and Multi-Vector embeddings
   - Async processing with batch support
   - Automatic device detection (CPU/GPU)
   - Graceful fallback when embedding libraries are not available

3. **ApertureDB Vector Database** (`app/models/aperturedb.py`):
   - Abstract base class for vector collections
   - Separate descriptor sets for dense and sparse vectors
   - HNSW indexing for dense vectors with L2 metric

4. **Authentication System** (`app/api/auth.py`, `app/core/auth.py`):
   - JWT token-based authentication
   - API key authentication
   - Role-based access control (RBAC)

### Vector Search Architecture

The system implements hybrid search using BGE-M3:
- **Dense vectors**: 1024-dimensional semantic embeddings
- **Sparse vectors**: Traditional keyword-based sparse representations
- **Multi-vectors**: ColBERT-style multi-vector representations
- **Fusion**: Uses Reciprocal Rank Fusion (RRF) for combining results

## Development Guidelines

### Code Style
- Use Python 3.11+ features
- Follow Black formatting (line length: 88)
- Use type hints for all functions
- Write Google-style docstrings for public APIs
- Follow snake_case naming convention

### Testing Strategy
- Target ≥80% test coverage
- Use pytest markers for test categorization:
  - `@pytest.mark.unit` for unit tests
  - `@pytest.mark.integration` for integration tests
  - `@pytest.mark.slow` for time-consuming tests
  - Authentication-related markers for auth testing

### Error Handling
- Use custom exceptions defined in `app/core/exceptions.py`
- All API errors return structured JSON responses with:
  - `error.code`: Machine-readable error code
  - `error.message`: Human-readable message
  - `error.type`: Error category
  - `timestamp`: ISO format timestamp
  - `request_id`: Unique request identifier

### Environment Variables
Key environment variables for configuration:
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `ENVIRONMENT`: deployment environment (development/staging/production)
- `TESTING`: set to "true" to disable authentication middleware in tests

### Performance Considerations
- Vector search response time target: <500ms (95th percentile)
- Embedding processing: batch operations preferred
- Use async/await for I/O-bound operations
- GPU acceleration for embedding generation when available

### External Dependencies
The system integrates with several external services:
- **ApertureDB**: Vector database (port 55555)
- **PostgreSQL**: Metadata storage (port 5432)
- **Redis**: Caching and task queue (port 6379)
- **Embedding Models**: BGE-M3 via HuggingFace Transformers

## Cursor Rules Integration

This project follows detailed development rules defined in `.cursor/rules/`:
- Project overview and tech stack requirements
- TDD development process mandatory
- Performance targets and monitoring requirements
- Security and data protection standards

When making changes, ensure compliance with the established architecture patterns and quality standards defined in the Cursor rules.

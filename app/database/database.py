"""Database session management"""

from collections.abc import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

# Create engine
engine = create_engine(settings.DATABASE_URL)

# Create async engine for async operations
# Only create async engine if we're not using SQLite
if "sqlite" not in settings.DATABASE_URL:
    async_engine = create_async_engine(
        settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
        echo=False,
    )
    AsyncSessionLocal = async_sessionmaker(
        async_engine, 
        class_=AsyncSession,
        expire_on_commit=False,
    )
else:
    # For SQLite, we'll use sync sessions only
    async_engine = None
    AsyncSessionLocal = None

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Create base class for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session."""
    if AsyncSessionLocal is None:
        raise RuntimeError("Async database is not available for SQLite")
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

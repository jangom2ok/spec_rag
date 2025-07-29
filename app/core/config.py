"""Application configuration"""

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    PROJECT_NAME: str = "RAG System"
    VERSION: str = "1.0.0"
    PRODUCTION: bool = os.getenv("ENVIRONMENT", "development") == "production"

    # CORS settings
    ALLOWED_ORIGINS: list[str] = ["*"]

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://user:password@localhost/ragdb"
    )

    @property
    def database_url(self) -> str:
        """Get database URL for SQLAlchemy"""
        return self.DATABASE_URL

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # ApertureDB settings
    APERTUREDB_HOST: str = os.getenv("APERTUREDB_HOST", "localhost")
    APERTUREDB_PORT: int = int(os.getenv("APERTUREDB_PORT", "55555"))

    # Application settings
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-jwt-secret-key-here")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    )

    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields from .env file


settings = Settings()

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

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")

    class Config:
        env_file = ".env"


settings = Settings()

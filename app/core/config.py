from typing import List
import os
from pydantic import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Time Series Analysis API"
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://localhost:3000"
    ]
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    
    # Celery
    CELERY_BROKER_URL: str = os.getenv(
        "CELERY_BROKER_URL", "redis://localhost:6379/0"
    )
    CELERY_RESULT_BACKEND: str = os.getenv(
        "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
    )
    
    # Model Settings
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "prophet")
    MAX_FORECAST_PERIODS: int = int(os.getenv("MAX_FORECAST_PERIODS", 365))
    CACHE_EXPIRY: int = int(os.getenv("CACHE_EXPIRY", 3600))
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", 60))
    
    class Config:
        case_sensitive = True

settings = Settings()

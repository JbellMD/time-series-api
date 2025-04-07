from typing import List, Dict, Any, Optional
import os
from pydantic import BaseSettings, validator
import secrets


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Time Series Analysis API"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Security settings
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    CACHE_EXPIRY: int = 3600  # 1 hour
    
    # Celery settings
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Authentication settings
    USERS_DB: Dict[str, Dict[str, Any]] = {
        "admin": {
            "username": "admin",
            "email": "admin@example.com",
            "full_name": "Administrator",
            "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "password"
            "is_active": True,
            "scopes": ["admin", "read:forecasts", "write:forecasts", "read:technical", "write:technical", "read:statistical", "write:statistical"]
        },
        "user": {
            "username": "user",
            "email": "user@example.com",
            "full_name": "Regular User",
            "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "password"
            "is_active": True,
            "scopes": ["read:forecasts", "read:technical", "read:statistical"]
        }
    }
    
    # API Keys
    API_KEYS: List[str] = []
    API_KEY_TO_USER: Dict[str, str] = {}
    
    # Model settings
    DEFAULT_FORECAST_PERIODS: int = 30
    MAX_FORECAST_PERIODS: int = 365
    DEFAULT_CONFIDENCE_INTERVAL: float = 0.95
    
    # Deep learning model settings
    LSTM_LOOKBACK: int = 60
    LSTM_EPOCHS: int = 50
    LSTM_BATCH_SIZE: int = 32
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: str) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

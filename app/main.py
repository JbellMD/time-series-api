from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
import time
import redis
from typing import List, Dict, Any, Optional

from app.api.api_v1.api import api_router
from app.core.config import settings
from app.core.auth import get_current_user_from_token_or_api_key, verify_api_key

# Initialize Redis client for rate limiting
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    decode_responses=True
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    description="""
    # Time Series Analysis API

    This API provides endpoints for time series forecasting, technical analysis, and statistical analysis.

    ## Features

    * **Forecasting**: Generate forecasts using various models (Prophet, ARIMA, LSTM)
    * **Technical Analysis**: Calculate technical indicators (SMA, EMA, MACD, RSI, etc.)
    * **Statistical Analysis**: Perform statistical tests and analyses on time series data
    * **Authentication**: Secure your API with OAuth2 or API keys
    * **Caching**: Improve performance with Redis caching
    * **Rate Limiting**: Prevent abuse with rate limiting

    ## Documentation

    For detailed documentation, visit the [/docs](/docs) endpoint.
    """,
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    # Skip rate limiting for certain paths
    if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
    
    # Get client IP
    client_ip = request.client.host
    
    # Check if API key is provided
    api_key = request.headers.get("X-API-Key")
    if api_key and verify_api_key(api_key):
        # Use API key as rate limit key
        rate_limit_key = f"rate_limit:{api_key}"
    else:
        # Use client IP as rate limit key
        rate_limit_key = f"rate_limit:{client_ip}"
    
    # Check rate limit
    current_count = redis_client.get(rate_limit_key)
    if current_count is None:
        # First request, set count to 1 with expiry
        redis_client.set(rate_limit_key, 1, ex=60)
    elif int(current_count) >= settings.RATE_LIMIT_PER_MINUTE:
        # Rate limit exceeded
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Try again in a minute."},
            headers={
                "X-RateLimit-Limit": str(settings.RATE_LIMIT_PER_MINUTE),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + 60)
            }
        )
    else:
        # Increment count
        redis_client.incr(rate_limit_key)
        ttl = redis_client.ttl(rate_limit_key)
        if ttl < 0:
            # Reset expiry if TTL is negative
            redis_client.expire(rate_limit_key, 60)
    
    # Add rate limit headers
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_PER_MINUTE)
    response.headers["X-RateLimit-Remaining"] = str(
        max(0, settings.RATE_LIMIT_PER_MINUTE - int(redis_client.get(rate_limit_key) or 0))
    )
    response.headers["X-RateLimit-Reset"] = str(
        int(time.time()) + redis_client.ttl(rate_limit_key)
    )
    
    return response

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Custom OpenAPI and documentation endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI"""
    return get_swagger_ui_html(
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        title=f"{settings.PROJECT_NAME} - Swagger UI",
        oauth2_redirect_url=f"{settings.API_V1_STR}/oauth2-redirect",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """ReDoc UI"""
    return get_redoc_html(
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        title=f"{settings.PROJECT_NAME} - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """Custom OpenAPI schema"""
    return get_openapi(
        title=settings.PROJECT_NAME,
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": time.time()}

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Time Series Analysis API",
        "docs": "/docs",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

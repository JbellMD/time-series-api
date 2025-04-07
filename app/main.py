from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import redis
from typing import Callable

from app.core.config import settings
from app.api.api_v1.api import api_router

# Initialize Redis for rate limiting and caching
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    decode_responses=True
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set up CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next: Callable):
    # Get client IP
    client_ip = request.client.host
    
    # Check rate limit
    request_count = redis_client.get(f"rate_limit:{client_ip}")
    
    if request_count is not None and int(request_count) > settings.RATE_LIMIT_PER_MINUTE:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    
    # Increment request count
    redis_client.incr(f"rate_limit:{client_ip}")
    redis_client.expire(f"rate_limit:{client_ip}", 60)  # Reset after 1 minute
    
    # Process request
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Time Series Analysis API",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    # Check Redis connection
    try:
        redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "api": "healthy",
            "redis": redis_status
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

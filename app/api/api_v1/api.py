from fastapi import APIRouter

from app.api.api_v1.endpoints import forecasting, technical, statistical

api_router = APIRouter()
api_router.include_router(forecasting.router, prefix="/forecast", tags=["forecasting"])
api_router.include_router(technical.router, prefix="/technical", tags=["technical"])
api_router.include_router(statistical.router, prefix="/statistical", tags=["statistical"])

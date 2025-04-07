from datetime import timedelta
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordRequestForm, APIKeyHeader
from pydantic import BaseModel

from app.core.auth import (
    authenticate_user, create_access_token, 
    get_current_active_user, verify_api_key,
    get_api_key_user
)
from app.core.config import settings
from app.models.user import Token, User, APIKeyCreate, APIKeyResponse

router = APIRouter()

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user = authenticate_user(settings.USERS_DB, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if requested scopes are valid for this user
    for scope in form_data.scopes:
        if scope not in user.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have access to scope: {scope}",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": form_data.scopes},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current user
    """
    return current_user

@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    api_key_create: APIKeyCreate,
    current_user: User = Security(get_current_active_user, scopes=["admin"])
):
    """
    Create a new API key (admin only)
    """
    # Check if requested scopes are valid for this user
    for scope in api_key_create.scopes:
        if scope not in current_user.scopes and "admin" not in current_user.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Cannot create API key with scope: {scope}",
            )
    
    # Generate a new API key
    import secrets
    import time
    
    api_key = f"ts_api_{secrets.token_hex(16)}_{int(time.time())}"
    
    # Store the API key (in a real app, this would be in a database)
    settings.API_KEYS.append(api_key)
    settings.API_KEY_TO_USER[api_key] = current_user.username
    
    # Return the API key
    return {
        "key": api_key,
        "name": api_key_create.name,
        "scopes": api_key_create.scopes,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    }

@router.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    current_user: User = Security(get_current_active_user, scopes=["admin"])
):
    """
    List all API keys (admin only)
    """
    # In a real app, this would fetch from a database
    return [
        {
            "key": key,
            "name": "API Key",
            "scopes": current_user.scopes,
            "created_at": "2023-01-01 00:00:00"
        }
        for key in settings.API_KEYS
        if settings.API_KEY_TO_USER.get(key) == current_user.username
    ]

@router.delete("/api-keys/{api_key}")
async def delete_api_key(
    api_key: str,
    current_user: User = Security(get_current_active_user, scopes=["admin"])
):
    """
    Delete an API key (admin only)
    """
    if api_key in settings.API_KEYS and settings.API_KEY_TO_USER.get(api_key) == current_user.username:
        settings.API_KEYS.remove(api_key)
        del settings.API_KEY_TO_USER[api_key]
        return {"message": "API key deleted"}
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="API key not found",
    )

async def get_current_user_from_token_or_api_key(
    api_key: str = Security(api_key_header),
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Get the current user from either a token or an API key
    """
    if current_user:
        return current_user
    
    if api_key and verify_api_key(api_key):
        api_key_user = get_api_key_user(api_key)
        if api_key_user:
            return api_key_user
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

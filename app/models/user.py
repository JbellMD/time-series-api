from typing import List, Optional
from pydantic import BaseModel, EmailStr


class Token(BaseModel):
    """Token model"""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token data model"""
    username: Optional[str] = None
    scopes: List[str] = []


class User(BaseModel):
    """User model"""
    username: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: bool = True
    scopes: List[str] = []


class UserInDB(User):
    """User in database model"""
    hashed_password: str


class APIKeyCreate(BaseModel):
    """API key creation model"""
    name: str
    scopes: List[str]


class APIKeyResponse(BaseModel):
    """API key response model"""
    key: str
    name: str
    scopes: List[str]
    created_at: str

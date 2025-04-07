from datetime import datetime, timedelta
from typing import Optional, Union, Any, Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.models.user import User, UserInDB, TokenData

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/token",
    scopes={
        "read:forecasts": "Read forecasts",
        "write:forecasts": "Create forecasts",
        "read:technical": "Read technical indicators",
        "write:technical": "Create technical indicators",
        "read:statistical": "Read statistical analyses",
        "write:statistical": "Create statistical analyses",
        "admin": "Admin access"
    }
)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def authenticate_user(fake_db: Dict[str, UserInDB], username: str, password: str) -> Union[UserInDB, bool]:
    """Authenticate a user"""
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def get_user(fake_db: Dict[str, UserInDB], username: str) -> Optional[UserInDB]:
    """Get a user from the database"""
    if username in fake_db:
        user_dict = fake_db[username]
        return UserInDB(**user_dict)
    return None

async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme)
) -> User:
    """Get the current user from the token"""
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(scopes=token_scopes, username=username)
    except (JWTError, ValidationError):
        raise credentials_exception
    
    user = get_user(settings.USERS_DB, username=token_data.username)
    if user is None:
        raise credentials_exception
    
    # Check if the user has the required scopes
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get the current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# API Key authentication
class APIKey(BaseModel):
    """API Key model"""
    key: str

def verify_api_key(api_key: str) -> bool:
    """Verify an API key"""
    return api_key in settings.API_KEYS

def get_api_key_user(api_key: str) -> Optional[User]:
    """Get a user from an API key"""
    if api_key in settings.API_KEY_TO_USER:
        username = settings.API_KEY_TO_USER[api_key]
        user_dict = settings.USERS_DB.get(username)
        if user_dict:
            return User(**user_dict)
    return None

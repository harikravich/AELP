"""
Authentication and authorization for the agent manager API
"""
import os
import logging
from typing import Optional
from datetime import datetime, timedelta

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


class User(BaseModel):
    """User model for authentication"""
    username: str
    email: Optional[str] = None
    is_admin: bool = False
    is_active: bool = True


class TokenData(BaseModel):
    """Token data model"""
    username: Optional[str] = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[str]:
    """Verify JWT token and return username"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify token
        username = verify_token(credentials.credentials)
        if username is None:
            raise credentials_exception
        
        # In a real implementation, you would fetch user from database
        # For now, we'll create a mock user based on token
        user = await get_user_by_username(username)
        if user is None:
            raise credentials_exception
        
        return user
        
    except JWTError:
        raise credentials_exception


async def get_user_by_username(username: str) -> Optional[User]:
    """Get user by username - mock implementation"""
    # In a real implementation, this would query your user database
    # For demo purposes, we'll create mock users
    
    mock_users = {
        "admin": User(username="admin", email="admin@gaelp.com", is_admin=True),
        "user1": User(username="user1", email="user1@gaelp.com", is_admin=False),
        "researcher": User(username="researcher", email="researcher@gaelp.com", is_admin=False),
    }
    
    return mock_users.get(username)


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin privileges"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


# Mock authentication for development/testing
class MockAuth:
    """Mock authentication for development"""
    
    @staticmethod
    def create_mock_token(username: str, is_admin: bool = False) -> str:
        """Create a mock token for testing"""
        return create_access_token(
            data={
                "sub": username,
                "admin": is_admin
            }
        )


# Development helper
if os.getenv("ENVIRONMENT") == "development":
    logger.warning("Running in development mode with mock authentication")
    
    # Create some test tokens
    admin_token = MockAuth.create_mock_token("admin", is_admin=True)
    user_token = MockAuth.create_mock_token("user1", is_admin=False)
    
    logger.info(f"Test admin token: {admin_token}")
    logger.info(f"Test user token: {user_token}")
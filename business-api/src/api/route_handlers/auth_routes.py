"""Auth routes for Business API."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ..auth import create_access_token

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login", tags=["Auth"])
async def login(request: LoginRequest):
    """Simple username/password login for internal use. Returns a JWT token."""
    valid_users = {"admin": "admin123", "test": "test123"}
    if request.username not in valid_users or valid_users[request.username] != request.password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token({"user_id": request.username, "role": "admin"})
    return {"access_token": token, "token_type": "bearer"}

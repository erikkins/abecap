"""Push notification API endpoints for mobile app."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, User
from app.core.security import get_current_user
from app.services.push_notification_service import push_notification_service

router = APIRouter()


class RegisterTokenRequest(BaseModel):
    token: str
    platform: str  # "ios" or "android"
    device_id: str | None = None


class UnregisterTokenRequest(BaseModel):
    token: str


@router.post("/register")
async def register_push_token(
    req: RegisterTokenRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Register an Expo push token for the authenticated user."""
    if req.platform not in ("ios", "android"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="platform must be 'ios' or 'android'",
        )
    if not req.token or len(req.token) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid push token",
        )

    push_token = await push_notification_service.register_token(
        db,
        user_id=str(current_user.id),
        token=req.token,
        platform=req.platform,
        device_id=req.device_id,
    )
    return {"status": "ok", "token_id": push_token.id}


@router.delete("/unregister")
async def unregister_push_token(
    req: UnregisterTokenRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Deactivate a push token (e.g. on logout)."""
    removed = await push_notification_service.unregister_token(db, req.token)
    return {"status": "ok", "removed": removed}


@router.get("/status")
async def push_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Check if the user has active push tokens."""
    has_tokens = await push_notification_service.has_active_tokens(
        db, str(current_user.id)
    )
    return {"has_active_tokens": has_tokens}

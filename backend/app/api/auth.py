"""Authentication API endpoints."""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db, User, Subscription
from app.core.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
)
from app.services.turnstile import verify_turnstile
from app.services.email_service import email_service

router = APIRouter()


# Request/Response schemas
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None
    turnstile_token: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict


class RefreshRequest(BaseModel):
    refresh_token: str


class GoogleAuthRequest(BaseModel):
    id_token: str
    turnstile_token: Optional[str] = None


class AppleAuthRequest(BaseModel):
    id_token: str
    user_data: Optional[dict] = None
    turnstile_token: Optional[str] = None


@router.post("/register", response_model=TokenResponse)
async def register(
    request: RegisterRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user with email/password."""
    # Verify Turnstile
    client_ip = req.client.host if req.client else None
    if not await verify_turnstile(request.turnstile_token, client_ip):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bot verification failed"
        )

    # Check if email already exists
    result = await db.execute(select(User).where(User.email == request.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Validate password
    if len(request.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters"
        )

    # Create user
    user = User(
        email=request.email,
        password_hash=get_password_hash(request.password),
        name=request.name,
        role="admin" if request.email == "erik@rigacap.com" else "user",
    )
    db.add(user)
    await db.flush()

    # Create trial subscription
    subscription = Subscription.create_trial(user.id)
    db.add(subscription)

    await db.commit()
    await db.refresh(user)

    # Generate tokens
    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))

    # Send welcome email (don't wait for it, fire and forget)
    import asyncio
    asyncio.create_task(
        email_service.send_welcome_email(user.email, user.name or user.email)
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user.to_dict()
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """Login with email/password."""
    # Find user
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if not user or not user.password_hash:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )

    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()
    await db.refresh(user)

    # Load subscription
    result = await db.execute(
        select(Subscription).where(Subscription.user_id == user.id)
    )
    subscription = result.scalar_one_or_none()

    user_dict = user.to_dict()
    if subscription:
        user_dict["subscription"] = subscription.to_dict()

    # Generate tokens
    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user_dict
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshRequest,
    db: AsyncSession = Depends(get_db)
):
    """Refresh access token using refresh token."""
    payload = decode_token(request.refresh_token)

    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    user_id = payload.get("sub")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled"
        )

    # Load subscription
    result = await db.execute(
        select(Subscription).where(Subscription.user_id == user.id)
    )
    subscription = result.scalar_one_or_none()

    user_dict = user.to_dict()
    if subscription:
        user_dict["subscription"] = subscription.to_dict()

    # Generate new tokens
    access_token = create_access_token(str(user.id))
    new_refresh_token = create_refresh_token(str(user.id))

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        user=user_dict
    )


@router.get("/me")
async def get_me(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current authenticated user."""
    # Load subscription
    result = await db.execute(
        select(Subscription).where(Subscription.user_id == user.id)
    )
    subscription = result.scalar_one_or_none()

    user_dict = user.to_dict()
    if subscription:
        user_dict["subscription"] = subscription.to_dict()

    return user_dict


@router.post("/logout")
async def logout(user: User = Depends(get_current_user)):
    """Logout current user (client should delete tokens)."""
    return {"message": "Logged out successfully"}


@router.post("/google", response_model=TokenResponse)
async def google_auth(
    request: GoogleAuthRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """Authenticate with Google OAuth."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={request.id_token}"
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid Google token"
                )
            google_data = response.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to verify Google token"
        )

    email = google_data.get("email")
    google_id = google_data.get("sub")
    name = google_data.get("name")

    if not email or not google_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token data"
        )

    # Check if user exists by Google ID or email
    result = await db.execute(
        select(User).where((User.google_id == google_id) | (User.email == email))
    )
    user = result.scalar_one_or_none()

    if user:
        # Update Google ID if not set
        if not user.google_id:
            user.google_id = google_id
        user.last_login = datetime.utcnow()
    else:
        # Verify Turnstile for new users
        if request.turnstile_token:
            client_ip = req.client.host if req.client else None
            if not await verify_turnstile(request.turnstile_token, client_ip):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Bot verification failed"
                )

        # Create new user
        user = User(
            email=email,
            name=name,
            google_id=google_id,
            role="admin" if email == "erik@rigacap.com" else "user",
        )
        db.add(user)
        await db.flush()

        # Create trial subscription
        subscription = Subscription.create_trial(user.id)
        db.add(subscription)

    await db.commit()
    await db.refresh(user)

    # Load subscription
    result = await db.execute(
        select(Subscription).where(Subscription.user_id == user.id)
    )
    subscription = result.scalar_one_or_none()

    user_dict = user.to_dict()
    if subscription:
        user_dict["subscription"] = subscription.to_dict()

    # Generate tokens
    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user_dict
    )


@router.post("/apple", response_model=TokenResponse)
async def apple_auth(
    request: AppleAuthRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """Authenticate with Apple Sign In."""
    from jose import jwt

    try:
        # Decode without verification (client already verified)
        unverified = jwt.get_unverified_claims(request.id_token)
        apple_id = unverified.get("sub")
        email = unverified.get("email")

        # Apple only sends email on first auth, use user_data if provided
        if not email and request.user_data:
            email = request.user_data.get("email")
            name = request.user_data.get("name", {})
            full_name = f"{name.get('firstName', '')} {name.get('lastName', '')}".strip()
        else:
            full_name = None
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Apple token"
        )

    if not apple_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token data"
        )

    # Check if user exists by Apple ID
    result = await db.execute(select(User).where(User.apple_id == apple_id))
    user = result.scalar_one_or_none()

    # Also check by email if we have one
    if not user and email:
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

    if user:
        # Update Apple ID if not set
        if not user.apple_id:
            user.apple_id = apple_id
        user.last_login = datetime.utcnow()
    else:
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email required for new accounts"
            )

        # Verify Turnstile for new users
        if request.turnstile_token:
            client_ip = req.client.host if req.client else None
            if not await verify_turnstile(request.turnstile_token, client_ip):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Bot verification failed"
                )

        # Create new user
        user = User(
            email=email,
            name=full_name,
            apple_id=apple_id,
            role="admin" if email == "erik@rigacap.com" else "user",
        )
        db.add(user)
        await db.flush()

        # Create trial subscription
        subscription = Subscription.create_trial(user.id)
        db.add(subscription)

    await db.commit()
    await db.refresh(user)

    # Load subscription
    result = await db.execute(
        select(Subscription).where(Subscription.user_id == user.id)
    )
    subscription = result.scalar_one_or_none()

    user_dict = user.to_dict()
    if subscription:
        user_dict["subscription"] = subscription.to_dict()

    # Generate tokens
    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user_dict
    )

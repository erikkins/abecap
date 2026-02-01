"""Admin API endpoints for user management and service monitoring."""

from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.core.database import get_db, User, Subscription, Signal
from app.core.security import get_admin_user
from app.core.config import settings

router = APIRouter()


# Response schemas
class UserSummary(BaseModel):
    id: str
    email: str
    name: Optional[str]
    role: str
    is_active: bool
    created_at: str
    last_login: Optional[str]
    subscription_status: Optional[str]
    trial_days_remaining: Optional[int]


class UserListResponse(BaseModel):
    users: List[UserSummary]
    total: int
    page: int
    per_page: int


class ServiceStatus(BaseModel):
    status: str
    latency_ms: Optional[float] = None
    last_check: Optional[str] = None
    error: Optional[str] = None


class AdminStatsResponse(BaseModel):
    total_users: int
    active_trials: int
    paid_subscribers: int
    expired_trials: int
    disabled_users: int
    new_users_today: int
    new_users_week: int
    mrr: float  # Monthly recurring revenue


class ServiceStatusResponse(BaseModel):
    overall_status: str
    services: dict
    metrics: dict


@router.get("/users", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    status_filter: Optional[str] = None,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """List all users with pagination and filtering."""
    # Base query
    query = select(User).order_by(User.created_at.desc())

    # Apply search filter
    if search:
        search_term = f"%{search}%"
        query = query.where(
            (User.email.ilike(search_term)) | (User.name.ilike(search_term))
        )

    # Get total count
    count_query = select(func.count(User.id))
    if search:
        search_term = f"%{search}%"
        count_query = count_query.where(
            (User.email.ilike(search_term)) | (User.name.ilike(search_term))
        )
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Apply pagination
    offset = (page - 1) * per_page
    query = query.offset(offset).limit(per_page)

    # Execute query
    result = await db.execute(query)
    users = result.scalars().all()

    # Get subscription status for each user
    user_summaries = []
    for user in users:
        sub_result = await db.execute(
            select(Subscription).where(Subscription.user_id == user.id)
        )
        subscription = sub_result.scalar_one_or_none()

        user_summaries.append(UserSummary(
            id=str(user.id),
            email=user.email,
            name=user.name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at.isoformat() if user.created_at else None,
            last_login=user.last_login.isoformat() if user.last_login else None,
            subscription_status=subscription.status if subscription else None,
            trial_days_remaining=subscription.days_remaining() if subscription else None
        ))

    return UserListResponse(
        users=user_summaries,
        total=total,
        page=page,
        per_page=per_page
    )


@router.get("/users/{user_id}")
async def get_user_details(
    user_id: str,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed information about a specific user."""
    import uuid

    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get subscription
    sub_result = await db.execute(
        select(Subscription).where(Subscription.user_id == user.id)
    )
    subscription = sub_result.scalar_one_or_none()

    user_dict = user.to_dict()
    if subscription:
        user_dict["subscription"] = subscription.to_dict()

    return user_dict


@router.post("/users/{user_id}/disable")
async def disable_user(
    user_id: str,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Disable a user account."""
    import uuid

    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot disable your own account")

    user.is_active = False
    await db.commit()

    return {"message": f"User {user.email} has been disabled"}


@router.post("/users/{user_id}/enable")
async def enable_user(
    user_id: str,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Enable a user account."""
    import uuid

    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_active = True
    await db.commit()

    return {"message": f"User {user.email} has been enabled"}


@router.post("/users/{user_id}/extend-trial")
async def extend_trial(
    user_id: str,
    days: int = Query(7, ge=1, le=90),
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Extend a user's trial period."""
    import uuid

    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get subscription
    sub_result = await db.execute(
        select(Subscription).where(Subscription.user_id == user.id)
    )
    subscription = sub_result.scalar_one_or_none()

    if not subscription:
        raise HTTPException(status_code=400, detail="User has no subscription")

    if subscription.status not in ["trial", "expired"]:
        raise HTTPException(status_code=400, detail="Can only extend trial subscriptions")

    # Extend trial
    if subscription.trial_end:
        subscription.trial_end = subscription.trial_end + timedelta(days=days)
    else:
        subscription.trial_end = datetime.utcnow() + timedelta(days=days)

    subscription.status = "trial"
    await db.commit()

    return {
        "message": f"Trial extended by {days} days",
        "new_trial_end": subscription.trial_end.isoformat()
    }


@router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Get admin dashboard statistics."""
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = now - timedelta(days=7)

    # Total users
    total_result = await db.execute(select(func.count(User.id)))
    total_users = total_result.scalar()

    # Active trials
    trials_result = await db.execute(
        select(func.count(Subscription.id)).where(
            and_(
                Subscription.status == "trial",
                Subscription.trial_end > now
            )
        )
    )
    active_trials = trials_result.scalar()

    # Paid subscribers
    paid_result = await db.execute(
        select(func.count(Subscription.id)).where(Subscription.status == "active")
    )
    paid_subscribers = paid_result.scalar()

    # Expired trials
    expired_result = await db.execute(
        select(func.count(Subscription.id)).where(
            and_(
                Subscription.status == "trial",
                Subscription.trial_end <= now
            )
        )
    )
    expired_trials = expired_result.scalar()

    # Disabled users
    disabled_result = await db.execute(
        select(func.count(User.id)).where(User.is_active == False)
    )
    disabled_users = disabled_result.scalar()

    # New users today
    today_result = await db.execute(
        select(func.count(User.id)).where(User.created_at >= today_start)
    )
    new_users_today = today_result.scalar()

    # New users this week
    week_result = await db.execute(
        select(func.count(User.id)).where(User.created_at >= week_ago)
    )
    new_users_week = week_result.scalar()

    # MRR (Monthly Recurring Revenue) - $10 per subscriber
    mrr = paid_subscribers * 10.0

    return AdminStatsResponse(
        total_users=total_users,
        active_trials=active_trials,
        paid_subscribers=paid_subscribers,
        expired_trials=expired_trials,
        disabled_users=disabled_users,
        new_users_today=new_users_today,
        new_users_week=new_users_week,
        mrr=mrr
    )


@router.get("/service-status", response_model=ServiceStatusResponse)
async def get_service_status(
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Get service health status."""
    import time

    services = {}
    overall_healthy = True

    # Database check
    try:
        start = time.time()
        await db.execute(select(func.count(User.id)))
        latency = (time.time() - start) * 1000
        services["database"] = ServiceStatus(
            status="ok",
            latency_ms=round(latency, 2)
        ).model_dump()
    except Exception as e:
        services["database"] = ServiceStatus(
            status="error",
            error=str(e)
        ).model_dump()
        overall_healthy = False

    # yfinance check
    try:
        from app.services.scanner import scanner_service
        yf_status = "ok" if scanner_service.data_cache else "no_data"
        symbols_loaded = len(scanner_service.data_cache)
        services["yfinance"] = {
            "status": yf_status,
            "symbols_loaded": symbols_loaded,
            "last_fetch": scanner_service.last_fetch_time.isoformat() if hasattr(scanner_service, 'last_fetch_time') and scanner_service.last_fetch_time else None
        }
    except Exception as e:
        services["yfinance"] = {
            "status": "error",
            "error": str(e)
        }

    # Stripe check
    if settings.STRIPE_SECRET_KEY:
        try:
            import stripe
            stripe.api_key = settings.STRIPE_SECRET_KEY
            stripe.Account.retrieve()
            services["stripe"] = {"status": "ok"}
        except Exception as e:
            services["stripe"] = {
                "status": "error",
                "error": str(e)
            }
    else:
        services["stripe"] = {"status": "not_configured"}

    # Scanner status
    try:
        from app.services.scanner import scanner_service

        # Count today's signals
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        signals_result = await db.execute(
            select(func.count(Signal.id)).where(Signal.created_at >= today_start)
        )
        signals_today = signals_result.scalar()

        services["scanner"] = {
            "status": "ok",
            "signals_today": signals_today,
            "cached_symbols": len(scanner_service.data_cache)
        }
    except Exception as e:
        services["scanner"] = {
            "status": "error",
            "error": str(e)
        }

    # Get metrics
    stats_result = await db.execute(select(func.count(User.id)))
    total_users = stats_result.scalar()

    paid_result = await db.execute(
        select(func.count(Subscription.id)).where(Subscription.status == "active")
    )
    paid_subscribers = paid_result.scalar()

    trials_result = await db.execute(
        select(func.count(Subscription.id)).where(
            and_(
                Subscription.status == "trial",
                Subscription.trial_end > datetime.utcnow()
            )
        )
    )
    active_trials = trials_result.scalar()

    metrics = {
        "total_users": total_users,
        "active_trials": active_trials,
        "paid_subscribers": paid_subscribers,
        "mrr": paid_subscribers * 10.0
    }

    return ServiceStatusResponse(
        overall_status="healthy" if overall_healthy else "degraded",
        services=services,
        metrics=metrics
    )

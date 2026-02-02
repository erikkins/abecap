"""Admin API endpoints for user management and service monitoring."""

import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc

from app.core.database import get_db, User, Subscription, Signal, Position, StrategyDefinition, StrategyEvaluation
from app.core.security import get_admin_user
from app.core.config import settings
from app.services.scheduler import scheduler_service
from app.services.strategy_analyzer import strategy_analyzer_service

router = APIRouter()


# ============================================================================
# Initial Strategy Definitions
# ============================================================================

INITIAL_STRATEGIES = [
    {
        "name": "DWAP Classic",
        "description": "Buy when price > DWAP x 1.05, fixed 8% stop loss, 20% profit target",
        "strategy_type": "dwap",
        "parameters": {
            "dwap_threshold_pct": 5.0,
            "stop_loss_pct": 8.0,
            "profit_target_pct": 20.0,
            "max_positions": 15,
            "position_size_pct": 6.6,
            "min_volume": 500000,
            "min_price": 20.0,
            "volume_spike_mult": 1.5
        },
        "is_active": False
    },
    {
        "name": "Momentum v2",
        "description": "10/60 day momentum ranking, 15% trailing stop, SPY market filter",
        "strategy_type": "momentum",
        "parameters": {
            "max_positions": 5,
            "position_size_pct": 18.0,
            "short_momentum_days": 10,
            "long_momentum_days": 60,
            "trailing_stop_pct": 15.0,
            "market_filter_enabled": True,
            "rebalance_frequency": "weekly",
            "short_mom_weight": 0.5,
            "long_mom_weight": 0.3,
            "volatility_penalty": 0.2,
            "near_50d_high_pct": 5.0,
            "min_volume": 500000,
            "min_price": 20.0
        },
        "is_active": True
    }
]


async def seed_strategies(db: AsyncSession) -> int:
    """Seed initial strategies if none exist. Returns count of created strategies."""
    result = await db.execute(select(func.count(StrategyDefinition.id)))
    count = result.scalar()

    if count > 0:
        return 0  # Already seeded

    created = 0
    for strat_data in INITIAL_STRATEGIES:
        strategy = StrategyDefinition(
            name=strat_data["name"],
            description=strat_data["description"],
            strategy_type=strat_data["strategy_type"],
            parameters=json.dumps(strat_data["parameters"]),
            is_active=strat_data["is_active"],
            activated_at=datetime.utcnow() if strat_data["is_active"] else None
        )
        db.add(strategy)
        created += 1

    await db.commit()
    return created


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


# Strategy Management Schemas
class EvaluationSummary(BaseModel):
    date: Optional[str] = None
    sharpe_ratio: Optional[float] = None
    total_return_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    win_rate: Optional[float] = None
    recommendation_score: Optional[float] = None


class StrategyResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    strategy_type: str
    parameters: Dict[str, Any]
    is_active: bool
    created_at: Optional[str] = None
    activated_at: Optional[str] = None
    latest_evaluation: Optional[EvaluationSummary] = None


class StrategyEvaluationDetail(BaseModel):
    strategy_id: int
    name: str
    strategy_type: str
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    recommendation_score: float
    lookback_days: int


class StrategyAnalysisResponse(BaseModel):
    analysis_date: str
    lookback_days: int
    evaluations: List[StrategyEvaluationDetail]
    recommended_strategy_id: Optional[int] = None
    recommendation_notes: str
    current_active_strategy_id: Optional[int] = None


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


@router.post("/ticker-health-check")
async def run_ticker_health_check(
    admin: User = Depends(get_admin_user)
):
    """
    Manually trigger the ticker health check.

    Checks:
    - All open positions for valid data
    - Must-include symbols from stock universe

    Sends alert email if issues found.
    """
    try:
        await scheduler_service.check_ticker_health()
        return {
            "status": "completed",
            "message": "Health check completed. Alert email sent if issues found."
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/ticker-health-check")
async def get_ticker_health(
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current ticker health status without sending alerts.

    Returns list of any problematic tickers.
    """
    from app.services.scanner import scanner_service
    from app.services.stock_universe import MUST_INCLUDE
    import yfinance as yf

    issues = []

    # Check open positions
    result = await db.execute(
        select(Position).where(Position.status == "open")
    )
    positions = result.scalars().all()

    for pos in positions:
        symbol = pos.symbol

        # Check if in cache with recent data
        has_data = False
        if symbol in scanner_service.data_cache:
            df = scanner_service.data_cache[symbol]
            if not df.empty:
                last_date = df.index[-1]
                days_old = (datetime.utcnow() - last_date.to_pydatetime().replace(tzinfo=None)).days
                if days_old <= 7:
                    has_data = True

        if not has_data:
            # Try yfinance
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="5d")
                if hist.empty:
                    issues.append({
                        "symbol": symbol,
                        "type": "position",
                        "issue": "No data available",
                        "entry_date": pos.entry_date.strftime('%Y-%m-%d') if pos.entry_date else None,
                        "entry_price": pos.entry_price
                    })
            except Exception as e:
                issues.append({
                    "symbol": symbol,
                    "type": "position",
                    "issue": f"Fetch failed: {str(e)[:50]}",
                    "entry_date": pos.entry_date.strftime('%Y-%m-%d') if pos.entry_date else None,
                    "entry_price": pos.entry_price
                })

    # Check sample of must-includes
    must_check = MUST_INCLUDE[:10]
    for symbol in must_check:
        if any(i['symbol'] == symbol for i in issues):
            continue

        has_data = symbol in scanner_service.data_cache and not scanner_service.data_cache[symbol].empty
        if not has_data:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="5d")
                if hist.empty:
                    issues.append({
                        "symbol": symbol,
                        "type": "must_include",
                        "issue": "No data available"
                    })
            except Exception as e:
                issues.append({
                    "symbol": symbol,
                    "type": "must_include",
                    "issue": f"Fetch failed: {str(e)[:50]}"
                })

    return {
        "status": "healthy" if not issues else "issues_found",
        "positions_checked": len(positions),
        "must_includes_checked": len(must_check),
        "issues_count": len(issues),
        "issues": issues
    }


# ============================================================================
# Strategy Management Endpoints
# ============================================================================

@router.get("/strategies", response_model=List[StrategyResponse])
async def list_strategies(
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all trading strategies with their latest evaluation.

    Returns all strategies in the library with performance metrics
    from the most recent evaluation.
    """
    # Seed strategies if needed
    seeded = await seed_strategies(db)
    if seeded > 0:
        print(f"Seeded {seeded} initial strategies")

    # Get all strategies
    result = await db.execute(
        select(StrategyDefinition).order_by(StrategyDefinition.id)
    )
    strategies = result.scalars().all()

    responses = []
    for strat in strategies:
        # Get latest evaluation
        eval_result = await db.execute(
            select(StrategyEvaluation)
            .where(StrategyEvaluation.strategy_id == strat.id)
            .order_by(desc(StrategyEvaluation.evaluation_date))
            .limit(1)
        )
        latest_eval = eval_result.scalar_one_or_none()

        latest_eval_summary = None
        if latest_eval:
            latest_eval_summary = EvaluationSummary(
                date=latest_eval.evaluation_date.isoformat() if latest_eval.evaluation_date else None,
                sharpe_ratio=latest_eval.sharpe_ratio,
                total_return_pct=latest_eval.total_return_pct,
                max_drawdown_pct=latest_eval.max_drawdown_pct,
                win_rate=latest_eval.win_rate,
                recommendation_score=latest_eval.recommendation_score
            )

        responses.append(StrategyResponse(
            id=strat.id,
            name=strat.name,
            description=strat.description,
            strategy_type=strat.strategy_type,
            parameters=json.loads(strat.parameters),
            is_active=strat.is_active,
            created_at=strat.created_at.isoformat() if strat.created_at else None,
            activated_at=strat.activated_at.isoformat() if strat.activated_at else None,
            latest_evaluation=latest_eval_summary
        ))

    return responses


@router.get("/strategies/active", response_model=StrategyResponse)
async def get_active_strategy(
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Get the currently active trading strategy."""
    # Seed strategies if needed
    await seed_strategies(db)

    result = await db.execute(
        select(StrategyDefinition).where(StrategyDefinition.is_active == True)
    )
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(status_code=404, detail="No active strategy found")

    # Get latest evaluation
    eval_result = await db.execute(
        select(StrategyEvaluation)
        .where(StrategyEvaluation.strategy_id == strategy.id)
        .order_by(desc(StrategyEvaluation.evaluation_date))
        .limit(1)
    )
    latest_eval = eval_result.scalar_one_or_none()

    latest_eval_summary = None
    if latest_eval:
        latest_eval_summary = EvaluationSummary(
            date=latest_eval.evaluation_date.isoformat() if latest_eval.evaluation_date else None,
            sharpe_ratio=latest_eval.sharpe_ratio,
            total_return_pct=latest_eval.total_return_pct,
            max_drawdown_pct=latest_eval.max_drawdown_pct,
            win_rate=latest_eval.win_rate,
            recommendation_score=latest_eval.recommendation_score
        )

    return StrategyResponse(
        id=strategy.id,
        name=strategy.name,
        description=strategy.description,
        strategy_type=strategy.strategy_type,
        parameters=json.loads(strategy.parameters),
        is_active=strategy.is_active,
        created_at=strategy.created_at.isoformat() if strategy.created_at else None,
        activated_at=strategy.activated_at.isoformat() if strategy.activated_at else None,
        latest_evaluation=latest_eval_summary
    )


@router.post("/strategies/{strategy_id}/activate", response_model=StrategyResponse)
async def activate_strategy(
    strategy_id: int,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Activate a strategy (deactivates current active strategy).

    Only one strategy can be active at a time.
    """
    # Find the strategy to activate
    result = await db.execute(
        select(StrategyDefinition).where(StrategyDefinition.id == strategy_id)
    )
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Deactivate all other strategies
    all_strategies = await db.execute(select(StrategyDefinition))
    for strat in all_strategies.scalars():
        strat.is_active = False

    # Activate the selected strategy
    strategy.is_active = True
    strategy.activated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(strategy)

    # Get latest evaluation
    eval_result = await db.execute(
        select(StrategyEvaluation)
        .where(StrategyEvaluation.strategy_id == strategy.id)
        .order_by(desc(StrategyEvaluation.evaluation_date))
        .limit(1)
    )
    latest_eval = eval_result.scalar_one_or_none()

    latest_eval_summary = None
    if latest_eval:
        latest_eval_summary = EvaluationSummary(
            date=latest_eval.evaluation_date.isoformat() if latest_eval.evaluation_date else None,
            sharpe_ratio=latest_eval.sharpe_ratio,
            total_return_pct=latest_eval.total_return_pct,
            max_drawdown_pct=latest_eval.max_drawdown_pct,
            win_rate=latest_eval.win_rate,
            recommendation_score=latest_eval.recommendation_score
        )

    return StrategyResponse(
        id=strategy.id,
        name=strategy.name,
        description=strategy.description,
        strategy_type=strategy.strategy_type,
        parameters=json.loads(strategy.parameters),
        is_active=strategy.is_active,
        created_at=strategy.created_at.isoformat() if strategy.created_at else None,
        activated_at=strategy.activated_at.isoformat() if strategy.activated_at else None,
        latest_evaluation=latest_eval_summary
    )


@router.post("/strategies/analyze", response_model=StrategyAnalysisResponse)
async def run_strategy_analysis(
    lookback_days: int = Query(90, ge=30, le=365),
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Run backtest analysis on all strategies.

    Backtests each strategy over the specified lookback period
    and generates AI-powered recommendations based on performance.

    Args:
        lookback_days: Number of days to backtest (default: 90, 3 months)

    Returns:
        Performance comparison and recommendation
    """
    # Seed strategies if needed
    await seed_strategies(db)

    try:
        result = await strategy_analyzer_service.evaluate_all_strategies(
            db=db,
            lookback_days=lookback_days
        )

        return StrategyAnalysisResponse(
            analysis_date=result['analysis_date'],
            lookback_days=result['lookback_days'],
            evaluations=[
                StrategyEvaluationDetail(
                    strategy_id=e['strategy_id'],
                    name=e['name'],
                    strategy_type=e['strategy_type'],
                    total_return_pct=e['total_return_pct'],
                    sharpe_ratio=e['sharpe_ratio'],
                    max_drawdown_pct=e['max_drawdown_pct'],
                    win_rate=e['win_rate'],
                    total_trades=e['total_trades'],
                    recommendation_score=e['recommendation_score'],
                    lookback_days=e['lookback_days']
                )
                for e in result['evaluations']
            ],
            recommended_strategy_id=result['recommended_strategy_id'],
            recommendation_notes=result['recommendation_notes'],
            current_active_strategy_id=result['current_active_strategy_id']
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/strategies/analysis", response_model=StrategyAnalysisResponse)
async def get_latest_analysis(
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the most recent strategy analysis results.

    Returns cached results from the last analysis run.
    """
    result = await strategy_analyzer_service.get_latest_analysis(db)

    if not result:
        raise HTTPException(
            status_code=404,
            detail="No analysis results found. Run /strategies/analyze first."
        )

    return StrategyAnalysisResponse(
        analysis_date=result['analysis_date'],
        lookback_days=result['lookback_days'],
        evaluations=[
            StrategyEvaluationDetail(
                strategy_id=e['strategy_id'],
                name=e['name'],
                strategy_type=e['strategy_type'],
                total_return_pct=e['total_return_pct'],
                sharpe_ratio=e['sharpe_ratio'],
                max_drawdown_pct=e['max_drawdown_pct'],
                win_rate=e['win_rate'],
                total_trades=e['total_trades'],
                recommendation_score=e['recommendation_score'],
                lookback_days=e['lookback_days']
            )
            for e in result['evaluations']
        ],
        recommended_strategy_id=result['recommended_strategy_id'],
        recommendation_notes=result['recommendation_notes'],
        current_active_strategy_id=result['current_active_strategy_id']
    )


@router.post("/strategies/seed")
async def seed_strategies_endpoint(
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Manually seed initial strategies.

    This is called automatically when listing strategies,
    but can be triggered manually if needed.
    """
    count = await seed_strategies(db)
    return {
        "message": f"Seeded {count} strategies" if count > 0 else "Strategies already exist",
        "count": count
    }

"""
RigaCap API - FastAPI Backend with Real Data

Connects to:
- yfinance for market data (5 years historical)
- PostgreSQL for persistence
- Real DWAP-based signal generation
- APScheduler for daily EOD updates
"""

import logging
from contextlib import asynccontextmanager
from mangum import Mangum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
import pandas as pd

from app.core.config import settings
from app.core.database import init_db, get_db, Position as DBPosition, Trade as DBTrade, Signal as DBSignal, User, async_session
from app.core.security import get_current_user, get_admin_user, require_valid_subscription
from app.api.signals import router as signals_router
from app.api.email import router as email_router
from app.api.auth import router as auth_router
from app.api.billing import router as billing_router
from app.api.admin import router as admin_router
from app.api.social import router as social_router
from app.services.scanner import scanner_service
from app.services.scheduler import scheduler_service
from app.services.backtester import backtester_service
from app.services.market_analysis import market_analysis_service
from app.services.data_export import data_export_service


# ============================================================================
# Helper Functions
# ============================================================================

def get_split_adjusted_price(symbol: str, entry_date: datetime, fallback_price: float) -> float:
    """
    Get the split-adjusted close price for a symbol on a given date.

    yfinance retroactively adjusts all historical prices after splits,
    so looking up the close price for an entry date gives us the
    split-adjusted value automatically.

    Args:
        symbol: Stock symbol
        entry_date: Date the position was opened
        fallback_price: Original stored entry price (used if date not found)

    Returns:
        Split-adjusted close price, or fallback_price if not found
    """
    if symbol not in scanner_service.data_cache:
        return fallback_price

    df = scanner_service.data_cache[symbol]
    if df.empty:
        return fallback_price

    # Convert entry_date to date-only for comparison
    target_date = entry_date.date() if hasattr(entry_date, 'date') else entry_date

    # Find the closest date on or before entry_date
    # (markets may be closed on exact entry date)
    try:
        # Filter to dates <= entry_date
        df_before = df[df.index.date <= target_date]
        if df_before.empty:
            return fallback_price

        # Get the most recent date
        adjusted_price = float(df_before.iloc[-1]['close'])
        return adjusted_price
    except Exception:
        return fallback_price


# ============================================================================
# Pydantic Models
# ============================================================================

class PositionResponse(BaseModel):
    id: int
    symbol: str
    shares: float
    entry_price: float
    entry_date: str
    current_price: float
    stop_loss: float
    profit_target: float
    pnl_pct: float
    days_held: int
    # Trailing stop fields
    high_water_mark: float = 0.0  # Highest price since entry
    trailing_stop_price: float = 0.0  # Current trailing stop level
    trailing_stop_pct: float = 12.0  # Trailing stop percentage
    distance_to_stop_pct: float = 0.0  # How far price is from trailing stop (negative = below stop)
    sell_signal: str = "hold"  # hold, warning, sell


class PositionsListResponse(BaseModel):
    positions: List[PositionResponse]
    total_value: float
    total_pnl_pct: float


class OpenPositionRequest(BaseModel):
    symbol: str
    shares: Optional[float] = None
    price: Optional[float] = None
    entry_date: Optional[str] = None  # YYYY-MM-DD, for time-travel mode


class EquityPoint(BaseModel):
    date: str
    equity: float


# ============================================================================
# Lifespan (startup/shutdown)
# ============================================================================

async def store_signals_callback(signals):
    """Callback to store signals in database and export to S3 after scheduled scan"""
    if not signals:
        return

    # Export signals to S3 as static JSON for CDN delivery
    try:
        result = data_export_service.export_signals_json(signals)
        if result.get("success"):
            print(f"📤 Exported {result['count']} signals to S3 for CDN")
        else:
            print(f"⚠️ Signal export failed: {result.get('message')}")
    except Exception as e:
        print(f"⚠️ Signal export error: {e}")

    # Also store in database for historical tracking
    try:
        async with async_session() as db:
            for sig in signals:
                db_signal = DBSignal(
                    symbol=sig.symbol,
                    signal_type=sig.signal_type,
                    price=sig.price,
                    dwap=sig.dwap,
                    pct_above_dwap=sig.pct_above_dwap,
                    volume=sig.volume,
                    volume_ratio=sig.volume_ratio,
                    stop_loss=sig.stop_loss,
                    profit_target=sig.profit_target,
                    is_strong=sig.is_strong,
                    status="active"
                )
                db.add(db_signal)
            await db.commit()
            print(f"💾 Stored {len(signals)} signals in database")
    except Exception as e:
        print(f"⚠️ Database storage skipped: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup - skip DB for Lambda to avoid INIT timeout"""
    print("🚀 Starting RigaCap API...")

    import os
    is_lambda = os.environ.get("ENVIRONMENT") == "prod"

    if is_lambda:
        # Lambda: Skip DB init during startup to avoid 10s INIT timeout
        # Database will initialize lazily on first request
        print("📦 Lambda mode: Skipping DB init (will initialize on first request)")
    else:
        # Local dev: Initialize DB and start scheduler
        try:
            await init_db()
        except Exception as e:
            print(f"⚠️ Database not available: {e}")
            print("   Running in memory-only mode (positions won't persist)")

        cached_data = data_export_service.import_all()
        if cached_data:
            scanner_service.data_cache = cached_data
            print(f"📊 Loaded {len(cached_data)} symbols from cached parquet files")
        scheduler_service.add_callback(store_signals_callback)
        scheduler_service.start()
        print("📅 Scheduler started for daily EOD updates")

    yield

    # Cleanup
    print("👋 Shutting down RigaCap API...")
    if not is_lambda:
        scheduler_service.stop()


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="RigaCap API",
    version="2.0.0",
    description="DWAP-based stock trading signals with 5-year historical data",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)


# Security headers middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# Include API routers
app.include_router(signals_router, prefix="/api/signals", tags=["signals"])
app.include_router(email_router, prefix="/api/email", tags=["email"])
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(billing_router, prefix="/api/billing", tags=["billing"])
app.include_router(admin_router, prefix="/api/admin", tags=["admin"])
app.include_router(social_router, prefix="/api/admin/social", tags=["social"])

# Lambda handler (for AWS Lambda deployment)
# lifespan="off" avoids issues with event loop reuse on warm Lambdas
_mangum_handler = None
_lambda_data_loaded = False


def _ensure_lambda_data_loaded():
    """Load data from S3 on Lambda cold start (runs once per container)."""
    global _lambda_data_loaded
    if _lambda_data_loaded:
        return

    import os
    if os.environ.get("ENVIRONMENT") != "prod":
        _lambda_data_loaded = True
        return

    # Only load if cache is empty
    if not scanner_service.data_cache:
        print("📦 Lambda cold start: Loading data from S3...")
        try:
            cached_data = data_export_service.import_all()
            if cached_data:
                scanner_service.data_cache = cached_data
                print(f"✅ Loaded {len(cached_data)} symbols from S3")
                _lambda_data_loaded = True
            else:
                print("⚠️ No cached data found in S3 — will retry next request")
        except Exception as e:
            print(f"⚠️ Failed to load data from S3: {e} — will retry next request")
    else:
        _lambda_data_loaded = True


async def _run_walk_forward_job(job_config: dict):
    """Run walk-forward simulation job asynchronously."""
    import json
    from datetime import datetime
    from app.services.walk_forward_service import walk_forward_service
    from sqlalchemy import select
    from app.core.database import WalkForwardSimulation

    job_id = job_config.get("job_id")

    async with async_session() as db:
        try:
            # If no job_id provided, create a new job record
            if not job_id:
                start = datetime.strptime(job_config["start_date"], "%Y-%m-%d")
                end = datetime.strptime(job_config["end_date"], "%Y-%m-%d")
                new_job = WalkForwardSimulation(
                    simulation_date=datetime.utcnow(),
                    start_date=start,
                    end_date=end,
                    reoptimization_frequency=job_config.get("frequency", "biweekly"),
                    status="running",
                    total_return_pct=0,
                    sharpe_ratio=0,
                    max_drawdown_pct=0,
                    num_strategy_switches=0,
                    benchmark_return_pct=0,
                )
                db.add(new_job)
                await db.commit()
                await db.refresh(new_job)
                job_id = new_job.id
                print(f"[ASYNC-WF] Created new job {job_id}")
            else:
                # Update existing job status to running
                result = await db.execute(
                    select(WalkForwardSimulation).where(WalkForwardSimulation.id == job_id)
                )
                job = result.scalar_one_or_none()
                if job:
                    job.status = "running"
                    await db.commit()

            print(f"[ASYNC-WF] Starting walk-forward job {job_id}")

            # Run the simulation
            start = datetime.strptime(job_config["start_date"], "%Y-%m-%d")
            end = datetime.strptime(job_config["end_date"], "%Y-%m-%d")

            sim_result = await walk_forward_service.run_walk_forward_simulation(
                db=db,
                start_date=start,
                end_date=end,
                reoptimization_frequency=job_config["frequency"],
                min_score_diff=job_config["min_score_diff"],
                enable_ai_optimization=job_config["enable_ai"],
                max_symbols=job_config["max_symbols"],
                existing_job_id=job_id,
                fixed_strategy_id=job_config.get("strategy_id"),
                n_trials=job_config.get("n_trials", 30),
                carry_positions=job_config.get("carry_positions", False)
            )

            print(f"[ASYNC-WF] Job {job_id} completed: return={sim_result.total_return_pct}%")
            return {"status": "completed", "job_id": job_id}

        except Exception as e:
            import traceback
            print(f"[ASYNC-WF] Job {job_id} failed: {e}")
            print(traceback.format_exc())

            # Update job status to failed
            try:
                result = await db.execute(
                    select(WalkForwardSimulation).where(WalkForwardSimulation.id == job_id)
                )
                job = result.scalar_one_or_none()
                if job:
                    job.status = "failed"
                    job.switch_history_json = json.dumps({"error": str(e)})
                    await db.commit()
            except Exception:
                pass

            return {"status": "failed", "job_id": job_id, "error": str(e)}


async def _get_walk_forward_history(limit: int = 10):
    """Get list of recent walk-forward simulations."""
    import json
    from sqlalchemy import select, desc
    from app.core.database import WalkForwardSimulation

    async with async_session() as db:
        result = await db.execute(
            select(WalkForwardSimulation)
            .order_by(desc(WalkForwardSimulation.simulation_date))
            .limit(limit)
        )
        sims = result.scalars().all()

        simulations = []
        for s in sims:
            # Try to get the initial strategy from switch_history
            strategy_name = None
            if s.switch_history_json:
                try:
                    switch_history = json.loads(s.switch_history_json)
                    if switch_history and len(switch_history) > 0:
                        strategy_name = switch_history[0].get("strategy_name")
                except (json.JSONDecodeError, KeyError):
                    pass

            simulations.append({
                "id": s.id,
                "simulation_date": s.simulation_date.isoformat() if s.simulation_date else None,
                "start_date": s.start_date.isoformat() if s.start_date else None,
                "end_date": s.end_date.isoformat() if s.end_date else None,
                "strategy_name": strategy_name,
                "reoptimization_frequency": s.reoptimization_frequency,
                "total_return_pct": s.total_return_pct,
                "sharpe_ratio": s.sharpe_ratio,
                "max_drawdown_pct": s.max_drawdown_pct,
                "benchmark_return_pct": s.benchmark_return_pct,
                "num_strategy_switches": s.num_strategy_switches,
                "status": s.status,
                "has_trades": bool(s.trades_json),
            })

        return {
            "status": "success",
            "simulations": simulations
        }


async def _seed_and_list_strategies():
    """Seed strategies if needed and return the list."""
    from sqlalchemy import select
    from app.core.database import StrategyDefinition
    from app.api.admin import seed_strategies

    async with async_session() as db:
        # Seed strategies
        count = await seed_strategies(db)
        print(f"[SEED] Seeded {count} strategies")

        # List all strategies
        result = await db.execute(select(StrategyDefinition).order_by(StrategyDefinition.id))
        strategies = result.scalars().all()

        return {
            "status": "success",
            "seeded": count,
            "strategies": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "strategy_type": s.strategy_type,
                    "is_active": s.is_active
                }
                for s in strategies
            ]
        }


async def _list_strategies():
    """List all strategies."""
    from sqlalchemy import select
    from app.core.database import StrategyDefinition

    async with async_session() as db:
        result = await db.execute(select(StrategyDefinition).order_by(StrategyDefinition.id))
        strategies = result.scalars().all()

        return {
            "status": "success",
            "strategies": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "strategy_type": s.strategy_type,
                    "is_active": s.is_active
                }
                for s in strategies
            ]
        }


async def _get_walk_forward_trades(simulation_id: int):
    """Get detailed trades from a walk-forward simulation."""
    import json
    from sqlalchemy import select
    from app.core.database import WalkForwardSimulation

    async with async_session() as db:
        result = await db.execute(
            select(WalkForwardSimulation).where(WalkForwardSimulation.id == simulation_id)
        )
        sim = result.scalars().first()

        if not sim:
            return {"status": "error", "error": f"Simulation {simulation_id} not found"}

        trades = json.loads(sim.trades_json) if sim.trades_json else []

        # Calculate summary statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('pnl_pct', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl_pct', 0) <= 0]

        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = sum(t.get('pnl_pct', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.get('pnl_pct', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        total_pnl = sum(t.get('pnl_dollars', 0) for t in trades)

        # Group by exit reason
        exit_reasons = {}
        for t in trades:
            reason = t.get('exit_reason', 'unknown')
            if reason not in exit_reasons:
                exit_reasons[reason] = 0
            exit_reasons[reason] += 1

        # Get strategy name from switch history
        strategy_name = None
        if sim.switch_history_json:
            try:
                switch_history = json.loads(sim.switch_history_json)
                if switch_history and len(switch_history) > 0:
                    strategy_name = switch_history[0].get("strategy_name")
            except (json.JSONDecodeError, KeyError):
                pass

        return {
            "status": "success",
            "simulation_id": simulation_id,
            "strategy_name": strategy_name,
            "simulation_date": sim.simulation_date.isoformat() if sim.simulation_date else None,
            "start_date": sim.start_date.isoformat() if sim.start_date else None,
            "end_date": sim.end_date.isoformat() if sim.end_date else None,
            "total_return_pct": sim.total_return_pct,
            "benchmark_return_pct": sim.benchmark_return_pct,
            "trades": trades,
            "summary": {
                "total_trades": total_trades,
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate_pct": round(win_rate, 1),
                "avg_win_pct": round(avg_win, 2),
                "avg_loss_pct": round(avg_loss, 2),
                "total_pnl_dollars": round(total_pnl, 2),
                "exit_reasons": exit_reasons
            }
        }


def handler(event, context):
    """
    Lambda handler that supports:
    1. Warmer events (from EventBridge scheduled warmer)
    2. Walk-forward async jobs (from async Lambda invocation)
    3. API Gateway HTTP API events (via Mangum)
    """
    import asyncio

    # Ensure data is loaded on cold start
    _ensure_lambda_data_loaded()

    # Handle warmer events - just return success to keep Lambda warm
    if event.get("warmer"):
        print(f"🔥 Warmer ping - {len(scanner_service.data_cache)} symbols in cache")
        return {
            "statusCode": 200,
            "body": f'{{"status": "warm", "symbols_loaded": {len(scanner_service.data_cache)}}}'
        }

    # Handle daily scan (EventBridge: 4 PM ET Mon-Fri)
    # Refreshes data from yfinance, persists cache to S3, exports signals + dashboard + snapshot
    if event.get("daily_scan"):
        print(f"📡 Daily scan triggered - {len(scanner_service.data_cache)} symbols in cache")
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _run_daily_scan():
            from app.services.data_export import data_export_service
            from app.api.signals import compute_shared_dashboard_data
            from datetime import date

            # 1. Force-refresh data from yfinance (scan() skips yfinance in Lambda mode)
            print("📡 Fetching fresh price data from yfinance...")
            await scanner_service.fetch_data()
            print(f"📡 Fetched {len(scanner_service.data_cache)} symbols from yfinance")

            # 2. Run scan on fresh data
            signals = await scanner_service.scan(refresh_data=False)
            print(f"📡 Scan complete: {len(signals)} signals")

            # 3. Persist refreshed cache to S3 pickle for future cold starts
            export_result = data_export_service.export_pickle(scanner_service.data_cache)
            print(f"💾 Data cache persisted to S3: {export_result.get('count', 0)} symbols")

            # 4. Store signals in DB + export to S3
            await store_signals_callback(signals)

            # 5. Export dashboard JSON + daily snapshot
            async with async_session() as db:
                data = await compute_shared_dashboard_data(db)
                dash_result = data_export_service.export_dashboard_json(data)
                today_str = date.today().strftime("%Y-%m-%d")
                snap_result = data_export_service.export_snapshot(today_str, data)

            return {
                "status": "success",
                "signals": len(signals),
                "symbols_cached": len(scanner_service.data_cache),
                "dashboard": dash_result,
                "snapshot": snap_result,
            }

        try:
            result = loop.run_until_complete(_run_daily_scan())
            print(f"📡 Daily scan result: {result}")
            return result
        except Exception as e:
            import traceback
            print(f"❌ Daily scan failed: {e}")
            traceback.print_exc()
            return {"status": "failed", "error": str(e)}

    # Handle dashboard cache export
    if event.get("export_dashboard_cache"):
        print("📦 Dashboard cache export requested")
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _export_dashboard():
            from app.api.signals import compute_shared_dashboard_data
            from app.services.data_export import data_export_service
            async with async_session() as db:
                data = await compute_shared_dashboard_data(db)
                result = data_export_service.export_dashboard_json(data)
                return {"status": "success", **result}

        try:
            result = loop.run_until_complete(_export_dashboard())
            print(f"📦 Dashboard cache export: {result}")
            return result
        except Exception as e:
            import traceback
            print(f"❌ Dashboard cache export failed: {e}")
            traceback.print_exc()
            return {"status": "failed", "error": str(e)}

    # Handle intraday position monitor (manual trigger for testing)
    if event.get("intraday_monitor"):
        print(f"📡 Intraday position monitor requested - {len(scanner_service.data_cache)} symbols in cache")
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _run_intraday_monitor():
            from app.core.database import async_session as async_sess, Position as DBPosition, User as DBUser
            from app.services.email_service import email_service
            from app.services.data_export import data_export_service as des
            from app.services.scheduler import scheduler_service as sched
            from sqlalchemy import select
            from datetime import date
            import yfinance as yf

            async with async_sess() as db:
                # Query all open positions with user email
                result = await db.execute(
                    select(DBPosition, DBUser.email, DBUser.full_name)
                    .join(DBUser, DBPosition.user_id == DBUser.id)
                    .where(DBPosition.status == 'open')
                )
                rows = result.all()

                if not rows:
                    return {"status": "success", "positions_checked": 0, "alerts_sent": 0, "message": "No open positions"}

                # Fetch live prices
                symbols = list({row[0].symbol for row in rows})
                live_prices = {}
                tickers = yf.Tickers(' '.join(symbols))
                for sym in symbols:
                    try:
                        ticker = tickers.tickers.get(sym)
                        if ticker:
                            live_prices[sym] = ticker.fast_info.last_price
                    except Exception:
                        continue

                # Get regime forecast
                regime_forecast = None
                dashboard_data = des.read_dashboard_json()
                if dashboard_data:
                    regime_forecast = dashboard_data.get('regime_forecast')

                # Check positions
                alerts_sent = 0
                today = date.today()
                details = []

                for position, user_email, user_name in rows:
                    sym = position.symbol
                    if sym not in live_prices:
                        details.append({"symbol": sym, "status": "no_price"})
                        continue

                    live_price = live_prices[sym]

                    if live_price > (position.highest_price or position.entry_price):
                        position.highest_price = live_price

                    guidance = sched._check_sell_trigger(position, live_price, regime_forecast)

                    if guidance and guidance['action'] in ('sell', 'warning'):
                        dedup_key = f"{position.id}_{guidance['action']}_{today}"
                        if dedup_key not in sched._alerted_sell_positions:
                            try:
                                await email_service.send_sell_alert(
                                    to_email=user_email,
                                    user_name=user_name or "",
                                    symbol=sym,
                                    action=guidance['action'],
                                    reason=guidance['reason'],
                                    current_price=live_price,
                                    entry_price=position.entry_price,
                                    stop_price=guidance.get('stop_price'),
                                    user_id=str(position.user_id),
                                )
                                sched._alerted_sell_positions.add(dedup_key)
                                alerts_sent += 1
                            except Exception as e:
                                print(f"Failed to send alert for {sym}: {e}")

                        details.append({
                            "symbol": sym,
                            "price": live_price,
                            "action": guidance['action'],
                            "reason": guidance['reason'],
                        })
                    else:
                        details.append({
                            "symbol": sym,
                            "price": live_price,
                            "action": "hold",
                        })

                await db.commit()

                return {
                    "status": "success",
                    "positions_checked": len(rows),
                    "symbols_priced": len(live_prices),
                    "alerts_sent": alerts_sent,
                    "regime": regime_forecast.get("recommended_action") if regime_forecast else None,
                    "details": details,
                }

        try:
            result = loop.run_until_complete(_run_intraday_monitor())
            print(f"📡 Intraday monitor: {result}")
            return result
        except Exception as e:
            import traceback
            print(f"❌ Intraday monitor failed: {e}")
            traceback.print_exc()
            return {"status": "failed", "error": str(e)}

    # Query walk-forward job details (read-only)
    if event.get("wf_query"):
        config = event["wf_query"]
        job_id = config.get("job_id")

        async def _wf_query():
            from app.services.walk_forward_service import walk_forward_service
            async with async_session() as db:
                return await walk_forward_service.get_simulation_details(db, job_id)

        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_wf_query())
        return result or {"error": f"Job {job_id} not found"}

    # Handle async walk-forward jobs
    if event.get("walk_forward_job"):
        print(f"📊 Walk-forward async job received - {len(scanner_service.data_cache)} symbols in cache, SPY={'SPY' in scanner_service.data_cache}")
        job_config = event["walk_forward_job"]
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_run_walk_forward_job(job_config))
        return result

    # Handle Step Functions walk-forward: init
    if event.get("wf_init"):
        print(f"📊 WF-INIT: Step Functions initialization")
        config = event["wf_init"]
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _run_wf_init():
            from app.services.walk_forward_service import walk_forward_service
            async with async_session() as db:
                return await walk_forward_service.init_simulation(db, config)

        try:
            result = loop.run_until_complete(_run_wf_init())
            return result
        except Exception as e:
            import traceback
            print(f"❌ WF-INIT failed: {e}")
            print(traceback.format_exc())
            return {"error": str(e)}

    # Handle Step Functions walk-forward: single period
    if event.get("wf_period"):
        print(f"📊 WF-PERIOD: Processing period {event['wf_period'].get('period_index', '?')}")
        state = event["wf_period"]
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _run_wf_period():
            from app.services.walk_forward_service import walk_forward_service
            async with async_session() as db:
                return await walk_forward_service.run_single_period(db, state)

        try:
            result = loop.run_until_complete(_run_wf_period())
            return result
        except Exception as e:
            import traceback
            print(f"❌ WF-PERIOD failed: {e}")
            print(traceback.format_exc())
            # Don't kill the whole simulation — increment period and continue
            return {
                **state,
                "period_index": state.get("period_index", 0) + 1,
                "error": str(e)
            }

    # Handle Step Functions walk-forward: finalize
    if event.get("wf_finalize"):
        print(f"📊 WF-FINALIZE: Computing final metrics")
        state = event["wf_finalize"]
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _run_wf_finalize():
            from app.services.walk_forward_service import walk_forward_service
            async with async_session() as db:
                return await walk_forward_service.finalize_simulation(db, state)

        try:
            result = loop.run_until_complete(_run_wf_finalize())
            return result
        except Exception as e:
            import traceback
            print(f"❌ WF-FINALIZE failed: {e}")
            print(traceback.format_exc())
            return {"error": str(e), "simulation_id": state.get("simulation_id")}

    # Handle Step Functions walk-forward: mark failed
    if event.get("wf_fail"):
        print(f"📊 WF-FAIL: Marking simulation as failed")
        state = event["wf_fail"]
        error_msg = state.get("error", "Unknown error in Step Functions execution")
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _run_wf_fail():
            from app.services.walk_forward_service import walk_forward_service
            async with async_session() as db:
                return await walk_forward_service.mark_simulation_failed(db, state, error_msg)

        try:
            result = loop.run_until_complete(_run_wf_fail())
            return result
        except Exception as e:
            print(f"❌ WF-FAIL handler itself failed: {e}")
            return {"error": str(e)}

    # Handle nightly walk-forward job (direct Lambda invocation)
    if event.get("nightly_wf_job"):
        print(f"🌙 Nightly WF job received - {len(scanner_service.data_cache)} symbols in cache")
        config = event["nightly_wf_job"]
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _run_nightly_wf():
            from datetime import timedelta
            from app.core.database import WalkForwardSimulation
            from app.services.walk_forward_service import walk_forward_service
            from sqlalchemy import delete, select
            import json

            days_back = config.get("days_back", 90)
            strategy_id = config.get("strategy_id", 5)
            max_symbols = config.get("max_symbols", 100)

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            async with async_session() as db:
                # Delete old nightly cache
                await db.execute(
                    delete(WalkForwardSimulation).where(
                        WalkForwardSimulation.is_nightly_missed_opps == True
                    )
                )

                job = WalkForwardSimulation(
                    simulation_date=datetime.utcnow(),
                    start_date=start_date,
                    end_date=end_date,
                    reoptimization_frequency="biweekly",
                    status="running",
                    is_nightly_missed_opps=True,
                    total_return_pct=0,
                    sharpe_ratio=0,
                    max_drawdown_pct=0,
                    num_strategy_switches=0,
                    benchmark_return_pct=0,
                )
                db.add(job)
                await db.commit()
                await db.refresh(job)
                job_id = job.id

                print(f"[NIGHTLY-WF] Starting {days_back}-day walk-forward (job {job_id})")

                result = await walk_forward_service.run_walk_forward_simulation(
                    db=db,
                    start_date=start_date,
                    end_date=end_date,
                    reoptimization_frequency="biweekly",
                    min_score_diff=10.0,
                    enable_ai_optimization=False,
                    max_symbols=max_symbols,
                    existing_job_id=job_id,
                    fixed_strategy_id=strategy_id,
                )

                # Generate social content
                post_count = 0
                try:
                    from app.services.social_content_service import social_content_service
                    posts = await social_content_service.generate_from_nightly_wf(db, job_id)
                    post_count = len(posts)
                except Exception as e:
                    print(f"[NIGHTLY-WF] Social content error: {e}")

                return {
                    "status": "completed",
                    "job_id": job_id,
                    "total_return_pct": result.total_return_pct,
                    "total_trades": len(result.trades) if hasattr(result, 'trades') else 0,
                    "social_posts_generated": post_count,
                }

        try:
            result = loop.run_until_complete(_run_nightly_wf())
            return result
        except Exception as e:
            import traceback
            print(f"❌ Nightly WF failed: {e}")
            print(traceback.format_exc())
            return {"status": "failed", "error": str(e)}

    # Handle backtest requests (direct Lambda invocation)
    if event.get("backtest_request"):
        print(f"📊 Backtest request received")
        req = event["backtest_request"]
        days = req.get("days", 252)
        strategy_type = req.get("strategy_type", "momentum")
        include_trades = req.get("include_trades", True)

        try:
            from app.services.backtester import backtester_service
            result = backtester_service.run_backtest(
                lookback_days=days,
                strategy_type=strategy_type
            )

            response = {
                "status": "success",
                "total_return_pct": result.total_return_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown_pct": result.max_drawdown_pct,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "total_pnl": result.total_pnl,
                "start_date": result.start_date,
                "end_date": result.end_date,
            }

            if include_trades:
                response["trades"] = [t.to_dict() for t in result.trades]

            print(f"📊 Backtest complete: {result.total_return_pct:.1f}% return, {result.total_trades} trades")
            return response

        except Exception as e:
            import traceback
            print(f"❌ Backtest failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Handle walk-forward history lookup (direct Lambda invocation)
    if event.get("walk_forward_history"):
        print("📊 Walk-forward history request received")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_get_walk_forward_history(event.get("limit", 10)))
            return result
        except Exception as e:
            import traceback
            print(f"❌ Walk-forward history failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Handle walk-forward trades lookup (direct Lambda invocation)
    if event.get("walk_forward_trades"):
        print("📊 Walk-forward trades request received")
        simulation_id = event.get("walk_forward_trades")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_get_walk_forward_trades(simulation_id))
            return result
        except Exception as e:
            import traceback
            print(f"❌ Walk-forward trades failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Handle AI content generation test (direct Lambda invocation)
    if event.get("test_ai_content"):
        print("🤖 AI content generation test received")
        config = event["test_ai_content"]

        async def _test_ai_content():
            from app.services.ai_content_service import ai_content_service

            trade = config.get("trade", {
                "symbol": "NVDA",
                "entry_date": "2026-01-15",
                "exit_date": "2026-02-10",
                "entry_price": 125.50,
                "exit_price": 148.75,
                "pnl_pct": 18.5,
                "strategy": "DWAP+Momentum Ensemble",
            })
            platform = config.get("platform", "twitter")
            post_type = config.get("post_type", "trade_result")

            post = await ai_content_service.generate_post(
                trade=trade,
                post_type=post_type,
                platform=platform,
            )
            if not post:
                return {"status": "error", "error": "AI generation returned None — check ANTHROPIC_API_KEY"}

            return {
                "status": "success",
                "platform": platform,
                "post_type": post_type,
                "generated_text": post.text_content,
                "hashtags": post.hashtags,
                "char_count": len(post.text_content) if post.text_content else 0,
                "ai_model": post.ai_model,
            }

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_test_ai_content())
            return result
        except Exception as e:
            import traceback
            print(f"❌ AI content test failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Generate AI posts from real WF trades and save to DB (direct Lambda invocation)
    if event.get("generate_social_posts"):
        print("🤖 Generate social posts from WF trades")
        config = event["generate_social_posts"]

        async def _generate_social_posts():
            import json as _json
            from app.core.database import SocialPost, WalkForwardSimulation
            from app.services.ai_content_service import ai_content_service
            from sqlalchemy import delete as sa_delete

            job_id = config.get("job_id", 112)
            min_pnl = config.get("min_pnl_pct", 5.0)
            since_date = config.get("since_date", "2026-01-01")
            clear_existing = config.get("clear_existing", False)
            platforms = config.get("platforms", ["twitter", "instagram"])
            post_types = config.get("post_types", ["trade_result", "missed_opportunity"])
            max_trades = config.get("max_trades", 5)

            async with async_session() as db:
                # Load trades from WF simulation JSON
                result = await db.execute(
                    select(WalkForwardSimulation).where(WalkForwardSimulation.id == job_id)
                )
                sim = result.scalars().first()
                if not sim or not sim.trades_json:
                    return {"status": "error", "error": f"Simulation {job_id} not found or has no trades"}

                all_trades = _json.loads(sim.trades_json)
                # Filter: exit_date >= since_date AND pnl_pct >= min_pnl
                filtered = [
                    t for t in all_trades
                    if str(t.get("exit_date", ""))[:10] >= since_date
                    and t.get("pnl_pct", 0) >= min_pnl
                ]
                # Deduplicate by symbol (keep best trade per symbol)
                seen_symbols = set()
                unique_trades = []
                for t in sorted(filtered, key=lambda x: -x.get("pnl_pct", 0)):
                    if t["symbol"] not in seen_symbols:
                        seen_symbols.add(t["symbol"])
                        unique_trades.append(t)

                unique_trades = unique_trades[:max_trades]

                if not unique_trades:
                    return {"status": "error", "error": f"No trades found for job {job_id} since {since_date} with pnl >= {min_pnl}%"}

                print(f"Found {len(unique_trades)} qualifying trades (deduped by symbol)")

                # Clear existing posts if requested
                deleted = 0
                if clear_existing:
                    result = await db.execute(sa_delete(SocialPost))
                    deleted = result.rowcount
                    await db.commit()
                    print(f"Cleared {deleted} existing posts")

                # Generate posts for each trade x platform x post_type
                created = []
                for trade_data in unique_trades:
                    trade_data["strategy"] = "DWAP+Momentum Ensemble"
                    for platform in platforms:
                        for post_type in post_types:
                            post = await ai_content_service.generate_post(
                                trade=trade_data,
                                post_type=post_type,
                                platform=platform,
                            )
                            if post:
                                db.add(post)
                                created.append({
                                    "symbol": trade_data["symbol"],
                                    "platform": platform,
                                    "post_type": post_type,
                                    "text": post.text_content[:80] + "...",
                                    "chars": len(post.text_content),
                                })
                                print(f"  Created {platform}/{post_type} for {trade_data['symbol']} ({len(post.text_content)} chars)")

                await db.commit()

                return {
                    "status": "success",
                    "deleted_existing": deleted,
                    "trades_used": len(unique_trades),
                    "posts_created": len(created),
                    "posts": created,
                }

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_generate_social_posts())
            return result
        except Exception as e:
            import traceback
            print(f"❌ Generate social posts failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Scan followed accounts for reply opportunities (direct Lambda invocation)
    if event.get("scan_replies"):
        print("🔍 Scanning for reply opportunities")
        config = event["scan_replies"]

        async def _scan_replies():
            from app.services.reply_scanner_service import reply_scanner_service
            from app.core.database import SocialPost as SocialPostModel
            from sqlalchemy import delete

            async with async_session() as db:
                # Optionally clear old reply drafts before regenerating
                if config.get("clear_existing"):
                    deleted = await db.execute(
                        delete(SocialPostModel).where(
                            SocialPostModel.post_type == "contextual_reply",
                            SocialPostModel.status.in_(["draft", "approved"]),
                        )
                    )
                    await db.commit()
                    print(f"🗑️ Cleared {deleted.rowcount} existing reply drafts")

                result = await reply_scanner_service.scan_and_generate(
                    db=db,
                    since_hours=config.get("since_hours", 4),
                    dry_run=config.get("dry_run", False),
                    accounts=config.get("accounts"),
                )
                return result

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_scan_replies())
            return result
        except Exception as e:
            import traceback
            print(f"❌ Reply scan failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Send test emails for all templates (direct Lambda invocation)
    if event.get("test_emails"):
        print("📧 Test all email templates")
        config = event.get("test_emails") or {}

        async def _test_emails():
            from app.services.email_service import email_service, admin_email_service
            from types import SimpleNamespace

            to = config.get("to_email", "erik@rigacap.com")
            # Allow sending a subset: {"only": ["welcome", "sell_alert"]}
            only = config.get("only")
            results = {}

            # Look up user_id for footer links (accept override for mail-tester)
            test_user_id = config.get("user_id")
            if not test_user_id:
                try:
                    from app.core.database import async_session, User
                    from sqlalchemy import select
                    async with async_session() as db:
                        r = await db.execute(select(User.id).where(User.email == to))
                        row = r.scalar_one_or_none()
                        if row:
                            test_user_id = str(row)
                except Exception:
                    pass

            async def _try(name, coro):
                if only and name not in only:
                    return
                try:
                    ok = await coro
                    results[name] = "sent" if ok else "failed"
                    print(f"  {'✅' if ok else '❌'} {name}")
                except Exception as e:
                    results[name] = f"error: {e}"
                    print(f"  ❌ {name}: {e}")

            # --- Subscriber emails (EmailService) ---

            # 1. Daily Summary
            await _try("daily_summary", email_service.send_daily_summary(
                to_email=to,
                signals=[
                    {"symbol": "NVDA", "price": 156.20, "is_fresh": True, "is_strong": True,
                     "pct_above_dwap": 7.3, "momentum_rank": 2, "days_since_crossover": 0},
                    {"symbol": "PLTR", "price": 97.30, "is_fresh": True, "is_strong": False,
                     "pct_above_dwap": 5.8, "momentum_rank": 5, "days_since_crossover": 2},
                    {"symbol": "MSTR", "price": 341.50, "is_fresh": False, "is_strong": False,
                     "pct_above_dwap": 6.1, "momentum_rank": 8, "days_since_crossover": 12},
                    {"symbol": "COIN", "price": 267.80, "is_fresh": False, "is_strong": False,
                     "pct_above_dwap": 5.4, "momentum_rank": 14, "days_since_crossover": 18},
                ],
                market_regime={"regime": "weak_bull", "spy_price": 580.50, "vix_level": 16.3},
                positions=[
                    {"symbol": "AAPL", "entry_price": 228.0, "current_price": 245.50, "shares": 80},
                    {"symbol": "MSFT", "entry_price": 420.0, "current_price": 408.30, "shares": 45},
                ],
                missed_opportunities=[
                    {"symbol": "META", "would_be_pnl": 3200, "would_be_pct": 18.5, "signal_date": "2026-01-28"},
                    {"symbol": "AMZN", "would_be_pnl": 1850, "would_be_pct": 12.1, "signal_date": "2026-02-03"},
                ],
                watchlist=[
                    {"symbol": "TSLA", "price": 312.40, "pct_above_dwap": 3.8, "distance_to_trigger": 1.2},
                    {"symbol": "AMD", "price": 178.90, "pct_above_dwap": 4.1, "distance_to_trigger": 0.9},
                ],
                user_id=test_user_id,
            ))

            # 2. Welcome
            await _try("welcome", email_service.send_welcome_email(
                to_email=to, name="Erik Kinsman",
            ))

            # 3. Password Reset
            await _try("password_reset", email_service.send_password_reset_email(
                to_email=to, name="Erik Kinsman",
                reset_url="https://rigacap.com/reset-password?token=sample-test-token-abc123",
            ))

            # 4. Trial Ending
            await _try("trial_ending", email_service.send_trial_ending_email(
                to_email=to, name="Erik Kinsman",
                days_remaining=1, signals_generated=47, strong_signals_seen=12,
            ))

            # 5. Goodbye
            await _try("goodbye", email_service.send_goodbye_email(
                to_email=to, name="Erik Kinsman",
            ))

            # 6. Sell Alert
            await _try("sell_alert", email_service.send_sell_alert(
                to_email=to, user_name="Erik Kinsman",
                symbol="MSFT", action="sell",
                reason="Trailing stop triggered — 15% from high water mark",
                current_price=408.30, entry_price=420.00, stop_price=411.60,
                user_id=test_user_id,
            ))

            # 7. Double Signal Alert (breakout)
            await _try("double_signal_alert", email_service.send_double_signal_alert(
                to_email=to,
                new_signals=[
                    {"symbol": "NVDA", "price": 156.20, "pct_above_dwap": 7.3,
                     "momentum_rank": 2, "short_momentum": 12.4,
                     "dwap_crossover_date": "2026-02-17", "days_since_crossover": 0},
                    {"symbol": "PLTR", "price": 97.30, "pct_above_dwap": 5.8,
                     "momentum_rank": 5, "short_momentum": 9.1,
                     "dwap_crossover_date": "2026-02-15", "days_since_crossover": 2},
                ],
                approaching=[
                    {"symbol": "TSLA", "price": 312.40, "pct_above_dwap": 3.8, "distance_to_trigger": 1.2},
                    {"symbol": "AMD", "price": 178.90, "pct_above_dwap": 4.1, "distance_to_trigger": 0.9},
                ],
                market_regime={"regime": "weak_bull", "spy_price": 580.50},
                user_id=test_user_id,
            ))

            # 8. Intraday Signal Alert
            await _try("intraday_signal", email_service.send_intraday_signal_alert(
                to_email=to, user_name="Erik Kinsman",
                symbol="NVDA", live_price=156.20, dwap=145.50,
                pct_above_dwap=7.3, momentum_rank=2, sector="Technology",
                user_id=test_user_id,
            ))

            # --- Admin emails (AdminEmailService) ---

            # 9. Ticker Alert
            await _try("ticker_alert", admin_email_service.send_ticker_alert(
                to_email=to,
                issues=[
                    {"symbol": "TWTR", "issue": "Delisted — ticker changed to X",
                     "last_price": 53.70, "last_date": "2023-10-27",
                     "suggestion": "Remove TWTR, add X to universe"},
                    {"symbol": "SIVB", "issue": "No price data since March 2023",
                     "last_price": 106.04, "last_date": "2023-03-10"},
                ],
                check_type="universe",
            ))

            # 10. Strategy Analysis
            await _try("strategy_analysis", admin_email_service.send_strategy_analysis_email(
                to_email=to,
                analysis_results={
                    "evaluations": [
                        {"name": "DWAP+Momentum Ensemble", "recommendation_score": 87.2,
                         "sharpe_ratio": 1.48, "total_return_pct": 289.0},
                        {"name": "Concentrated Momentum", "recommendation_score": 72.1,
                         "sharpe_ratio": 1.15, "total_return_pct": 195.0},
                        {"name": "DWAP Classic", "recommendation_score": 45.8,
                         "sharpe_ratio": 0.19, "total_return_pct": 42.0},
                    ],
                    "analysis_date": datetime.now().isoformat(),
                    "lookback_days": 90,
                },
                recommendation="Ensemble continues to outperform. No switch recommended. "
                    "Sharpe ratio 1.48 is well above the 0.8 threshold.",
                switch_executed=False,
                switch_reason="Current strategy is top-ranked; no switch needed.",
            ))

            # 11. Strategy Switch
            await _try("switch_notification", admin_email_service.send_switch_notification_email(
                to_email=to,
                from_strategy="Concentrated Momentum",
                to_strategy="DWAP+Momentum Ensemble",
                reason="Ensemble outperformed by +15.1 points in 90-day backtest",
                metrics={"score_before": 72.1, "score_after": 87.2, "score_diff": 15.1},
            ))

            # 12. AI Generation Complete
            await _try("ai_generation_complete", admin_email_service.send_generation_complete_email(
                to_email=to,
                best_params={
                    "trailing_stop": "12%", "max_positions": 6,
                    "rebalance_freq": "biweekly", "dwap_threshold": "5%",
                    "momentum_window_short": "10d", "momentum_window_long": "60d",
                },
                expected_metrics={"sharpe": 1.48, "return": 31.0, "drawdown": -15.1},
                market_regime="weak_bull",
                created_strategy_name="AI-Optimized Ensemble v3",
            ))

            # 13. Social Post Notification (Twitter T-24h)
            twitter_post = SimpleNamespace(
                id=999, platform="twitter",
                text_content="NVDA called at $127.40 on Jan 15. Exited at $156.20 three weeks later.\n\n+22.6% while the market was flat.\n\nThe ensemble saw what pure momentum missed: DWAP breakout + top-5 ranking + volume surge.\n\nNot luck. Pattern recognition.",
                scheduled_for=datetime.utcnow() + timedelta(hours=24),
                post_type="we_called_it", ai_generated=True,
                ai_model="claude-sonnet-4-5-20250929",
                hashtags="#NVDA #TradingSignals #Momentum #RigaCap",
                image_s3_key=None,
            )
            await _try("post_notification_twitter", admin_email_service.send_post_approval_notification(
                to_email=to, post=twitter_post, hours_before=24,
                cancel_url="https://rigacap.com/api/admin/social/posts/999/cancel-email?token=test-preview",
            ))

            # 14. Social Post Notification (Instagram T-1h with chart)
            insta_post = SimpleNamespace(
                id=998, platform="instagram",
                text_content="We flagged PLTR at $78.50 when the ensemble fired all 3 signals.\n\nDWAP crossover confirmed. Momentum rank #2. Volume 1.6x average.\n\nThree weeks later: $97.30. That's +23.9%.",
                scheduled_for=datetime.utcnow() + timedelta(hours=1),
                post_type="trade_result", ai_generated=True,
                ai_model="claude-sonnet-4-5-20250929",
                hashtags="#PLTR #AlgoTrading #Ensemble #RigaCap #WalkForward",
                image_s3_key="social/images/75_SLV_20260116.png",
            )
            await _try("post_notification_instagram", admin_email_service.send_post_approval_notification(
                to_email=to, post=insta_post, hours_before=1,
                cancel_url="https://rigacap.com/api/admin/social/posts/998/cancel-email?token=test-preview",
            ))

            # 15-19. Onboarding drip emails (steps 1-5)
            for step in range(1, 6):
                await _try(f"onboarding_{step}", email_service.send_onboarding_email(
                    step=step, to_email=to, name="Erik Kinsman",
                    user_id=test_user_id,
                ))

            # 20. Referral Reward
            await _try("referral_reward", email_service.send_referral_reward_email(
                to_email=to, name="Erik Kinsman", friend_name="Jane Doe",
            ))

            sent_count = sum(1 for v in results.values() if v == "sent")
            return {"status": "success", "sent": sent_count, "total": len(results), "results": results, "to": to}

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_test_emails())
            return result
        except Exception as e:
            import traceback
            print(f"❌ Test emails failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Handle bulk chart card regeneration (direct Lambda invocation)
    if event.get("regenerate_charts"):
        print("🎨 Regenerate chart cards request received")

        async def _regenerate_charts():
            from sqlalchemy import select
            from app.core.database import async_session, SocialPost
            from app.services.chart_card_generator import chart_card_generator
            from app.services.scanner import scanner_service
            import json

            # Load price data so charts have real price lines
            if not scanner_service.data_cache:
                print("📊 Loading price data for chart rendering...")
                await scanner_service.fetch_data(period="1y")
                print(f"📊 Loaded {len(scanner_service.data_cache)} symbols")

            async with async_session() as db:
                result = await db.execute(
                    select(SocialPost).where(
                        SocialPost.platform == "instagram",
                        SocialPost.image_metadata_json.isnot(None),
                    )
                )
                posts = result.scalars().all()

                regenerated = 0
                errors = []
                for post in posts:
                    try:
                        meta = json.loads(post.image_metadata_json)
                        png_bytes = chart_card_generator.generate_trade_card(
                            symbol=meta.get("symbol", "???"),
                            entry_price=meta.get("entry_price", 0),
                            exit_price=meta.get("exit_price", 0),
                            entry_date=meta.get("entry_date", ""),
                            exit_date=meta.get("exit_date", ""),
                            pnl_pct=meta.get("pnl_pct", 0),
                            pnl_dollars=meta.get("pnl_dollars", 0),
                            exit_reason=meta.get("exit_reason", "trailing_stop"),
                            strategy_name=meta.get("strategy_name", "Ensemble"),
                            regime_name=meta.get("regime_name", ""),
                            company_name=meta.get("company_name", ""),
                        )
                        date_str = meta.get("exit_date", "")[:10].replace("-", "")
                        s3_key = chart_card_generator.upload_to_s3(
                            png_bytes, post.id, meta.get("symbol", "UNK"), date_str
                        )
                        if s3_key:
                            post.image_s3_key = s3_key
                            regenerated += 1
                            print(f"  ✅ Post {post.id} ({meta.get('symbol')}): {s3_key}")
                        else:
                            errors.append(f"Post {post.id}: S3 upload failed")
                    except Exception as e:
                        errors.append(f"Post {post.id}: {str(e)}")
                        print(f"  ❌ Post {post.id}: {e}")

                await db.commit()

            return {
                "status": "success",
                "total_posts": len(posts),
                "regenerated": regenerated,
                "errors": errors,
            }

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_regenerate_charts())
            return result
        except Exception as e:
            import traceback
            print(f"❌ Regenerate charts failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Handle daily email digest (EventBridge: 6 PM ET Mon-Fri)
    if event.get("daily_emails"):
        print("📧 Daily email digest triggered")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(scheduler_service.send_daily_emails())

            # Export dashboard cache + daily snapshot after daily emails
            async def _export_dashboard_and_snapshot():
                from app.api.signals import compute_shared_dashboard_data
                from app.services.data_export import data_export_service
                from datetime import date
                async with async_session() as db:
                    data = await compute_shared_dashboard_data(db)
                    dash_result = data_export_service.export_dashboard_json(data)
                    # Also save today's snapshot for time-travel mode
                    today_str = date.today().strftime("%Y-%m-%d")
                    snap_result = data_export_service.export_snapshot(today_str, data)
                    return {"dashboard": dash_result, "snapshot": snap_result}

            try:
                export_result = loop.run_until_complete(_export_dashboard_and_snapshot())
                print(f"📦 Dashboard cache exported: {export_result.get('dashboard')}")
                print(f"📸 Daily snapshot saved: {export_result.get('snapshot')}")
            except Exception as dash_err:
                print(f"⚠️ Dashboard/snapshot export failed (non-fatal): {dash_err}")

            return {"status": "success", "result": str(result)}
        except Exception as e:
            import traceback
            print(f"❌ Daily emails failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Handle double signal alerts (EventBridge: 5 PM ET Mon-Fri)
    if event.get("double_signal_alerts"):
        print("🔔 Double signal alerts triggered")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(scheduler_service.check_double_signal_alerts())
            return {"status": "success", "result": str(result)}
        except Exception as e:
            import traceback
            print(f"❌ Double signal alerts failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Handle ticker health check (EventBridge: 7 AM ET Mon-Fri)
    if event.get("ticker_health_check"):
        print("🩺 Ticker health check triggered")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(scheduler_service.check_ticker_health())
            return {"status": "success", "result": str(result)}
        except Exception as e:
            import traceback
            print(f"❌ Ticker health check failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Handle publish scheduled posts (EventBridge: every 15 min)
    if event.get("publish_scheduled_posts"):
        print("📤 Publish scheduled posts triggered")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(scheduler_service._publish_scheduled_posts())
            return {"status": "success", "result": str(result)}
        except Exception as e:
            import traceback
            print(f"❌ Publish scheduled posts failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Handle post notifications (EventBridge: every hour)
    if event.get("post_notifications"):
        print("🔔 Post notifications triggered")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(scheduler_service._send_post_notifications())
            return {"status": "success", "result": str(result)}
        except Exception as e:
            import traceback
            print(f"❌ Post notifications failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Handle strategy auto-analysis (EventBridge: Fri 6:30 PM ET)
    if event.get("strategy_auto_analysis"):
        print("📊 Strategy auto-analysis triggered")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(scheduler_service._strategy_auto_analysis())
            return {"status": "success", "result": str(result)}
        except Exception as e:
            import traceback
            print(f"❌ Strategy auto-analysis failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Handle onboarding drip emails (EventBridge: 10 AM ET daily)
    if event.get("onboarding_drip"):
        print("📧 Onboarding drip emails triggered")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(scheduler_service.send_onboarding_drip_emails())
            return {"status": "success", "result": result}
        except Exception as e:
            import traceback
            print(f"❌ Onboarding drip failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Handle seed strategies (direct Lambda invocation)
    if event.get("seed_strategies"):
        print("🌱 Seed strategies request received")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_seed_and_list_strategies())
            return result
        except Exception as e:
            import traceback
            print(f"❌ Seed strategies failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Handle list strategies (direct Lambda invocation)
    if event.get("list_strategies"):
        print("📋 List strategies request received")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_list_strategies())
            return result
        except Exception as e:
            import traceback
            print(f"❌ List strategies failed: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    # Handle snapshot backfill (direct Lambda invocation)
    if event.get("snapshot_backfill_job"):
        print(f"📸 Snapshot backfill job received - {len(scanner_service.data_cache)} symbols in cache")
        job_config = event["snapshot_backfill_job"]
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _run_snapshot_backfill():
            import pandas as pd
            from app.services.data_export import data_export_service
            from app.api.signals import _compute_dashboard_live

            spy_df = scanner_service.data_cache.get('SPY')
            if spy_df is None:
                return {"status": "failed", "error": "SPY data not in cache"}

            print(f"📸 SPY shape={spy_df.shape}, index type={type(spy_df.index).__name__}, "
                  f"tz={getattr(spy_df.index, 'tz', None)}, "
                  f"first={spy_df.index[0]}, last={spy_df.index[-1]}, "
                  f"cols={list(spy_df.columns[:5])}")

            start_ts = pd.Timestamp(job_config["start_date"])
            end_ts = pd.Timestamp(job_config["end_date"])

            if hasattr(spy_df.index, 'tz') and spy_df.index.tz is not None:
                start_ts = start_ts.tz_localize(spy_df.index.tz)
                end_ts = end_ts.tz_localize(spy_df.index.tz)

            trading_days = spy_df.loc[start_ts:end_ts].index
            total = len(trading_days)
            print(f"📸 Backfill: {total} trading days from {job_config['start_date']} to {job_config['end_date']}")

            saved = 0
            skipped = 0
            errors = 0

            async with async_session() as db:
                for i, ts in enumerate(trading_days):
                    date_str = ts.strftime('%Y-%m-%d')

                    existing = data_export_service.read_snapshot(date_str)
                    if existing:
                        skipped += 1
                        print(f"  [{i+1}/{total}] {date_str} — already exists, skipping")
                        continue

                    try:
                        data = await _compute_dashboard_live(
                            db=db, user=None, momentum_top_n=30,
                            fresh_days=5, as_of_date=date_str,
                        )
                        result = data_export_service.export_snapshot(date_str, data)
                        if result.get("success"):
                            saved += 1
                            print(f"  [{i+1}/{total}] {date_str} — saved")
                        else:
                            errors += 1
                            print(f"  [{i+1}/{total}] {date_str} — export failed: {result.get('message')}")
                    except Exception as e:
                        errors += 1
                        print(f"  [{i+1}/{total}] {date_str} — error: {e}")

            return {
                "status": "completed",
                "total_trading_days": total,
                "saved": saved,
                "skipped": skipped,
                "errors": errors,
            }

        try:
            result = loop.run_until_complete(_run_snapshot_backfill())
            print(f"📸 Snapshot backfill: {result}")
            return result
        except Exception as e:
            import traceback
            print(f"❌ Snapshot backfill failed: {e}")
            traceback.print_exc()
            return {"status": "failed", "error": str(e)}

    # Handle intraday crossover simulation (direct Lambda invoke for testing)
    if event.get("simulate_intraday_crossover"):
        config = event["simulate_intraday_crossover"]
        as_of_date = config.get("as_of_date")
        do_send_email = config.get("send_email", False)
        print(f"📡 Simulating intraday crossover for {as_of_date}")

        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _run_simulation():
            import pandas as pd
            from app.services.stock_universe import stock_universe_service
            from app.services.email_service import email_service

            effective_date = pd.Timestamp(as_of_date).normalize()

            def _truncate(df, ts):
                if hasattr(df.index, 'tz') and df.index.tz is not None and ts.tz is None:
                    ts = ts.tz_localize(df.index.tz)
                return df[df.index <= ts]

            spy_df = scanner_service.data_cache.get('SPY')
            if spy_df is None:
                return {"error": "SPY data not available"}

            spy_trunc = _truncate(spy_df, effective_date)
            if len(spy_trunc) < 2:
                return {"error": "Not enough data for this date"}

            prev_trading_day = spy_trunc.index[-2]
            prev_ts = prev_trading_day.tz_localize(None) if hasattr(prev_trading_day, 'tz') and prev_trading_day.tz else prev_trading_day

            # Watchlist as of previous trading day
            momentum_rankings = scanner_service.rank_stocks_momentum(
                apply_market_filter=True, as_of_date=prev_ts
            )
            top_momentum = {
                r.symbol: {'rank': i + 1, 'data': r}
                for i, r in enumerate(momentum_rankings[:30])
            }

            # Check ALL top-30 momentum stocks: below +5% prev day → above +5% today
            # This is broader than the narrow 3-5% watchlist to catch gap-ups
            prev_watchlist = []  # narrow 3-5% watchlist
            all_crossovers = []  # any stock that crossed +5% that day

            for symbol, mom in top_momentum.items():
                df = scanner_service.data_cache.get(symbol)
                if df is None or len(df) < 200:
                    continue
                df_prev = _truncate(df, prev_ts)
                if len(df_prev) < 1:
                    continue
                row = df_prev.iloc[-1]
                price = float(row['close'])
                dwap_val = row.get('dwap')
                if pd.isna(dwap_val) or dwap_val <= 0:
                    continue
                prev_pct = (price / dwap_val - 1) * 100

                # Track narrow watchlist (3-5%)
                if 3.0 <= prev_pct < 5.0:
                    prev_watchlist.append({
                        'symbol': symbol,
                        'prev_day_price': round(price, 2),
                        'dwap': round(float(dwap_val), 2),
                        'prev_day_pct_above': round(prev_pct, 2),
                        'distance_to_trigger': round(5.0 - prev_pct, 2),
                        'momentum_rank': mom['rank'],
                    })

                # Check if crossed +5% on as_of_date (broad check)
                if prev_pct < 5.0:
                    df_today = _truncate(df, effective_date)
                    if len(df_today) < 1:
                        continue
                    today_row = df_today.iloc[-1]
                    today_price = float(today_row['close'])
                    today_dwap = today_row.get('dwap')
                    if pd.isna(today_dwap) or today_dwap <= 0:
                        today_dwap = dwap_val
                    today_pct = (today_price / float(today_dwap) - 1) * 100
                    info = stock_universe_service.symbol_info.get(symbol, {})
                    all_crossovers.append({
                        'symbol': symbol,
                        'prev_day_price': round(price, 2),
                        'prev_day_pct_above': round(prev_pct, 2),
                        'as_of_date_price': round(today_price, 2),
                        'as_of_date_dwap': round(float(today_dwap), 2),
                        'as_of_date_pct_above': round(today_pct, 2),
                        'crossed': today_pct >= 5.0,
                        'was_on_watchlist': 3.0 <= prev_pct < 5.0,
                        'momentum_rank': mom['rank'],
                        'sector': info.get('sector', ''),
                    })

            prev_watchlist.sort(key=lambda x: x['distance_to_trigger'])
            prev_watchlist = prev_watchlist[:5]

            triggered = [c for c in all_crossovers if c['crossed']]

            # If force_example is set and no real crossovers found, create a
            # synthetic one using a specified symbol for email template testing
            example_symbol = config.get("force_example")
            if example_symbol and not triggered:
                df = scanner_service.data_cache.get(example_symbol)
                if df is not None and len(df) >= 200:
                    df_today = _truncate(df, effective_date)
                    if len(df_today) >= 2:
                        today_row = df_today.iloc[-1]
                        prev_row = df_today.iloc[-2]
                        today_dwap = today_row.get('dwap')
                        if not pd.isna(today_dwap) and today_dwap > 0:
                            info = stock_universe_service.symbol_info.get(example_symbol, {})
                            triggered.append({
                                'symbol': example_symbol,
                                'prev_day_price': round(float(prev_row['close']), 2),
                                'prev_day_pct_above': round((float(prev_row['close']) / float(prev_row.get('dwap', today_dwap)) - 1) * 100, 2),
                                'as_of_date_price': round(float(today_row['close']), 2),
                                'as_of_date_dwap': round(float(today_dwap), 2),
                                'as_of_date_pct_above': round((float(today_row['close']) / float(today_dwap) - 1) * 100, 2),
                                'crossed': True,
                                'was_on_watchlist': False,
                                'momentum_rank': 20,
                                'sector': info.get('sector', ''),
                                'synthetic': True,
                            })

            emails_sent = []
            if do_send_email and triggered:
                admin_email = config.get("email", "erik@rigacap.com")
                for sig in triggered:
                    success = await email_service.send_intraday_signal_alert(
                        to_email=admin_email,
                        user_name="Erik",
                        symbol=sig['symbol'],
                        live_price=sig['as_of_date_price'],
                        dwap=sig['as_of_date_dwap'],
                        pct_above_dwap=sig['as_of_date_pct_above'],
                        momentum_rank=sig['momentum_rank'],
                        sector=sig['sector'],
                    )
                    emails_sent.append({
                        'symbol': sig['symbol'],
                        'sent_to': admin_email,
                        'success': success,
                    })

            return {
                'simulation_date': as_of_date,
                'prev_trading_day': prev_ts.strftime('%Y-%m-%d'),
                'watchlist_prev_day': prev_watchlist,
                'all_crossovers': all_crossovers,
                'triggered': triggered,
                'triggered_count': len(triggered),
                'from_watchlist_count': len([t for t in triggered if t.get('was_on_watchlist')]),
                'gap_up_count': len([t for t in triggered if not t.get('was_on_watchlist')]),
                'emails_sent': emails_sent or None,
            }

        try:
            result = loop.run_until_complete(_run_simulation())
            print(f"📡 Simulation result: {result}")
            return result
        except Exception as e:
            import traceback
            print(f"❌ Intraday simulation failed: {e}")
            traceback.print_exc()
            return {"status": "failed", "error": str(e)}

    # Handle email template preview — send all templates to a given email
    if event.get("send_all_email_templates"):
        config = event["send_all_email_templates"]
        to_email = config.get("email", "erik@rigacap.com")
        print(f"📧 Sending all email templates to {to_email}")

        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _send_all_templates():
            from app.services.email_service import email_service
            results = []

            # 1. Welcome email
            try:
                ok = await email_service.send_welcome_email(to_email, "Erik")
                results.append({"template": "welcome", "success": ok})
            except Exception as e:
                results.append({"template": "welcome", "error": str(e)})

            # 2. Daily summary
            try:
                sample_signals = [
                    {'symbol': 'NVDA', 'price': 142.50, 'pct_above_dwap': 8.2, 'is_strong': True,
                     'momentum_rank': 1, 'ensemble_score': 85.0, 'dwap_crossover_date': '2026-02-13',
                     'days_since_crossover': 0, 'is_fresh': True},
                    {'symbol': 'AVT', 'price': 58.30, 'pct_above_dwap': 5.4, 'is_strong': False,
                     'momentum_rank': 20, 'ensemble_score': 52.0, 'dwap_crossover_date': '2026-02-12',
                     'days_since_crossover': 1, 'is_fresh': True},
                ]
                ok = await email_service.send_daily_summary(
                    to_email=to_email,
                    signals=sample_signals,
                    market_regime={'regime': 'strong_bull', 'spy_price': 605.20},
                    watchlist=[{'symbol': 'KLAC', 'price': 810.50, 'pct_above_dwap': 4.2, 'distance_to_trigger': 0.8}],
                )
                results.append({"template": "daily_summary", "success": ok})
            except Exception as e:
                results.append({"template": "daily_summary", "error": str(e)})

            # 3. Double signal alert (breakout)
            try:
                ok = await email_service.send_double_signal_alert(
                    to_email=to_email,
                    new_signals=[
                        {'symbol': 'NVDA', 'price': 142.50, 'pct_above_dwap': 8.2,
                         'momentum_rank': 1, 'short_momentum': 12.5, 'dwap_crossover_date': '2026-02-13',
                         'days_since_crossover': 0},
                    ],
                    approaching=[
                        {'symbol': 'KLAC', 'price': 810.50, 'pct_above_dwap': 4.2, 'distance_to_trigger': 0.8},
                    ],
                    market_regime={'regime': 'strong_bull', 'spy_price': 605.20},
                )
                results.append({"template": "double_signal_alert", "success": ok})
            except Exception as e:
                results.append({"template": "double_signal_alert", "error": str(e)})

            # 4. Sell alert
            try:
                ok = await email_service.send_sell_alert(
                    to_email=to_email,
                    user_name="Erik",
                    symbol="TSLA",
                    action="sell",
                    reason="Trailing stop hit: price dropped 15% from high",
                    current_price=245.80,
                    entry_price=220.00,
                    stop_price=238.50,
                )
                results.append({"template": "sell_alert", "success": ok})
            except Exception as e:
                results.append({"template": "sell_alert", "error": str(e)})

            # 5. Sell warning
            try:
                ok = await email_service.send_sell_alert(
                    to_email=to_email,
                    user_name="Erik",
                    symbol="AAPL",
                    action="warning",
                    reason="Approaching trailing stop: 2% away",
                    current_price=198.50,
                    entry_price=185.00,
                    stop_price=195.00,
                )
                results.append({"template": "sell_warning", "success": ok})
            except Exception as e:
                results.append({"template": "sell_warning", "error": str(e)})

            # 6. Trial ending
            try:
                ok = await email_service.send_trial_ending_email(
                    to_email=to_email,
                    name="Erik",
                    days_remaining=3,
                )
                results.append({"template": "trial_ending", "success": ok})
            except Exception as e:
                results.append({"template": "trial_ending", "error": str(e)})

            # 7. Goodbye
            try:
                ok = await email_service.send_goodbye_email(to_email, "Erik")
                results.append({"template": "goodbye", "success": ok})
            except Exception as e:
                results.append({"template": "goodbye", "error": str(e)})

            # 8. Intraday signal alert (NEW)
            try:
                ok = await email_service.send_intraday_signal_alert(
                    to_email=to_email,
                    user_name="Erik",
                    symbol="AVT",
                    live_price=58.30,
                    dwap=55.20,
                    pct_above_dwap=5.6,
                    momentum_rank=20,
                    sector="Technology",
                )
                results.append({"template": "intraday_signal_alert", "success": ok})
            except Exception as e:
                results.append({"template": "intraday_signal_alert", "error": str(e)})

            return {"emails_sent_to": to_email, "results": results}

        try:
            result = loop.run_until_complete(_send_all_templates())
            print(f"📧 All templates sent: {result}")
            return result
        except Exception as e:
            import traceback
            print(f"❌ Template send failed: {e}")
            traceback.print_exc()
            return {"status": "failed", "error": str(e)}

    # Social post admin (direct Lambda invocation)
    # Actions: list, approve, publish, approve_and_publish, delete, attach_image
    if event.get("social_admin"):
        config = event["social_admin"]
        action = config.get("action", "list")

        async def _social_admin():
            from app.core.database import async_session, SocialPost
            from sqlalchemy import select, desc

            async with async_session() as db:
                if action == "list":
                    limit = config.get("limit", 20)
                    status_filter = config.get("status")
                    q = select(SocialPost).order_by(desc(SocialPost.id)).limit(limit)
                    if status_filter:
                        q = q.where(SocialPost.status == status_filter)
                    result = await db.execute(q)
                    posts = result.scalars().all()
                    return {
                        "posts": [
                            {
                                "id": p.id,
                                "platform": p.platform,
                                "post_type": p.post_type,
                                "status": p.status,
                                "text_preview": (p.text_content or "")[:100],
                                "scheduled_for": str(p.scheduled_for) if p.scheduled_for else None,
                                "created_at": str(p.created_at),
                            }
                            for p in posts
                        ]
                    }

                elif action in ("approve", "publish", "approve_and_publish"):
                    post_id = config.get("post_id")
                    if not post_id:
                        return {"error": "post_id required"}

                    result = await db.execute(
                        select(SocialPost).where(SocialPost.id == post_id)
                    )
                    post = result.scalar_one_or_none()
                    if not post:
                        return {"error": f"Post {post_id} not found"}

                    if action in ("approve", "approve_and_publish"):
                        if post.status not in ("draft", "rejected", "scheduled"):
                            return {"error": f"Cannot approve post with status '{post.status}'"}
                        post.status = "approved"
                        post.scheduled_for = None
                        await db.commit()
                        print(f"✅ Post {post_id} approved")

                    if action in ("publish", "approve_and_publish"):
                        if post.status != "approved":
                            return {"error": f"Post must be approved first (current: '{post.status}')"}
                        from app.services.social_posting_service import social_posting_service
                        pub_result = await social_posting_service.publish_post(post)
                        await db.commit()
                        if "error" in pub_result:
                            return {"error": pub_result["error"]}
                        return {"status": "published", "post_id": post_id, "platform": post.platform, **pub_result}

                    return {"status": post.status, "post_id": post_id}

                elif action == "delete":
                    post_ids = config.get("post_ids")
                    if not post_ids:
                        return {"error": "post_ids required (list of IDs)"}
                    from sqlalchemy import delete as sa_delete
                    await db.execute(sa_delete(SocialPost).where(SocialPost.id.in_(post_ids)))
                    await db.commit()
                    print(f"🗑️ Deleted {len(post_ids)} posts: {post_ids}")
                    return {"deleted": len(post_ids)}

                elif action == "attach_image":
                    post_id = config.get("post_id")
                    image_s3_key = config.get("image_s3_key")
                    if not post_id or not image_s3_key:
                        return {"error": "post_id and image_s3_key required"}
                    result = await db.execute(
                        select(SocialPost).where(SocialPost.id == post_id)
                    )
                    post = result.scalar_one_or_none()
                    if not post:
                        return {"error": f"Post {post_id} not found"}
                    post.image_s3_key = image_s3_key
                    await db.commit()
                    print(f"🖼️ Attached image to post {post_id}: {image_s3_key}")
                    return {"post_id": post_id, "image_s3_key": image_s3_key}

                elif action == "bulk_schedule":
                    # Schedule multiple posts: [{"post_id": 97, "publish_at": "2026-02-19T14:00:00"}, ...]
                    schedule_list = config.get("posts", [])
                    if not schedule_list:
                        return {"error": "posts list required"}
                    results = []
                    for item in schedule_list:
                        pid = item.get("post_id")
                        pub_at = item.get("publish_at")
                        try:
                            publish_at = datetime.fromisoformat(pub_at)
                            r = await db.execute(
                                select(SocialPost).where(SocialPost.id == pid)
                            )
                            post = r.scalar_one_or_none()
                            if not post:
                                results.append({"post_id": pid, "scheduled": False, "error": "not found"})
                                continue
                            if post.status not in ("draft", "approved"):
                                results.append({"post_id": pid, "scheduled": False, "error": f"status is {post.status}"})
                                continue
                            post.status = "scheduled"
                            post.scheduled_for = publish_at
                            results.append({"post_id": pid, "scheduled": True, "publish_at": pub_at})
                        except Exception as e:
                            results.append({"post_id": pid, "scheduled": False, "error": str(e)})
                    await db.commit()
                    return {"scheduled": results}

                elif action == "follow_accounts":
                    # Batch follow Twitter accounts: {"usernames": ["unusual_whales", "PeterLBrandt", ...]}
                    usernames = config.get("usernames", [])
                    if not usernames:
                        return {"error": "usernames list required"}
                    from app.services.social_posting_service import social_posting_service
                    return await social_posting_service.batch_follow_twitter(usernames)

                else:
                    return {"error": f"Unknown action: {action}"}

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_social_admin())
            return result
        except Exception as e:
            import traceback
            print(f"❌ Social admin failed: {e}")
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    # Refresh Instagram long-lived token (scheduled weekly via EventBridge)
    if event.get("refresh_instagram_token"):
        print("🔄 Refreshing Instagram access token")
        import httpx

        from app.core.config import settings

        app_id = settings.META_APP_ID
        app_secret = settings.META_APP_SECRET
        current_token = settings.INSTAGRAM_ACCESS_TOKEN

        if not all([app_id, app_secret, current_token]):
            return {"status": "error", "error": "META_APP_ID, META_APP_SECRET, or INSTAGRAM_ACCESS_TOKEN not set"}

        try:
            resp = httpx.get(
                "https://graph.facebook.com/v24.0/oauth/access_token",
                params={
                    "grant_type": "fb_exchange_token",
                    "client_id": app_id,
                    "client_secret": app_secret,
                    "fb_exchange_token": current_token,
                },
                timeout=30,
            )
            data = resp.json()

            if "access_token" not in data:
                print(f"❌ Token refresh failed: {data}")
                # Send admin alert
                try:
                    from app.services.email_service import admin_email_service
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    loop.run_until_complete(
                        admin_email_service.send_admin_alert(
                            "Instagram Token Refresh Failed",
                            f"Token refresh failed. Error: {data}. "
                            "Please manually regenerate at developers.facebook.com"
                        )
                    )
                except Exception:
                    pass
                return {"status": "error", "error": str(data)}

            new_token = data["access_token"]
            expires_in = data.get("expires_in", 0)
            expires_days = expires_in // 86400

            # Update Lambda env var with new token
            import boto3
            lambda_client = boto3.client("lambda", region_name=settings.AWS_REGION)
            func_config = lambda_client.get_function_configuration(
                FunctionName=os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "rigacap-prod-api")
            )
            env_vars = func_config.get("Environment", {}).get("Variables", {})
            env_vars["INSTAGRAM_ACCESS_TOKEN"] = new_token
            lambda_client.update_function_configuration(
                FunctionName=os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "rigacap-prod-api"),
                Environment={"Variables": env_vars},
            )

            print(f"✅ Instagram token refreshed, expires in {expires_days} days")
            return {
                "status": "refreshed",
                "expires_in_days": expires_days,
            }
        except Exception as e:
            import traceback
            print(f"❌ Token refresh error: {e}")
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    # For API Gateway events, use Mangum
    # Create a fresh Mangum handler to avoid event loop issues on warm Lambdas
    global _mangum_handler

    # Check if event loop is closed and reset if needed
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
            _mangum_handler = None  # Force recreation
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
        _mangum_handler = None

    if _mangum_handler is None:
        _mangum_handler = Mangum(app, lifespan="off")

    return _mangum_handler(event, context)


# ============================================================================
# Health & Config Endpoints
# ============================================================================

@app.get("/")
async def root(admin: User = Depends(get_admin_user)):
    return {
        "name": "RigaCap API",
        "version": "2.0.0",
        "docs": "/docs",
        "data_loaded": len(scanner_service.data_cache),
        "last_scan": scanner_service.last_scan.isoformat() if scanner_service.last_scan else None
    }


@app.get("/health")
async def health(user: User = Depends(get_current_user)):
    scheduler_status = scheduler_service.get_status()

    # Use local cache if available, otherwise get metadata from S3
    symbols_loaded = len(scanner_service.data_cache)
    last_scan = scanner_service.last_scan.isoformat() if scanner_service.last_scan else None

    # If no local data, try to get metadata from S3
    if symbols_loaded == 0:
        try:
            status = data_export_service.get_status()
            symbols_loaded = status.get("files_count", 0)
            if status.get("last_export"):
                last_scan = status.get("last_export")
        except Exception:
            pass  # Ignore errors, just use default values

    return {
        "status": "healthy",
        "symbols_loaded": symbols_loaded,
        "last_scan": last_scan,
        "scheduler": {
            "running": scheduler_status["is_running"],
            "last_run": scheduler_status["last_run"],
            "next_runs": scheduler_status["next_runs"]
        }
    }


@app.post("/api/warmup")
async def warmup(admin: User = Depends(get_admin_user)):
    """
    Warm up Lambda by loading all data.
    Call this after deployment to preload data so user requests are fast.
    """
    results = {
        "universe": {"loaded": 0, "status": "pending"},
        "price_data": {"loaded": 0, "status": "pending"},
        "consolidated": {"status": "pending"}
    }

    # 1. Load universe
    try:
        await scanner_service.ensure_universe_loaded()
        results["universe"] = {
            "loaded": len(scanner_service.universe),
            "status": "success"
        }
    except Exception as e:
        results["universe"] = {"loaded": 0, "status": f"error: {e}"}

    # 2. Load price data from S3
    try:
        if not scanner_service.data_cache:
            cached_data = data_export_service.import_all()
            if cached_data:
                scanner_service.data_cache = cached_data
        results["price_data"] = {
            "loaded": len(scanner_service.data_cache),
            "status": "success"
        }
    except Exception as e:
        results["price_data"] = {"loaded": 0, "status": f"error: {e}"}

    # 3. Report consolidated status (don't export - could overwrite larger S3 file)
    if scanner_service.data_cache and len(scanner_service.data_cache) > 0:
        try:
            status = data_export_service.get_status()
            results["consolidated"] = {
                "success": True,
                "count": len(scanner_service.data_cache),
                "s3_status": status
            }
        except Exception as e:
            results["consolidated"] = {"status": f"error: {e}"}

    return {
        "status": "warmed",
        "results": results
    }


@app.post("/api/data/load-batch")
async def load_batch(batch_size: int = 50, admin: User = Depends(get_admin_user)):
    """
    Load a batch of stocks that aren't already cached.
    Call this repeatedly to gradually build up the full dataset.

    Args:
        batch_size: Number of stocks to load per call (default 50, max 100)
    """
    batch_size = min(batch_size, 100)  # Cap at 100 to avoid timeout

    # Ensure universe is loaded
    await scanner_service.ensure_universe_loaded()

    # Find symbols not yet cached
    cached_symbols = set(scanner_service.data_cache.keys())
    all_symbols = set(scanner_service.universe)
    missing_symbols = list(all_symbols - cached_symbols)

    if not missing_symbols:
        return {
            "status": "complete",
            "message": "All stocks already loaded",
            "total_cached": len(cached_symbols),
            "total_universe": len(all_symbols)
        }

    # Take a batch
    batch = missing_symbols[:batch_size]

    # Fetch data for this batch
    try:
        await scanner_service.fetch_data(batch)
        newly_loaded = len([s for s in batch if s in scanner_service.data_cache])

        # Save progress to S3
        export_result = data_export_service.export_consolidated(scanner_service.data_cache)

        return {
            "status": "progress",
            "batch_requested": len(batch),
            "batch_loaded": newly_loaded,
            "total_cached": len(scanner_service.data_cache),
            "total_universe": len(all_symbols),
            "remaining": len(missing_symbols) - newly_loaded,
            "export": export_result
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "total_cached": len(scanner_service.data_cache)
        }


@app.get("/api/data/status")
async def data_status(admin: User = Depends(get_admin_user)):
    """Get current data loading status"""
    await scanner_service.ensure_universe_loaded()

    cached_symbols = set(scanner_service.data_cache.keys())
    all_symbols = set(scanner_service.universe)

    return {
        "total_universe": len(all_symbols),
        "total_cached": len(cached_symbols),
        "remaining": len(all_symbols - cached_symbols),
        "percent_complete": round(len(cached_symbols) / len(all_symbols) * 100, 1) if all_symbols else 0,
        "cached_symbols_sample": list(cached_symbols)[:20]
    }


@app.get("/api/config")
async def get_config(admin: User = Depends(get_admin_user)):
    return {
        "dwap_threshold_pct": settings.DWAP_THRESHOLD_PCT,
        "stop_loss_pct": settings.STOP_LOSS_PCT,
        "profit_target_pct": settings.PROFIT_TARGET_PCT,
        "max_positions": settings.MAX_POSITIONS,
        "position_size_pct": settings.POSITION_SIZE_PCT,
        "min_volume": settings.MIN_VOLUME,
        "min_price": settings.MIN_PRICE,
        "universe_size": len(scanner_service.universe),
        "full_universe_loaded": scanner_service.full_universe_loaded
    }


@app.get("/api/debug/yfinance")
async def debug_yfinance(admin: User = Depends(get_admin_user)):
    """Debug endpoint to test yfinance import"""
    result = {"yfinance_available": False, "error": None, "version": None}
    try:
        import yfinance as yf
        result["yfinance_available"] = True
        result["version"] = yf.__version__
        # Try a simple download
        ticker = yf.Ticker("AAPL")
        info = ticker.fast_info
        result["test_ticker"] = "AAPL"
        result["test_price"] = info.last_price if hasattr(info, 'last_price') else None
    except Exception as e:
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
    return result


@app.post("/api/universe/load-full")
async def load_full_universe(admin: User = Depends(get_admin_user)):
    """
    Load the full NASDAQ + NYSE stock universe (~6000 stocks)

    This replaces the default 80-stock curated list.
    Note: Initial data fetch will take several minutes.
    """
    try:
        symbols = await scanner_service.load_full_universe()
        return {
            "success": True,
            "universe_size": len(symbols),
            "message": f"Loaded {len(symbols)} stocks from NASDAQ + NYSE"
        }
    except Exception as e:
        logger.error(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/universe/status")
async def get_universe_status(admin: User = Depends(get_admin_user)):
    """Get current universe status"""
    return {
        "universe_size": len(scanner_service.universe),
        "full_universe_loaded": scanner_service.full_universe_loaded,
        "symbols_with_data": len(scanner_service.data_cache),
        "sample_symbols": scanner_service.universe[:20] if scanner_service.universe else []
    }


@app.post("/api/data/load")
async def load_market_data(symbols: Optional[str] = None, period: str = "5y", admin: User = Depends(get_admin_user)):
    """
    Manually trigger market data loading

    Args:
        symbols: Optional comma-separated list of symbols (default: all in universe)
        period: Data period (1y, 2y, 5y)
    """
    try:
        # Parse comma-separated symbols
        symbol_list = None
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",")]
        await scanner_service.fetch_data(symbols=symbol_list, period=period)
        return {
            "success": True,
            "symbols_loaded": len(scanner_service.data_cache),
            "message": f"Loaded data for {len(scanner_service.data_cache)} symbols"
        }
    except Exception as e:
        return {
            "success": False,
            "symbols_loaded": len(scanner_service.data_cache),
            "error": str(e),
            "message": "Partial load completed or Yahoo Finance rate limited"
        }


@app.get("/api/data/status")
async def get_data_status(admin: User = Depends(get_admin_user)):
    """Get current data loading status"""
    return {
        "data_available": len(scanner_service.data_cache) > 0,
        "symbols_loaded": len(scanner_service.data_cache),
        "universe_size": len(scanner_service.universe),
        "last_scan": scanner_service.last_scan.isoformat() if scanner_service.last_scan else None,
        "loaded_symbols": list(scanner_service.data_cache.keys())[:50] if scanner_service.data_cache else []
    }


@app.post("/api/data/export")
async def export_data(admin: User = Depends(get_admin_user)):
    """
    Export all cached price data to individual CSV files

    This saves historical data permanently so it never needs to be re-fetched.
    """
    if not scanner_service.data_cache:
        raise HTTPException(status_code=400, detail="No data to export. Load data first.")

    result = data_export_service.export_all(scanner_service.data_cache)
    return result


@app.post("/api/data/export-consolidated")
async def export_data_consolidated(admin: User = Depends(get_admin_user)):
    """
    Export all cached price data to a single consolidated gzipped CSV.

    Much faster to load than individual files. Use this before deployments
    to save the latest data, and call /api/warmup after deployment to reload.
    """
    if not scanner_service.data_cache:
        raise HTTPException(status_code=400, detail="No data to export. Load data first.")

    result = data_export_service.export_consolidated(scanner_service.data_cache)
    return result


@app.post("/api/data/pre-deploy")
async def pre_deploy_export(admin: User = Depends(get_admin_user)):
    """
    Pre-deployment export: Save all current price data to S3.

    Call this BEFORE deploying new code to ensure no data is lost.
    The warmup endpoint after deployment will reload this data.
    """
    if not scanner_service.data_cache:
        return {"status": "skip", "message": "No data in cache to export"}

    # Export consolidated file
    result = data_export_service.export_consolidated(scanner_service.data_cache)

    return {
        "status": "success" if result.get("success") else "failed",
        "symbols_exported": result.get("count", 0),
        "size_mb": result.get("total_size_mb", 0),
        "message": "Data saved to S3. Safe to deploy."
    }


@app.get("/api/data/export-status")
async def get_export_status(admin: User = Depends(get_admin_user)):
    """Get status of exported data files"""
    return data_export_service.get_status()


@app.post("/api/data/import")
async def import_data(admin: User = Depends(get_admin_user)):
    """
    Import price data from Parquet files into memory

    This is called automatically on startup, but can be triggered manually.
    """
    cached_data = data_export_service.import_all()
    if cached_data:
        scanner_service.data_cache = cached_data
        return {
            "success": True,
            "symbols_loaded": len(cached_data),
            "message": f"Imported {len(cached_data)} symbols from parquet files"
        }
    return {
        "success": False,
        "symbols_loaded": 0,
        "message": "No parquet files found to import"
    }


# ============================================================================
# Market Analysis Endpoints
# ============================================================================

@app.get("/api/market/regime")
async def get_market_regime(user: User = Depends(require_valid_subscription)):
    """
    Get current market regime and trading recommendation.

    Uses multi-factor analysis: SPY trend, VIX, breadth, momentum.
    Returns one of 6 regimes: strong_bull, weak_bull, range_bound, weak_bear, panic_crash, recovery.
    """
    from app.services.market_regime import market_regime_service

    try:
        # Ensure data is loaded
        if not scanner_service.data_cache:
            await scanner_service.ensure_data_loaded()

        spy_df = scanner_service.data_cache.get('SPY')
        if spy_df is None or len(spy_df) < 200:
            raise HTTPException(status_code=503, detail="Insufficient SPY data")

        vix_df = scanner_service.data_cache.get('^VIX')

        # Use the 6-regime multi-factor detection
        regime = market_regime_service.detect_regime(
            spy_df=spy_df,
            universe_dfs=scanner_service.data_cache,
            vix_df=vix_df
        )

        regime_dict = regime.to_dict()

        # Map to format expected by frontend (backward compatibility)
        spy_price = spy_df.iloc[-1]['close'] if len(spy_df) > 0 else 0
        vix_level = vix_df.iloc[-1]['close'] if vix_df is not None and len(vix_df) > 0 else 20

        conditions = regime_dict.get('conditions', {})
        return {
            "regime": regime_dict.get('regime_type', regime_dict.get('name', 'neutral')),
            "regime_name": regime_dict.get('regime_name', regime_dict.get('name', 'Neutral').replace('_', ' ').title()),
            "spy_price": round(spy_price, 2),
            "spy_ma_200": round(conditions.get('spy_ma_200', 0), 2),
            "spy_ma_50": round(conditions.get('spy_ma_50', 0), 2),
            "spy_vs_200ma_pct": round(conditions.get('spy_vs_200ma_pct', 0), 2),
            "spy_pct_from_high": round(conditions.get('spy_pct_from_high', 0), 2),
            "vix_level": round(vix_level, 2),
            "vix_percentile": round(conditions.get('vix_percentile', 50), 1),
            "trend_strength": round(conditions.get('trend_strength', 0), 2),
            "long_term_trend": round(conditions.get('long_term_trend', 0), 2),
            "breadth_pct": round(conditions.get('breadth_pct', 50), 1),
            "new_highs_pct": round(conditions.get('new_highs_pct', 0), 1),
            "recommendation": regime_dict.get('description', ''),
            "risk_level": regime_dict.get('risk_level', 'medium'),
            "confidence": regime_dict.get('confidence', 0),
            "color": regime_dict.get('color', '#6B7280'),
            "updated": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/market/sectors")
async def get_sector_strength(admin: User = Depends(get_admin_user)):
    """
    Get sector strength rankings

    Returns sectors sorted by relative strength (0-100).
    """
    try:
        await market_analysis_service.update_sector_strength()
        return {
            "sectors": market_analysis_service.sector_strength,
            "strong_sectors": market_analysis_service.get_strong_sectors(),
            "weak_sectors": market_analysis_service.get_weak_sectors(),
            "updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/market/summary")
async def get_market_summary(admin: User = Depends(get_admin_user)):
    """
    Get complete market summary including regime and sectors
    """
    try:
        state = await market_analysis_service.update_market_state()
        await market_analysis_service.update_sector_strength()

        return {
            "regime": state.to_dict(),
            "sectors": {
                "rankings": market_analysis_service.sector_strength,
                "strong": market_analysis_service.get_strong_sectors(),
                "weak": market_analysis_service.get_weak_sectors()
            },
            "trading_guidance": {
                "regime": state.regime.value,
                "recommendation": state.recommendation,
                "vix_level": state.vix_level,
                "trend_strength": state.trend_strength
            }
        }
    except Exception as e:
        logger.error(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Scheduler Endpoints
# ============================================================================

@app.get("/api/scheduler/status")
async def get_scheduler_status(admin: User = Depends(get_admin_user)):
    """Get scheduler status and next run times"""
    return scheduler_service.get_status()


@app.post("/api/scheduler/run")
async def trigger_manual_update(admin: User = Depends(get_admin_user)):
    """
    Manually trigger a market update (for testing)

    This runs the same job that runs daily at 4:30 PM ET
    """
    try:
        await scheduler_service.run_now()
        return {
            "message": "Manual update completed successfully",
            "status": scheduler_service.get_status()
        }
    except Exception as e:
        logger.error(f"Scheduler update failed: {e}")
        raise HTTPException(status_code=500, detail="Update failed")


# ============================================================================
# Backtest Endpoints
# ============================================================================

@app.get("/api/backtest/run")
async def run_backtest(days: int = 252, strategy: str = "momentum", max_symbols: int = 200, user: User = Depends(require_valid_subscription)):
    """
    Run backtest over historical data

    Returns simulated positions and trades based on the selected strategy.
    Default is momentum strategy (v2). Use strategy="dwap" for legacy.

    Args:
        days: Number of trading days to simulate (default 252 = 1 year)
        strategy: "momentum" (default) or "dwap" for legacy
        max_symbols: Limit symbols for faster response (default 200)

    """
    if not scanner_service.data_cache:
        raise HTTPException(
            status_code=400,
            detail="No market data loaded. Please wait for data to load or trigger a scan."
        )

    use_momentum = strategy.lower() != "dwap"

    # Use top liquid symbols for faster response
    from app.services.strategy_analyzer import get_top_liquid_symbols
    top_symbols = get_top_liquid_symbols(max_symbols=max_symbols)

    try:
        result = backtester_service.run_backtest(
            lookback_days=days,
            use_momentum_strategy=use_momentum,
            ticker_list=top_symbols
        )
        return {
            "success": True,
            "strategy": "momentum" if use_momentum else "dwap",
            "backtest": {
                "start_date": result.start_date,
                "end_date": result.end_date,
                "total_return_pct": result.total_return_pct,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "open_positions": result.open_positions,
                "total_pnl": result.total_pnl,
                "max_drawdown_pct": result.max_drawdown_pct,
                "sharpe_ratio": result.sharpe_ratio
            },
            "positions": [p.to_dict() for p in result.positions],
            "trades": [t.to_dict() for t in result.trades]
        }
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail="Backtest failed")


@app.get("/api/backtest/positions")
async def get_backtest_positions(days: int = 252, admin: User = Depends(get_admin_user)):
    """
    Get simulated open positions from backtest

    These are positions we would currently hold if following the strategy.
    """
    if not scanner_service.data_cache:
        raise HTTPException(status_code=400, detail="No market data loaded")

    try:
        result = backtester_service.run_backtest(lookback_days=days)
        return {
            "positions": [p.to_dict() for p in result.positions],
            "total_value": sum(p.shares * p.current_price for p in result.positions),
            "total_pnl_pct": sum(p.pnl_pct * p.shares * p.entry_price for p in result.positions) /
                            sum(p.shares * p.entry_price for p in result.positions)
                            if result.positions else 0
        }
    except Exception as e:
        logger.error(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/backtest/trades")
async def get_backtest_trades(days: int = 252, limit: int = 50, admin: User = Depends(get_admin_user)):
    """
    Get trade history from backtest
    """
    if not scanner_service.data_cache:
        raise HTTPException(status_code=400, detail="No market data loaded")

    try:
        result = backtester_service.run_backtest(lookback_days=days)
        return {
            "trades": [t.to_dict() for t in result.trades[:limit]],
            "total": len(result.trades),
            "win_rate": result.win_rate,
            "total_pnl": result.total_pnl
        }
    except Exception as e:
        logger.error(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/backtest/walk-forward-cached")
async def get_cached_walk_forward(user: User = Depends(require_valid_subscription), db: AsyncSession = Depends(get_db)):
    """
    Get the latest cached daily walk-forward simulation results.

    This runs automatically once per day and provides more accurate simulated
    portfolio stats than a simple backtest (accounts for strategy switches).
    """
    from app.core.database import WalkForwardSimulation
    import json

    # Get the latest cached walk-forward result
    result = await db.execute(
        select(WalkForwardSimulation)
        .where(WalkForwardSimulation.is_daily_cache == True)
        .where(WalkForwardSimulation.status == "completed")
        .order_by(desc(WalkForwardSimulation.simulation_date))
        .limit(1)
    )
    cached = result.scalar_one_or_none()

    if not cached:
        # No cached result, return None so frontend can fall back to simple backtest
        return {
            "available": False,
            "message": "No cached walk-forward results available yet"
        }

    # Parse equity curve and switch history from JSON
    equity_curve = []
    switch_history = []
    try:
        if cached.equity_curve_json:
            equity_curve = json.loads(cached.equity_curve_json)
        if cached.switch_history_json:
            switch_history = json.loads(cached.switch_history_json)
    except json.JSONDecodeError:
        pass

    return {
        "available": True,
        "simulation_date": cached.simulation_date.isoformat(),
        "start_date": cached.start_date.isoformat(),
        "end_date": cached.end_date.isoformat(),
        "total_return_pct": cached.total_return_pct,
        "sharpe_ratio": cached.sharpe_ratio,
        "max_drawdown_pct": cached.max_drawdown_pct,
        "benchmark_return_pct": cached.benchmark_return_pct,
        "num_strategy_switches": cached.num_strategy_switches,
        "switch_history": switch_history,
        "equity_curve": equity_curve,
        "reoptimization_frequency": cached.reoptimization_frequency,
    }


# ============================================================================
# Portfolio Endpoints (with Database)
# ============================================================================

@app.get("/api/portfolio/positions", response_model=PositionsListResponse)
async def get_positions(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get all open positions with current prices (split-adjusted)"""

    result = await db.execute(
        select(DBPosition).where(DBPosition.status == "open", DBPosition.user_id == user.id).order_by(desc(DBPosition.created_at))
    )
    db_positions = result.scalars().all()

    positions = []
    total_value = 0.0
    total_cost = 0.0

    # Trailing stop configuration (ensemble strategy uses 12%)
    TRAILING_STOP_PCT = 12.0

    for pos in db_positions:
        # Get split-adjusted entry price from historical data
        # This handles stock splits automatically - yfinance adjusts all historical prices
        adjusted_entry = get_split_adjusted_price(pos.symbol, pos.entry_date, pos.entry_price)

        # Get current price from cache if available
        current_price = adjusted_entry  # Default to entry if no live data
        if pos.symbol in scanner_service.data_cache:
            df = scanner_service.data_cache[pos.symbol]
            if len(df) > 0:
                current_price = float(df.iloc[-1]['close'])

        # Calculate high water mark from historical data since entry
        high_water_mark = adjusted_entry
        if pos.symbol in scanner_service.data_cache:
            df = scanner_service.data_cache[pos.symbol]
            # Filter to dates on or after entry (normalize to midnight so
            # the entry day itself is included regardless of time-of-day)
            entry_ts = pd.Timestamp(pos.entry_date).normalize()
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                entry_ts = entry_ts.tz_localize(df.index.tz)
            mask = df.index >= entry_ts
            if mask.any():
                high_water_mark = max(adjusted_entry, float(df.loc[mask, 'close'].max()))

        # Update database with high water mark if it's higher
        if pos.highest_price is None or high_water_mark > pos.highest_price:
            pos.highest_price = high_water_mark

        # Calculate trailing stop from high water mark
        trailing_stop_price = round(high_water_mark * (1 - TRAILING_STOP_PCT / 100), 2)

        # Calculate distance to trailing stop (positive = above stop, negative = below)
        distance_to_stop_pct = ((current_price - trailing_stop_price) / trailing_stop_price) * 100

        # Determine sell signal
        if current_price <= trailing_stop_price:
            sell_signal = "sell"  # Already hit trailing stop
        elif distance_to_stop_pct <= 3.0:
            sell_signal = "warning"  # Within 3% of trailing stop
        else:
            sell_signal = "hold"

        # Calculate legacy stop/target for backwards compatibility
        stop_loss = round(adjusted_entry * (1 - settings.STOP_LOSS_PCT / 100), 2)
        profit_target = round(adjusted_entry * (1 + settings.PROFIT_TARGET_PCT / 100), 2)

        from app.core.timezone import days_since_et
        days_held = days_since_et(pos.entry_date)
        pnl_pct = ((current_price - adjusted_entry) / adjusted_entry) * 100
        position_value = pos.shares * current_price

        total_value += position_value
        total_cost += pos.shares * adjusted_entry

        positions.append(PositionResponse(
            id=pos.id,
            symbol=pos.symbol,
            shares=pos.shares,
            entry_price=round(adjusted_entry, 2),
            entry_date=pos.entry_date.strftime('%Y-%m-%d'),
            current_price=round(current_price, 2),
            stop_loss=stop_loss,
            profit_target=profit_target,
            pnl_pct=round(pnl_pct, 2),
            days_held=days_held,
            high_water_mark=round(high_water_mark, 2),
            trailing_stop_price=trailing_stop_price,
            trailing_stop_pct=TRAILING_STOP_PCT,
            distance_to_stop_pct=round(distance_to_stop_pct, 1),
            sell_signal=sell_signal
        ))

    # Commit any high water mark updates
    await db.commit()

    total_pnl_pct = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0

    return PositionsListResponse(
        positions=positions,
        total_value=round(total_value, 2),
        total_pnl_pct=round(total_pnl_pct, 2)
    )


@app.post("/api/portfolio/positions")
async def open_position(request: OpenPositionRequest, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Open a new position"""
    symbol = request.symbol.upper()

    # Get current price from cache or use provided price
    price = request.price
    if not price and symbol in scanner_service.data_cache:
        df = scanner_service.data_cache[symbol]
        if len(df) > 0:
            price = float(df.iloc[-1]['close'])

    if not price:
        raise HTTPException(status_code=400, detail=f"Could not get price for {symbol}. Provide price or run scan first.")

    shares = request.shares or (10000 / price)  # Default ~$10k position

    # Use provided entry_date (time-travel mode) or default to now
    if request.entry_date:
        entry_date = datetime.strptime(request.entry_date, '%Y-%m-%d')
    else:
        entry_date = datetime.now()

    position = DBPosition(
        user_id=user.id,
        symbol=symbol,
        entry_date=entry_date,
        entry_price=price,
        shares=round(shares, 2),
        stop_loss=round(price * (1 - settings.STOP_LOSS_PCT / 100), 2),
        profit_target=round(price * (1 + settings.PROFIT_TARGET_PCT / 100), 2),
        highest_price=price,
        status="open"
    )

    db.add(position)
    await db.commit()
    await db.refresh(position)

    return {
        "message": f"Opened position in {symbol}",
        "position": {
            "id": position.id,
            "symbol": position.symbol,
            "shares": position.shares,
            "entry_price": position.entry_price,
            "stop_loss": position.stop_loss,
            "profit_target": position.profit_target
        }
    }


@app.delete("/api/portfolio/positions/{position_id}")
async def close_position(position_id: int, exit_price: Optional[float] = None, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Close a position and record the trade (with split-adjusted prices)"""

    result = await db.execute(select(DBPosition).where(DBPosition.id == position_id, DBPosition.user_id == user.id))
    position = result.scalar_one_or_none()

    if not position:
        raise HTTPException(status_code=404, detail="Position not found")

    # Get split-adjusted entry price
    adjusted_entry = get_split_adjusted_price(position.symbol, position.entry_date, position.entry_price)

    # Get exit price
    price = exit_price
    if not price and position.symbol in scanner_service.data_cache:
        df = scanner_service.data_cache[position.symbol]
        if len(df) > 0:
            price = float(df.iloc[-1]['close'])

    if not price:
        price = adjusted_entry  # Fallback

    # Calculate P&L using split-adjusted entry
    pnl = (price - adjusted_entry) * position.shares
    pnl_pct = ((price - adjusted_entry) / adjusted_entry) * 100

    # Calculate split-adjusted stop/target for exit reason
    stop_loss = adjusted_entry * (1 - settings.STOP_LOSS_PCT / 100)
    profit_target = adjusted_entry * (1 + settings.PROFIT_TARGET_PCT / 100)

    # Determine exit reason
    exit_reason = "manual"
    if price <= stop_loss:
        exit_reason = "stop_loss"
    elif price >= profit_target:
        exit_reason = "profit_target"

    # Record trade with split-adjusted entry price
    trade = DBTrade(
        user_id=user.id,
        position_id=position.id,
        symbol=position.symbol,
        entry_date=position.entry_date,
        entry_price=round(adjusted_entry, 2),
        exit_date=datetime.now(),
        exit_price=price,
        shares=position.shares,
        pnl=round(pnl, 2),
        pnl_pct=round(pnl_pct, 2),
        exit_reason=exit_reason
    )
    db.add(trade)

    # Mark position as closed
    position.status = "closed"

    await db.commit()

    return {
        "message": f"Closed position in {position.symbol}",
        "trade": {
            "symbol": trade.symbol,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "pnl": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "exit_reason": trade.exit_reason
        }
    }


@app.get("/api/portfolio/trades")
async def get_trades(limit: int = 50, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get trade history"""

    result = await db.execute(
        select(DBTrade).where(DBTrade.user_id == user.id).order_by(desc(DBTrade.exit_date)).limit(limit)
    )
    trades = result.scalars().all()

    return {
        "trades": [
            {
                "id": t.id,
                "symbol": t.symbol,
                "entry_date": t.entry_date.strftime('%Y-%m-%d'),
                "entry_price": t.entry_price,
                "exit_date": t.exit_date.strftime('%Y-%m-%d'),
                "exit_price": t.exit_price,
                "shares": t.shares,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason
            }
            for t in trades
        ],
        "total": len(trades)
    }


@app.get("/api/portfolio/equity")
async def get_equity_curve(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Get equity curve based on trade history

    For now, returns cumulative P&L from trades.
    TODO: Implement proper daily equity tracking.
    """
    result = await db.execute(
        select(DBTrade).where(DBTrade.user_id == user.id).order_by(DBTrade.exit_date)
    )
    trades = result.scalars().all()

    if not trades:
        # Return empty curve if no trades yet
        return {"equity_curve": []}

    # Build cumulative equity curve
    initial_capital = 100000
    equity = initial_capital
    curve = []

    for trade in trades:
        equity += trade.pnl
        curve.append(EquityPoint(
            date=trade.exit_date.strftime('%Y-%m-%d'),
            equity=round(equity, 2)
        ))

    return {"equity_curve": curve}


@app.get("/api/stock/{symbol}/history")
async def get_stock_history(symbol: str, days: int = 252, user: User = Depends(require_valid_subscription)):
    """Get historical price data for a symbol from cache"""
    symbol = symbol.upper()

    # If cache is empty, try to load from S3 first (Lambda mode)
    if not scanner_service.data_cache:
        import os
        is_lambda = os.environ.get("ENVIRONMENT") == "prod"
        if is_lambda:
            try:
                cached_data = data_export_service.import_all()
                if cached_data:
                    scanner_service.data_cache = cached_data
                    print(f"📦 Loaded {len(cached_data)} symbols from S3 for history request")
            except Exception as e:
                print(f"⚠️ Failed to load from S3: {e}")

    if symbol not in scanner_service.data_cache:
        # Try to fetch it from yfinance
        try:
            await scanner_service.fetch_data([symbol], period="5y")
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Could not fetch data for {symbol}")

    if symbol not in scanner_service.data_cache:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")

    df = scanner_service.data_cache[symbol].copy()

    # Calculate indicators if they don't exist (e.g., for newly added symbols)
    if 'dwap' not in df.columns:
        # Calculate DWAP (Daily Weighted Average Price) - 200-day volume-weighted average
        df['dwap'] = (df['close'] * df['volume']).rolling(200).sum() / df['volume'].rolling(200).sum()
    if 'ma_50' not in df.columns:
        df['ma_50'] = df['close'].rolling(50).mean()
    if 'ma_200' not in df.columns:
        df['ma_200'] = df['close'].rolling(200).mean()

    df = df.tail(days)

    return {
        "symbol": symbol,
        "data": [
            {
                "date": idx.strftime('%Y-%m-%d'),
                "open": round(row['open'], 2),
                "high": round(row['high'], 2),
                "low": round(row['low'], 2),
                "close": round(row['close'], 2),
                "volume": int(row['volume']),
                "dwap": round(row['dwap'], 2) if pd.notna(row.get('dwap')) else None,
                "ma_50": round(row['ma_50'], 2) if pd.notna(row.get('ma_50')) else None,
                "ma_200": round(row['ma_200'], 2) if pd.notna(row.get('ma_200')) else None,
            }
            for idx, row in df.iterrows()
        ]
    }


# ============================================================================
# Live Quotes Endpoint (for real-time UI updates)
# ============================================================================

@app.get("/api/quotes/live")
async def get_live_quotes(symbols: str = "", user: User = Depends(get_current_user)):
    """
    Get live/current quotes for symbols.

    Uses yfinance to fetch current prices for display in UI.
    Note: Signals are still based on daily CLOSE prices.

    Args:
        symbols: Comma-separated list of symbols, or empty for all positions

    Returns:
        Dict of symbol -> quote data
    """
    import yfinance as yf

    # Parse symbols or get from open positions
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    else:
        # Get symbols from user's open positions
        try:
            async with async_session() as db:
                result = await db.execute(
                    select(DBPosition.symbol).where(DBPosition.status == 'open', DBPosition.user_id == user.id).distinct()
                )
                symbol_list = [row[0] for row in result.fetchall()]
        except:
            symbol_list = []

    if not symbol_list:
        return {"quotes": {}, "timestamp": datetime.now().isoformat()}

    # Fetch current quotes from yfinance
    quotes = {}
    try:
        # Use yfinance download with period="1d" for current day data
        # This gives us the most recent price
        tickers = yf.Tickers(" ".join(symbol_list))

        for symbol in symbol_list:
            try:
                ticker = tickers.tickers.get(symbol)
                if ticker:
                    info = ticker.fast_info
                    last_price = info.last_price if hasattr(info, 'last_price') else None
                    prev_close = info.previous_close if hasattr(info, 'previous_close') else None

                    if last_price:
                        change = last_price - prev_close if prev_close else 0
                        change_pct = (change / prev_close * 100) if prev_close else 0

                        quotes[symbol] = {
                            "price": round(last_price, 2),
                            "change": round(change, 2),
                            "change_pct": round(change_pct, 2),
                            "prev_close": round(prev_close, 2) if prev_close else None,
                        }
            except Exception as e:
                logger.warning(f"Failed to get quote for {symbol}: {e}")
                continue

    except Exception as e:
        logger.error(f"Failed to fetch live quotes: {e}")
        logger.error(f"Failed to fetch quotes: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch quotes")

    return {
        "quotes": quotes,
        "timestamp": datetime.now().isoformat(),
        "count": len(quotes)
    }


@app.post("/api/quotes/batch")
async def get_batch_quotes(symbols: List[str], admin: User = Depends(get_admin_user)):
    """
    Get live quotes for a batch of symbols (POST for larger lists).
    """
    import yfinance as yf

    if not symbols:
        return {"quotes": {}, "timestamp": datetime.now().isoformat()}

    symbol_list = [s.upper() for s in symbols[:100]]  # Limit to 100 symbols

    quotes = {}
    try:
        tickers = yf.Tickers(" ".join(symbol_list))

        for symbol in symbol_list:
            try:
                ticker = tickers.tickers.get(symbol)
                if ticker:
                    info = ticker.fast_info
                    last_price = info.last_price if hasattr(info, 'last_price') else None
                    prev_close = info.previous_close if hasattr(info, 'previous_close') else None

                    if last_price:
                        change = last_price - prev_close if prev_close else 0
                        change_pct = (change / prev_close * 100) if prev_close else 0

                        quotes[symbol] = {
                            "price": round(last_price, 2),
                            "change": round(change, 2),
                            "change_pct": round(change_pct, 2),
                            "prev_close": round(prev_close, 2) if prev_close else None,
                        }
            except Exception as e:
                continue

    except Exception as e:
        logger.error(f"Failed to fetch batch quotes: {e}")
        logger.error(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return {
        "quotes": quotes,
        "timestamp": datetime.now().isoformat(),
        "count": len(quotes)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

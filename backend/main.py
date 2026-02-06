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
from app.core.database import init_db, get_db, Position as DBPosition, Trade as DBTrade, Signal as DBSignal, async_session
from app.api.signals import router as signals_router
from app.api.email import router as email_router
from app.api.auth import router as auth_router
from app.api.billing import router as billing_router
from app.api.admin import router as admin_router
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
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(signals_router, prefix="/api/signals", tags=["signals"])
app.include_router(email_router, prefix="/api/email", tags=["email"])
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(billing_router, prefix="/api/billing", tags=["billing"])
app.include_router(admin_router, prefix="/api/admin", tags=["admin"])

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
            else:
                print("⚠️ No cached data found in S3")
        except Exception as e:
            print(f"⚠️ Failed to load data from S3: {e}")

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
                existing_job_id=job_id
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
async def root():
    return {
        "name": "RigaCap API",
        "version": "2.0.0",
        "docs": "/docs",
        "data_loaded": len(scanner_service.data_cache),
        "last_scan": scanner_service.last_scan.isoformat() if scanner_service.last_scan else None
    }


@app.get("/health")
async def health():
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
async def warmup():
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
async def load_batch(batch_size: int = 50):
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
async def data_status():
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
async def get_config():
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
async def debug_yfinance():
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
async def load_full_universe():
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/universe/status")
async def get_universe_status():
    """Get current universe status"""
    return {
        "universe_size": len(scanner_service.universe),
        "full_universe_loaded": scanner_service.full_universe_loaded,
        "symbols_with_data": len(scanner_service.data_cache),
        "sample_symbols": scanner_service.universe[:20] if scanner_service.universe else []
    }


@app.post("/api/data/load")
async def load_market_data(symbols: Optional[str] = None, period: str = "5y"):
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
async def get_data_status():
    """Get current data loading status"""
    return {
        "data_available": len(scanner_service.data_cache) > 0,
        "symbols_loaded": len(scanner_service.data_cache),
        "universe_size": len(scanner_service.universe),
        "last_scan": scanner_service.last_scan.isoformat() if scanner_service.last_scan else None,
        "loaded_symbols": list(scanner_service.data_cache.keys())[:50] if scanner_service.data_cache else []
    }


@app.post("/api/data/export")
async def export_data():
    """
    Export all cached price data to individual CSV files

    This saves historical data permanently so it never needs to be re-fetched.
    """
    if not scanner_service.data_cache:
        raise HTTPException(status_code=400, detail="No data to export. Load data first.")

    result = data_export_service.export_all(scanner_service.data_cache)
    return result


@app.post("/api/data/export-consolidated")
async def export_data_consolidated():
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
async def pre_deploy_export():
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
async def get_export_status():
    """Get status of exported data files"""
    return data_export_service.get_status()


@app.post("/api/data/import")
async def import_data():
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
async def get_market_regime():
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/sectors")
async def get_sector_strength():
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/summary")
async def get_market_summary():
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
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Scheduler Endpoints
# ============================================================================

@app.get("/api/scheduler/status")
async def get_scheduler_status():
    """Get scheduler status and next run times"""
    return scheduler_service.get_status()


@app.post("/api/scheduler/run")
async def trigger_manual_update():
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
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


# ============================================================================
# Backtest Endpoints
# ============================================================================

@app.get("/api/backtest/run")
async def run_backtest(days: int = 252, strategy: str = "momentum", max_symbols: int = 200):
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
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@app.get("/api/backtest/positions")
async def get_backtest_positions(days: int = 252):
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/trades")
async def get_backtest_trades(days: int = 252, limit: int = 50):
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/walk-forward-cached")
async def get_cached_walk_forward(db: AsyncSession = Depends(get_db)):
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
async def get_positions(db: AsyncSession = Depends(get_db)):
    """Get all open positions with current prices (split-adjusted)"""

    result = await db.execute(
        select(DBPosition).where(DBPosition.status == "open").order_by(desc(DBPosition.created_at))
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
            # Filter to dates after entry
            entry_ts = pd.Timestamp(pos.entry_date)
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

        days_held = (datetime.now() - pos.entry_date).days
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
async def open_position(request: OpenPositionRequest, db: AsyncSession = Depends(get_db)):
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

    position = DBPosition(
        symbol=symbol,
        entry_date=datetime.now(),
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
async def close_position(position_id: int, exit_price: Optional[float] = None, db: AsyncSession = Depends(get_db)):
    """Close a position and record the trade (with split-adjusted prices)"""

    result = await db.execute(select(DBPosition).where(DBPosition.id == position_id))
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
async def get_trades(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Get trade history"""

    result = await db.execute(
        select(DBTrade).order_by(desc(DBTrade.exit_date)).limit(limit)
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
async def get_equity_curve(db: AsyncSession = Depends(get_db)):
    """
    Get equity curve based on trade history

    For now, returns cumulative P&L from trades.
    TODO: Implement proper daily equity tracking.
    """
    result = await db.execute(
        select(DBTrade).order_by(DBTrade.exit_date)
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
async def get_stock_history(symbol: str, days: int = 252):
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
            raise HTTPException(status_code=404, detail=f"Could not fetch data for {symbol}: {e}")

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
async def get_live_quotes(symbols: str = ""):
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
                    select(DBPosition.symbol).where(DBPosition.status == 'open').distinct()
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
        raise HTTPException(status_code=500, detail=f"Failed to fetch quotes: {str(e)}")

    return {
        "quotes": quotes,
        "timestamp": datetime.now().isoformat(),
        "count": len(quotes)
    }


@app.post("/api/quotes/batch")
async def get_batch_quotes(symbols: List[str]):
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
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "quotes": quotes,
        "timestamp": datetime.now().isoformat(),
        "count": len(quotes)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Signals API - Trading signal endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from app.core.database import get_db, Signal
from app.services.scanner import scanner_service, SignalData
from app.services.stock_universe import stock_universe_service
from app.services.data_export import data_export_service

router = APIRouter()


# Pydantic models for API
class SignalResponse(BaseModel):
    id: Optional[int] = None
    symbol: str
    signal_type: str
    price: float
    dwap: float
    pct_above_dwap: float
    volume: int
    volume_ratio: float
    stop_loss: float
    profit_target: float
    ma_50: float = 0.0
    ma_200: float = 0.0
    high_52w: float = 0.0
    is_strong: bool
    signal_strength: float = 0.0
    sector: str = ""
    recommendation: str = ""
    timestamp: str

    class Config:
        from_attributes = True


class ScanResponse(BaseModel):
    timestamp: str
    total_signals: int
    strong_signals: int
    signals: List[SignalResponse]


class WatchlistItem(BaseModel):
    symbol: str
    price: float
    dwap: float
    pct_above_dwap: float


# Endpoints
@router.get("/scan", response_model=ScanResponse)
async def run_scan(
    refresh: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """
    Run market scan for buy signals
    
    - **refresh**: If true, fetch fresh data from Yahoo Finance
    """
    try:
        signals = await scanner_service.scan(refresh_data=refresh)
        
        # Store signals in database
        for sig in signals:
            db_signal = Signal(
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
        
        return ScanResponse(
            timestamp=datetime.now().isoformat(),
            total_signals=len(signals),
            strong_signals=len([s for s in signals if s.is_strong]),
            signals=[SignalResponse(**s.to_dict()) for s in signals]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest", response_model=ScanResponse)
async def get_latest_signals(
    limit: int = 20,
    strong_only: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    Get latest signals from database
    
    - **limit**: Maximum number of signals to return
    - **strong_only**: If true, only return strong signals
    """
    query = select(Signal).order_by(desc(Signal.created_at)).limit(limit)
    
    if strong_only:
        query = query.where(Signal.is_strong == True)
    
    result = await db.execute(query)
    signals = result.scalars().all()
    
    return ScanResponse(
        timestamp=datetime.now().isoformat(),
        total_signals=len(signals),
        strong_signals=len([s for s in signals if s.is_strong]),
        signals=[SignalResponse(
            id=s.id,
            symbol=s.symbol,
            signal_type=s.signal_type,
            price=s.price,
            dwap=s.dwap,
            pct_above_dwap=s.pct_above_dwap,
            volume=int(s.volume),
            volume_ratio=s.volume_ratio,
            stop_loss=s.stop_loss,
            profit_target=s.profit_target,
            is_strong=s.is_strong,
            timestamp=s.created_at.isoformat()
        ) for s in signals]
    )


@router.get("/watchlist", response_model=List[WatchlistItem])
async def get_watchlist(threshold: float = 3.0):
    """
    Get stocks approaching DWAP threshold (watchlist)
    
    - **threshold**: Minimum % above DWAP to include
    """
    watchlist = scanner_service.get_watchlist(threshold)
    return [WatchlistItem(**item) for item in watchlist]


@router.get("/memory-scan", response_model=ScanResponse)
async def run_memory_scan(
    refresh: bool = False,
    apply_market_filter: bool = True,
    min_strength: float = 0,
    export_to_cdn: bool = True
):
    """
    Run market scan without database (memory only)

    - **refresh**: If true, fetch fresh data from Yahoo Finance
    - **apply_market_filter**: Apply bull/bear market filtering
    - **min_strength**: Minimum signal strength (0-100)
    - **export_to_cdn**: Export signals to S3 for CDN delivery (default true)
    """
    try:
        signals = await scanner_service.scan(
            refresh_data=refresh,
            apply_market_filter=apply_market_filter,
            min_signal_strength=min_strength
        )

        # Export signals to S3 for CDN delivery
        if export_to_cdn and signals:
            data_export_service.export_signals_json(signals)

        return ScanResponse(
            timestamp=datetime.now().isoformat(),
            total_signals=len(signals),
            strong_signals=len([s for s in signals if s.is_strong]),
            signals=[SignalResponse(**s.to_dict()) for s in signals]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cdn-url")
async def get_signals_cdn_url():
    """
    Get the CDN URL for the latest signals JSON.

    Frontend should fetch signals from this URL for instant loading.
    """
    return {
        "url": data_export_service.get_signals_url(),
        "cache_max_age": 300,  # 5 minutes
        "description": "Fetch signals from this URL for instant loading"
    }


@router.get("/symbol/{symbol}", response_model=SignalResponse)
async def get_signal_for_symbol(symbol: str):
    """
    Get current signal analysis for a specific symbol
    """
    if symbol.upper() not in scanner_service.data_cache:
        # Try to fetch data
        await scanner_service.fetch_data([symbol.upper()])

    signal = scanner_service.analyze_stock(symbol.upper())

    if not signal:
        raise HTTPException(status_code=404, detail=f"No signal for {symbol}")

    return SignalResponse(**signal.to_dict())


class MissedOpportunity(BaseModel):
    symbol: str
    signal_date: str
    signal_price: float
    exit_date: str  # When it hit +20% target
    exit_price: float
    would_be_return: float
    would_be_pnl: float
    days_held: int = 0
    signal_strength: float = 0.0
    sector: str = ""


@router.get("/missed", response_model=List[MissedOpportunity])
async def get_missed_opportunities(
    days: int = 90,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Get missed opportunities - signals from the past N days that HIT the +20% profit target.
    These are completed trades the user could have made.

    - **days**: Look back period (default 90 days)
    - **limit**: Maximum results to return (default 10)
    """
    from datetime import timedelta
    from sqlalchemy import and_
    import pandas as pd

    # Look back further to find completed opportunities
    cutoff_date = datetime.now() - timedelta(days=days)
    # Don't include signals from the last 1 day (too recent to have hit target)
    recent_cutoff = datetime.now() - timedelta(days=1)

    # Get signals from the lookback period (but not too recent)
    query = select(Signal).where(
        and_(
            Signal.created_at >= cutoff_date,
            Signal.created_at <= recent_cutoff,
            Signal.is_strong == True
        )
    ).order_by(desc(Signal.created_at))

    result = await db.execute(query)
    signals = result.scalars().all()

    # Get current positions to exclude
    from app.core.database import Position
    positions_query = select(Position.symbol).where(Position.status == 'open')
    positions_result = await db.execute(positions_query)
    owned_symbols = set(row[0] for row in positions_result.fetchall())

    # Find signals that hit +20% profit target
    opportunities = []
    seen_symbols = set()  # Track symbols to avoid duplicates
    PROFIT_TARGET = 0.20  # 20%

    for sig in signals:
        # Skip if user owns this stock or we've already processed it
        if sig.symbol in owned_symbols or sig.symbol in seen_symbols:
            continue

        # Check if we have price data for this symbol
        if sig.symbol not in scanner_service.data_cache:
            continue

        df = scanner_service.data_cache[sig.symbol]
        if len(df) == 0:
            continue

        # Get the signal date and find prices after that date
        signal_date = sig.created_at.date()
        target_price = sig.price * (1 + PROFIT_TARGET)

        # Filter to dates after the signal
        try:
            future_prices = df[df.index.date > signal_date]
        except:
            continue

        if len(future_prices) == 0:
            continue

        # Check if the stock ever hit the +20% target
        high_prices = future_prices['high'] if 'high' in future_prices.columns else future_prices['close']
        hit_target = high_prices >= target_price

        if hit_target.any():
            # Find when it first hit the target
            first_hit_idx = hit_target.idxmax()
            exit_date = first_hit_idx.date() if hasattr(first_hit_idx, 'date') else first_hit_idx
            days_held = (first_hit_idx - pd.Timestamp(signal_date)).days

            seen_symbols.add(sig.symbol)

            # Get sector info
            sector = ""
            info = stock_universe_service.symbol_info.get(sig.symbol, {})
            sector = info.get("sector", "")

            # Calculate P&L (assume $6,000 position size = 6% of $100k)
            position_value = 6000
            shares = position_value / sig.price
            pnl = shares * sig.price * PROFIT_TARGET  # Always 20% gain

            opportunities.append(MissedOpportunity(
                symbol=sig.symbol,
                signal_date=sig.created_at.strftime("%Y-%m-%d"),
                signal_price=round(sig.price, 2),
                exit_date=str(exit_date),
                exit_price=round(target_price, 2),
                would_be_return=round(PROFIT_TARGET * 100, 1),  # Always 20%
                would_be_pnl=round(pnl, 2),
                days_held=days_held,
                signal_strength=getattr(sig, 'signal_strength', 0) or 0,
                sector=sector
            ))

    # Sort by most recent signal date first
    opportunities.sort(key=lambda x: x.signal_date, reverse=True)
    return opportunities[:limit]


@router.post("/backfill")
async def backfill_historical_signals(
    days: int = 90,
    db: AsyncSession = Depends(get_db)
):
    """
    Backfill historical signals by scanning past price data.

    This simulates what signals would have been generated over the past N days.
    Used to populate the "missed opportunities" section.

    - **days**: How many days back to scan (default 90)
    """
    import pandas as pd
    import numpy as np
    from datetime import timedelta
    from app.core.config import settings

    if not scanner_service.data_cache:
        raise HTTPException(status_code=400, detail="No price data loaded")

    # Settings for signal detection
    DWAP_THRESHOLD = settings.DWAP_THRESHOLD_PCT / 100  # 5%
    MIN_VOLUME = settings.MIN_VOLUME
    MIN_PRICE = settings.MIN_PRICE
    VOLUME_SPIKE = settings.VOLUME_SPIKE_MULT
    STOP_LOSS_PCT = settings.STOP_LOSS_PCT
    PROFIT_TARGET_PCT = settings.PROFIT_TARGET_PCT

    # Date range to scan
    end_date = datetime.now() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=days)

    signals_created = 0
    days_scanned = 0

    # Get trading days from one of the symbols
    sample_symbol = list(scanner_service.data_cache.keys())[0]
    sample_df = scanner_service.data_cache[sample_symbol]

    # Filter to our date range - normalize dates for comparison
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()

    trading_days = [
        d for d in sample_df.index
        if start_ts <= pd.Timestamp(d).normalize() <= end_ts
    ]

    # Process each trading day
    for trade_date in trading_days:
        days_scanned += 1
        trade_date_normalized = pd.Timestamp(trade_date).normalize()

        # Check each symbol for signals on this date
        for symbol, df in scanner_service.data_cache.items():
            try:
                # Find the closest date in this symbol's data
                date_matches = [d for d in df.index if pd.Timestamp(d).normalize() == trade_date_normalized]
                if not date_matches:
                    continue

                actual_date = date_matches[0]
                idx = df.index.get_loc(actual_date)
                if idx < 200:  # Need enough history for DWAP
                    continue

                row = df.loc[actual_date]
                price = row['close']
                volume = row['volume']
                dwap = row.get('dwap', np.nan)
                vol_avg = row.get('vol_avg', np.nan)
                ma_50 = row.get('ma_50', np.nan)
                ma_200 = row.get('ma_200', np.nan)

                if pd.isna(dwap) or dwap <= 0:
                    continue

                pct_above_dwap = (price / dwap - 1)

                # Check buy conditions
                if (pct_above_dwap >= DWAP_THRESHOLD and
                    volume >= MIN_VOLUME and
                    price >= MIN_PRICE):

                    vol_ratio = volume / vol_avg if vol_avg and vol_avg > 0 else 0

                    # Strong signal check
                    is_strong = (
                        vol_ratio >= VOLUME_SPIKE and
                        not pd.isna(ma_50) and not pd.isna(ma_200) and
                        price > ma_50 > ma_200
                    )

                    # Check if signal already exists for this symbol on this date
                    check_start = trade_date_normalized.to_pydatetime()
                    check_end = check_start + timedelta(days=1)
                    existing = await db.execute(
                        select(Signal).where(
                            Signal.symbol == symbol,
                            Signal.created_at >= check_start,
                            Signal.created_at < check_end
                        )
                    )
                    if existing.scalar_one_or_none():
                        continue  # Already have this signal

                    # Create the signal
                    stop_loss = price * (1 - STOP_LOSS_PCT / 100)
                    profit_target = price * (1 + PROFIT_TARGET_PCT / 100)

                    signal = Signal(
                        symbol=symbol,
                        signal_type='BUY',
                        price=round(float(price), 2),
                        dwap=round(float(dwap), 2),
                        pct_above_dwap=round(float(pct_above_dwap) * 100, 1),
                        volume=int(volume),
                        volume_ratio=round(float(vol_ratio), 2),
                        stop_loss=round(float(stop_loss), 2),
                        profit_target=round(float(profit_target), 2),
                        is_strong=bool(is_strong),
                        status='historical'
                    )
                    # Set created_at to the historical date
                    signal.created_at = trade_date_normalized.to_pydatetime()

                    db.add(signal)
                    signals_created += 1

            except Exception as e:
                continue  # Skip symbols with issues

        # Commit in batches
        if days_scanned % 10 == 0:
            await db.commit()

    # Final commit
    await db.commit()

    return {
        "success": True,
        "days_scanned": days_scanned,
        "signals_created": signals_created,
        "date_range": {
            "start": start_date.strftime('%Y-%m-%d'),
            "end": end_date.strftime('%Y-%m-%d')
        }
    }


@router.get("/info/{symbol}")
async def get_stock_info(symbol: str):
    """
    Get company information for a stock symbol

    Returns name, sector, industry, description, market cap, etc.
    Fetches from yfinance if sector/description not already cached.
    """
    symbol = symbol.upper()

    # Load cache if needed
    if not stock_universe_service.symbol_info:
        stock_universe_service._load_from_cache()

    # Get basic info from cache
    info = stock_universe_service.symbol_info.get(symbol, {})

    # Fetch detailed info (sector, description) from yfinance if not cached
    if not info.get('sector') or not info.get('description'):
        try:
            info = await stock_universe_service.fetch_company_details(symbol)
        except Exception:
            pass  # Use whatever info we have

    # Get current price and technical data if available
    current_data = {}
    if symbol in scanner_service.data_cache:
        df = scanner_service.data_cache[symbol]
        if len(df) > 0:
            row = df.iloc[-1]
            current_data = {
                "current_price": round(float(row['close']), 2),
                "dwap": round(float(row['dwap']), 2) if row['dwap'] else None,
                "ma_50": round(float(row['ma_50']), 2) if row['ma_50'] else None,
                "ma_200": round(float(row['ma_200']), 2) if row['ma_200'] else None,
                "high_52w": round(float(row['high_52w']), 2) if row['high_52w'] else None,
                "volume": int(row['volume']),
                "avg_volume": int(row['vol_avg']) if row['vol_avg'] else None,
            }

    return {
        "symbol": symbol,
        "name": info.get("name", symbol),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "description": info.get("description", ""),
        "market_cap": info.get("market_cap", ""),
        "exchange": info.get("exchange", ""),
        "website": info.get("website", ""),
        "employees": info.get("employees"),
        **current_data
    }

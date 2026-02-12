"""
Signals API - Trading signal endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from app.core.database import get_db, Signal, Position, User
from app.core.security import get_current_user_optional
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
    entry_date: str
    entry_price: float
    sell_date: str  # When it hit +20% profit target
    sell_price: float
    would_be_return: float
    would_be_pnl: float
    days_held: int = 0
    sector: str = ""


@router.get("/missed", response_model=List[MissedOpportunity])
async def get_missed_opportunities(
    days: int = 90,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Get missed opportunities - profitable trades from a backtest of the past N days.
    Shows what returns users could have achieved using our momentum strategy.

    - **days**: Look back period (default 90 days)
    - **limit**: Maximum results to return (default 10)
    """
    from app.services.backtester import backtester_service

    # Run backtest for the specified period
    try:
        result = backtester_service.run_backtest(
            lookback_days=days,
            use_momentum_strategy=True
        )
    except Exception as e:
        print(f"[MISSED] Backtest failed: {e}")
        return []

    # Filter to only profitable closed trades (winners)
    opportunities = []
    for trade in result.trades:
        # Only show profitable trades
        if trade.pnl_pct <= 0:
            continue

        # Get sector info
        sector = ""
        info = stock_universe_service.symbol_info.get(trade.symbol, {})
        sector = info.get("sector", "")

        opportunities.append(MissedOpportunity(
            symbol=trade.symbol,
            entry_date=trade.entry_date,
            entry_price=round(trade.entry_price, 2),
            sell_date=trade.exit_date,
            sell_price=round(trade.exit_price, 2),
            would_be_return=round(trade.pnl_pct, 1),
            would_be_pnl=round(trade.pnl, 2),
            days_held=trade.days_held,
            sector=sector
        ))

    # Sort by highest return first (most impressive opportunities)
    opportunities.sort(key=lambda x: x.would_be_return, reverse=True)
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
            import pandas as pd
            row = df.iloc[-1]
            current_data = {
                "current_price": round(float(row['close']), 2),
                "dwap": round(float(row['dwap']), 2) if pd.notna(row.get('dwap')) else None,
                "ma_50": round(float(row['ma_50']), 2) if pd.notna(row.get('ma_50')) else None,
                "ma_200": round(float(row['ma_200']), 2) if pd.notna(row.get('ma_200')) else None,
                "high_52w": round(float(row['high_52w']), 2) if pd.notna(row.get('high_52w')) else None,
                "volume": int(row['volume']) if pd.notna(row.get('volume')) else 0,
                "avg_volume": int(row['vol_avg']) if pd.notna(row.get('vol_avg')) else None,
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


class DashboardResponse(BaseModel):
    """Unified dashboard response — one call, everything the frontend needs."""
    regime_forecast: Optional[dict] = None
    buy_signals: List[dict] = []
    positions_with_guidance: List[dict] = []
    watchlist: List[dict] = []
    market_stats: dict = {}
    recent_signals: List[dict] = []
    missed_opportunities: List[dict] = []
    generated_at: str


@router.get("/dashboard")
async def get_dashboard_data(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_optional),
    momentum_top_n: int = 20,
    fresh_days: int = 5
):
    """
    Unified dashboard endpoint.

    Returns buy signals (ensemble), sell guidance for positions,
    regime forecast, watchlist, and market stats in one call.
    """
    from app.core.config import settings
    from app.services.market_regime import market_regime_service, RegimeForecast
    import pandas as pd

    if not scanner_service.data_cache:
        raise HTTPException(status_code=503, detail="Price data not loaded")

    # --- Regime forecast ---
    regime_forecast_data = None
    regime_forecast_obj = None
    try:
        spy_df = scanner_service.data_cache.get('SPY')
        vix_df = scanner_service.data_cache.get('^VIX')
        if spy_df is not None and len(spy_df) >= 200:
            regime_forecast_obj = market_regime_service.predict_transitions(
                spy_df=spy_df,
                universe_dfs=scanner_service.data_cache,
                vix_df=vix_df
            )
            regime_forecast_data = regime_forecast_obj.to_dict()
    except Exception as e:
        import traceback
        print(f"Regime forecast error: {e}")
        traceback.print_exc()

    # --- Buy signals (ensemble double-signals with freshness tracking) ---
    buy_signals = []
    try:
        # Get DWAP signals and momentum rankings
        dwap_signals = await scanner_service.scan(refresh_data=False, apply_market_filter=True)
        dwap_by_symbol = {s.symbol: s for s in dwap_signals}

        momentum_rankings = scanner_service.rank_stocks_momentum(apply_market_filter=True)
        momentum_by_symbol = {
            r.symbol: {'rank': i + 1, 'data': r}
            for i, r in enumerate(momentum_rankings[:momentum_top_n])
        }

        for symbol in dwap_by_symbol:
            if symbol in momentum_by_symbol:
                dwap = dwap_by_symbol[symbol]
                mom = momentum_by_symbol[symbol]
                mom_data = mom['data']
                mom_rank = mom['rank']

                dwap_score = min(dwap.pct_above_dwap * 10, 50)
                rank_score = (momentum_top_n - mom_rank + 1) * 2.5
                ensemble_score = dwap_score + rank_score

                crossover_date, days_since = find_dwap_crossover_date(symbol)
                is_fresh = days_since is not None and days_since <= fresh_days

                buy_signals.append({
                    'symbol': symbol,
                    'price': float(dwap.price),
                    'dwap': float(dwap.dwap),
                    'pct_above_dwap': float(dwap.pct_above_dwap),
                    'volume': int(dwap.volume),
                    'volume_ratio': float(dwap.volume_ratio),
                    'is_strong': bool(dwap.is_strong),
                    'momentum_rank': int(mom_rank),
                    'momentum_score': round(float(mom_data.composite_score), 2),
                    'short_momentum': round(float(mom_data.short_momentum), 2),
                    'long_momentum': round(float(mom_data.long_momentum), 2),
                    'ensemble_score': round(float(ensemble_score), 1),
                    'dwap_crossover_date': crossover_date,
                    'days_since_crossover': int(days_since) if days_since is not None else None,
                    'is_fresh': bool(is_fresh),
                })

        # Sort: fresh first by recency, then stale by ensemble score
        buy_signals.sort(key=lambda x: (
            0 if x['is_fresh'] else 1,
            x.get('days_since_crossover') or 999,
            -x['ensemble_score']
        ))
    except Exception as e:
        print(f"Buy signals error: {e}")

    # --- Positions with sell guidance ---
    positions_with_guidance = []
    try:
        if user is None:
            open_positions = []
        else:
            result = await db.execute(
                select(Position).where(Position.status == 'open', Position.user_id == user.id)
            )
            open_positions = result.scalars().all()

        pos_dicts = []
        for p in open_positions:
            # Look up current price from data cache (Position table has no current_price column)
            current_price = p.entry_price
            df = scanner_service.data_cache.get(p.symbol)
            if df is not None and len(df) > 0:
                current_price = float(df.iloc[-1]['close'])
            pos_dicts.append({
                'id': p.id,
                'symbol': p.symbol,
                'shares': float(p.shares),
                'entry_price': float(p.entry_price),
                'entry_date': p.created_at.strftime('%Y-%m-%d') if p.created_at else None,
                'current_price': current_price,
                'highest_price': float(getattr(p, 'highest_price', None) or p.entry_price),
            })

        if pos_dicts:
            positions_with_guidance = scanner_service.generate_sell_signals(
                positions=pos_dicts,
                regime_forecast=regime_forecast_obj,
            )
    except Exception as e:
        print(f"Positions guidance error: {e}")

    # --- Watchlist (approaching trigger) ---
    watchlist = []
    try:
        momentum_rankings = scanner_service.rank_stocks_momentum(apply_market_filter=True)
        top_momentum = {
            r.symbol: {'rank': i + 1, 'data': r}
            for i, r in enumerate(momentum_rankings[:momentum_top_n])
        }

        for symbol, mom in top_momentum.items():
            df = scanner_service.data_cache.get(symbol)
            if df is None or len(df) < 1:
                continue

            row = df.iloc[-1]
            price = row['close']
            dwap_val = row.get('dwap')

            if pd.isna(dwap_val) or dwap_val <= 0:
                continue

            pct_above = (price / dwap_val - 1) * 100

            if 3.0 <= pct_above < 5.0:
                mom_data = mom['data']
                watchlist.append({
                    'symbol': symbol,
                    'price': round(float(price), 2),
                    'dwap': round(float(dwap_val), 2),
                    'pct_above_dwap': round(float(pct_above), 2),
                    'distance_to_trigger': round(float(5.0 - pct_above), 2),
                    'momentum_rank': int(mom['rank']),
                    'momentum_score': round(float(mom_data.composite_score), 2),
                })

        watchlist.sort(key=lambda x: x['distance_to_trigger'])
        watchlist = watchlist[:5]
    except Exception as e:
        print(f"Watchlist error: {e}")

    # --- Market stats ---
    market_stats = {}
    try:
        spy_df = scanner_service.data_cache.get('SPY')
        vix_df = scanner_service.data_cache.get('^VIX')
        if spy_df is not None and len(spy_df) > 0:
            market_stats['spy_price'] = round(float(spy_df.iloc[-1]['close']), 2)
        if vix_df is not None and len(vix_df) > 0:
            market_stats['vix_level'] = round(float(vix_df.iloc[-1]['close']), 2)
        if regime_forecast_data:
            market_stats['regime_name'] = regime_forecast_data.get('current_regime_name', '')
            market_stats['regime'] = regime_forecast_data.get('current_regime', '')
        market_stats['signal_count'] = len(buy_signals)
        market_stats['fresh_count'] = len([s for s in buy_signals if s.get('is_fresh')])
    except Exception as e:
        print(f"Market stats error: {e}")

    # --- Missed opportunities (from nightly WF cache, with on-the-fly fallback) ---
    missed_opportunities = []
    TRAILING_STOP_PCT = 0.15  # 15% trailing stop for missed-opportunity simulation
    try:
        from app.core.database import WalkForwardSimulation
        import json as _json

        # Try to read from nightly WF cache first
        nightly_result = await db.execute(
            select(WalkForwardSimulation)
            .where(WalkForwardSimulation.is_nightly_missed_opps == True)
            .where(WalkForwardSimulation.status == "completed")
            .order_by(desc(WalkForwardSimulation.simulation_date))
            .limit(1)
        )
        nightly_sim = nightly_result.scalar_one_or_none()

        if nightly_sim and nightly_sim.trades_json:
            # Parse trades from nightly WF simulation
            trades = _json.loads(nightly_sim.trades_json)

            # Get open position symbols to exclude
            open_syms = set()
            for pg in positions_with_guidance:
                open_syms.add(pg.get('symbol', ''))

            for trade in trades:
                symbol = trade.get('symbol', '')
                if symbol in open_syms:
                    continue

                exit_reason = trade.get('exit_reason', '')
                entry_price = float(trade.get('entry_price', 0))
                entry_date_str = str(trade.get('entry_date', ''))[:10]

                # For simulation_end trades, re-simulate with trailing stop
                # using all available price data (the sim ended early, but we
                # have more data — find where the 15% trailing stop would hit)
                if exit_reason == 'simulation_end':
                    df = scanner_service.data_cache.get(symbol)
                    if df is None or len(df) < 50 or entry_price <= 0:
                        continue

                    # Find entry date in price data and simulate forward
                    entry_ts = pd.Timestamp(entry_date_str)
                    mask = df.index >= entry_ts
                    if not mask.any():
                        continue
                    post_entry = df.loc[mask]

                    high_water = entry_price
                    sell_price = None
                    sell_date = None
                    for idx, row in post_entry.iterrows():
                        price_j = float(row['close'])
                        high_water = max(high_water, price_j)
                        trailing_stop = high_water * (1 - TRAILING_STOP_PCT)
                        if price_j <= trailing_stop:
                            sell_price = price_j
                            sell_date = idx
                            break

                    if sell_price is None:
                        continue  # Trailing stop never hit — position still alive

                    pnl_pct = (sell_price / entry_price - 1) * 100
                    if pnl_pct <= 5.0:
                        continue

                    days_held = (sell_date - entry_ts).days

                    missed_opportunities.append({
                        'symbol': symbol,
                        'entry_date': entry_date_str,
                        'entry_price': round(entry_price, 2),
                        'sell_date': sell_date.strftime('%Y-%m-%d') if hasattr(sell_date, 'strftime') else str(sell_date)[:10],
                        'sell_price': round(sell_price, 2),
                        'would_be_return': round(pnl_pct, 1),
                        'would_be_pnl': round((sell_price - entry_price) * 100, 0),
                        'days_held': int(days_held),
                        'strategy_name': trade.get('strategy_name', 'Ensemble'),
                        'exit_reason': 'trailing_stop',
                    })
                else:
                    # Normal exit (trailing_stop, regime, etc.) — keep as-is
                    pnl_pct = trade.get('pnl_pct', 0)
                    if pnl_pct <= 5.0:
                        continue

                    exit_date = trade.get('exit_date', '')
                    exit_price = trade.get('exit_price', 0)
                    days_held = trade.get('days_held', 0)

                    missed_opportunities.append({
                        'symbol': symbol,
                        'entry_date': entry_date_str,
                        'entry_price': round(float(entry_price), 2),
                        'sell_date': str(exit_date)[:10],
                        'sell_price': round(float(exit_price), 2),
                        'would_be_return': round(float(pnl_pct), 1),
                        'would_be_pnl': round((float(exit_price) - float(entry_price)) * 100, 0),
                        'days_held': int(days_held),
                        'strategy_name': trade.get('strategy_name', 'Ensemble'),
                        'exit_reason': exit_reason,
                    })

            missed_opportunities.sort(key=lambda x: x['would_be_return'], reverse=True)
            missed_opportunities = missed_opportunities[:5]
        else:
            # Fallback: compute on-the-fly from price data
            lookback = 90
            min_price = settings.MIN_PRICE
            min_volume = settings.MIN_VOLUME

            open_syms = set()
            for pg in positions_with_guidance:
                open_syms.add(pg.get('symbol', ''))

            for symbol, df in scanner_service.data_cache.items():
                if df is None or len(df) < 250 or symbol in ('SPY', '^VIX'):
                    continue
                if symbol in open_syms:
                    continue
                if 'dwap' not in df.columns:
                    continue

                recent = df.tail(lookback + 60)
                if len(recent) < lookback:
                    continue

                closes = recent['close'].values
                volumes = recent['volume'].values if 'volume' in recent.columns else None
                dwaps = recent['dwap'].values
                dates = recent.index

                for i in range(1, min(lookback, len(recent) - 1)):
                    if dwaps[i] <= 0 or pd.isna(dwaps[i]) or dwaps[i-1] <= 0 or pd.isna(dwaps[i-1]):
                        continue
                    if closes[i] < min_price:
                        continue
                    if volumes is not None and volumes[i] < min_volume:
                        continue

                    prev_pct = (closes[i-1] / dwaps[i-1] - 1)
                    curr_pct = (closes[i] / dwaps[i] - 1)

                    if prev_pct < 0.05 and curr_pct >= 0.05:
                        entry_price = float(closes[i])
                        entry_date = dates[i]

                        high_water = entry_price
                        sell_price = None
                        sell_date = None

                        for j in range(i + 1, len(recent)):
                            price_j = float(closes[j])
                            high_water = max(high_water, price_j)
                            trailing_stop = high_water * (1 - TRAILING_STOP_PCT)
                            if price_j <= trailing_stop:
                                sell_price = price_j
                                sell_date = dates[j]
                                break

                        # Skip if trailing stop never hit — position still alive
                        if sell_price is None:
                            break

                        pnl_pct = (sell_price / entry_price - 1) * 100
                        days_held = (sell_date - entry_date).days

                        if pnl_pct > 5.0:
                            missed_opportunities.append({
                                'symbol': symbol,
                                'entry_date': entry_date.strftime('%Y-%m-%d') if hasattr(entry_date, 'strftime') else str(entry_date)[:10],
                                'entry_price': round(entry_price, 2),
                                'sell_date': sell_date.strftime('%Y-%m-%d') if hasattr(sell_date, 'strftime') else str(sell_date)[:10],
                                'sell_price': round(sell_price, 2),
                                'would_be_return': round(pnl_pct, 1),
                                'would_be_pnl': round((sell_price - entry_price) * 100, 0),
                                'days_held': int(days_held),
                            })
                        break

            missed_opportunities.sort(key=lambda x: x['would_be_return'], reverse=True)
            missed_opportunities = missed_opportunities[:5]
    except Exception as e:
        print(f"Missed opportunities error: {e}")

    # --- Recent signals with performance ---
    recent_signals = []
    try:
        result = await db.execute(
            select(Signal).where(Signal.signal_type == 'BUY')
            .order_by(desc(Signal.created_at)).limit(5)
        )
        for sig in result.scalars().all():
            current_price = None
            df = scanner_service.data_cache.get(sig.symbol)
            if df is not None and len(df) > 0:
                current_price = round(float(df.iloc[-1]['close']), 2)
            perf_pct = round((current_price / sig.price - 1) * 100, 1) if current_price and sig.price > 0 else None
            recent_signals.append({
                'symbol': sig.symbol,
                'signal_date': sig.created_at.strftime('%Y-%m-%d'),
                'signal_price': round(float(sig.price), 2),
                'current_price': current_price,
                'performance_pct': perf_pct,
            })
    except Exception as e:
        print(f"Recent signals error: {e}")

    # Filter out stocks the user already holds from buy signals
    open_position_syms = {pg.get('symbol', '') for pg in positions_with_guidance}
    buy_signals = [s for s in buy_signals if s['symbol'] not in open_position_syms]

    return {
        'regime_forecast': regime_forecast_data,
        'buy_signals': buy_signals,
        'positions_with_guidance': positions_with_guidance,
        'watchlist': watchlist,
        'market_stats': market_stats,
        'recent_signals': recent_signals,
        'missed_opportunities': missed_opportunities,
        'generated_at': datetime.now().isoformat(),
    }


class MomentumRankingItem(BaseModel):
    rank: int
    symbol: str
    price: float
    momentum_score: float
    short_momentum: float
    long_momentum: float
    volatility: float
    near_50d_high: float
    passes_quality: bool


class MomentumRankingsResponse(BaseModel):
    rankings: List[MomentumRankingItem]
    market_filter_active: bool
    generated_at: str


@router.get("/momentum-rankings", response_model=MomentumRankingsResponse)
async def get_momentum_rankings(top_n: int = 20):
    """
    Get current top stocks by momentum score.

    Returns the top N stocks ranked by the momentum scoring formula:
    score = short_momentum * 0.5 + long_momentum * 0.3 - volatility * 0.2

    Stocks must pass quality filters:
    - Price > MA20 and MA50 (uptrend)
    - Within 5% of 50-day high (breakout potential)
    - Volume > 500,000
    - Price > $20

    These are the same stocks that the momentum strategy would consider buying.
    """
    from app.core.config import settings

    if not scanner_service.data_cache:
        raise HTTPException(status_code=503, detail="Price data not loaded")

    rankings = scanner_service.rank_stocks_momentum(apply_market_filter=True)

    if not rankings:
        # Market filter might have blocked all signals
        return MomentumRankingsResponse(
            rankings=[],
            market_filter_active=settings.MARKET_FILTER_ENABLED,
            generated_at=datetime.now().isoformat()
        )

    return MomentumRankingsResponse(
        rankings=[
            MomentumRankingItem(
                rank=i + 1,
                symbol=r.symbol,
                price=r.price,
                momentum_score=round(r.composite_score, 2),
                short_momentum=round(r.short_momentum, 2),
                long_momentum=round(r.long_momentum, 2),
                volatility=round(r.volatility, 2),
                near_50d_high=round(r.dist_from_50d_high, 2),
                passes_quality=r.passes_quality_filter
            )
            for i, r in enumerate(rankings[:top_n])
        ],
        market_filter_active=settings.MARKET_FILTER_ENABLED,
        generated_at=datetime.now().isoformat()
    )


class DoubleSignalItem(BaseModel):
    """A stock with both DWAP trigger AND top momentum ranking."""
    symbol: str
    price: float
    # DWAP data
    dwap: float
    pct_above_dwap: float
    volume: int
    volume_ratio: float
    is_strong: bool
    # Momentum data
    momentum_rank: int
    momentum_score: float
    short_momentum: float
    long_momentum: float
    # Combined score (higher = better)
    ensemble_score: float
    # Crossover tracking
    dwap_crossover_date: Optional[str] = None  # When stock first crossed DWAP +5%
    days_since_crossover: Optional[int] = None  # Days since the crossover
    is_fresh: bool = False  # True if crossover was recent (actionable buy signal)


class DoubleSignalsResponse(BaseModel):
    signals: List[DoubleSignalItem]
    fresh_signals: List[DoubleSignalItem]  # Only fresh (recent crossover) signals
    dwap_only_count: int
    momentum_only_count: int
    fresh_count: int  # Number of actionable fresh signals
    stale_count: int  # Number of stale (old crossover) signals
    fresh_threshold_days: int  # Days threshold for "fresh" (e.g., 5)
    market_filter_active: bool
    generated_at: str


def find_dwap_crossover_date(symbol: str, threshold_pct: float = 5.0, lookback_days: int = 60) -> tuple:
    """
    Find when a stock first crossed above DWAP threshold in recent history.
    Returns (crossover_date, days_since) or (None, None) if not found.
    """
    import pandas as pd

    df = scanner_service.data_cache.get(symbol)
    if df is None or len(df) < 200:
        return None, None

    # Look at recent history (last N trading days)
    recent = df.tail(lookback_days)
    if len(recent) < 2:
        return None, None

    crossover_date = None

    # Scan from oldest to newest to find first crossover
    for i in range(1, len(recent)):
        prev_row = recent.iloc[i - 1]
        curr_row = recent.iloc[i]

        prev_dwap = prev_row.get('dwap')
        curr_dwap = curr_row.get('dwap')
        prev_close = prev_row['close']
        curr_close = curr_row['close']

        if pd.isna(prev_dwap) or pd.isna(curr_dwap) or prev_dwap <= 0 or curr_dwap <= 0:
            continue

        prev_pct = (prev_close / prev_dwap - 1) * 100
        curr_pct = (curr_close / curr_dwap - 1) * 100

        # Check for crossover: was below threshold, now above
        if prev_pct < threshold_pct and curr_pct >= threshold_pct:
            crossover_date = curr_row.name
            break

    if crossover_date is None:
        # Stock may have been above threshold for the entire lookback period
        # Check if first day was already above threshold
        first_row = recent.iloc[0]
        first_dwap = first_row.get('dwap')
        if first_dwap and first_dwap > 0:
            first_pct = (first_row['close'] / first_dwap - 1) * 100
            if first_pct >= threshold_pct:
                crossover_date = first_row.name

    if crossover_date is not None:
        # Calculate days since crossover
        today = df.index[-1]
        if hasattr(crossover_date, 'tz') and crossover_date.tz is not None:
            crossover_date = crossover_date.tz_localize(None)
        if hasattr(today, 'tz') and today.tz is not None:
            today = today.tz_localize(None)
        days_since = (today - crossover_date).days
        date_str = crossover_date.strftime('%Y-%m-%d')
        return date_str, days_since

    return None, None


class ApproachingTriggerItem(BaseModel):
    """A momentum stock approaching the DWAP +5% trigger."""
    symbol: str
    price: float
    dwap: float
    pct_above_dwap: float
    distance_to_trigger: float  # How far from +5% (e.g., 1.5 means +1.5% to go)
    momentum_rank: int
    momentum_score: float
    short_momentum: float
    long_momentum: float


class ApproachingTriggerResponse(BaseModel):
    approaching: List[ApproachingTriggerItem]
    market_filter_active: bool
    trigger_threshold: float  # The threshold they're approaching (e.g., 5.0)
    generated_at: str


@router.get("/approaching-trigger", response_model=ApproachingTriggerResponse)
async def get_approaching_trigger(
    momentum_top_n: int = 20,
    min_pct: float = 3.0,
    max_pct: float = 5.0
):
    """
    Get momentum stocks approaching the DWAP +5% trigger.

    Shows stocks in the top momentum rankings that are at +3-4% above DWAP,
    meaning they're close to triggering a double signal but haven't yet.

    These are "watch list" stocks - if they push up another 1-2%, they'll
    become actionable double signals.

    - **momentum_top_n**: Consider top N momentum stocks (default 20)
    - **min_pct**: Minimum % above DWAP to include (default 3.0)
    - **max_pct**: Maximum % above DWAP (default 5.0 - the trigger threshold)
    """
    from app.core.config import settings
    import pandas as pd

    if not scanner_service.data_cache:
        raise HTTPException(status_code=503, detail="Price data not loaded")

    # Get momentum rankings
    momentum_rankings = scanner_service.rank_stocks_momentum(apply_market_filter=True)
    top_momentum = {
        r.symbol: {'rank': i + 1, 'data': r}
        for i, r in enumerate(momentum_rankings[:momentum_top_n])
    }

    approaching = []

    for symbol, mom in top_momentum.items():
        df = scanner_service.data_cache.get(symbol)
        if df is None or len(df) < 1:
            continue

        row = df.iloc[-1]
        price = row['close']
        dwap = row.get('dwap')

        if pd.isna(dwap) or dwap <= 0:
            continue

        pct_above = (price / dwap - 1) * 100  # Convert to percentage

        # Check if in the "approaching" range (3-5%)
        if min_pct <= pct_above < max_pct:
            distance_to_trigger = max_pct - pct_above
            mom_data = mom['data']

            approaching.append(ApproachingTriggerItem(
                symbol=symbol,
                price=round(float(price), 2),
                dwap=round(float(dwap), 2),
                pct_above_dwap=round(pct_above, 2),
                distance_to_trigger=round(distance_to_trigger, 2),
                momentum_rank=mom['rank'],
                momentum_score=round(mom_data.composite_score, 2),
                short_momentum=round(mom_data.short_momentum, 2),
                long_momentum=round(mom_data.long_momentum, 2)
            ))

    # Sort by closest to trigger (smallest distance first)
    approaching.sort(key=lambda x: x.distance_to_trigger)

    return ApproachingTriggerResponse(
        approaching=approaching,
        market_filter_active=settings.MARKET_FILTER_ENABLED,
        trigger_threshold=max_pct,
        generated_at=datetime.now().isoformat()
    )


@router.get("/double-signals", response_model=DoubleSignalsResponse)
async def get_double_signals(
    momentum_top_n: int = 20,
    fresh_days: int = 5
):
    """
    Get stocks with BOTH DWAP trigger (+5% above DWAP) AND top momentum ranking.

    These "double signals" have historically shown 2.5x higher returns than
    DWAP-only signals (2.91% vs 1.16% at 20 days).

    **Fresh vs Stale Signals:**
    - **Fresh**: Crossed DWAP +5% within the last `fresh_days` (default 5) - actionable BUY
    - **Stale**: Crossed longer ago - may have missed the optimal entry

    DWAP typically crosses before momentum picks up, so fresh crossovers are
    the ideal entry points for the ensemble strategy.

    - **momentum_top_n**: Consider top N momentum stocks (default 20)
    - **fresh_days**: Days threshold for "fresh" signals (default 5)
    """
    from app.core.config import settings

    if not scanner_service.data_cache:
        raise HTTPException(status_code=503, detail="Price data not loaded")

    # Get current DWAP signals
    dwap_signals = await scanner_service.scan(refresh_data=False, apply_market_filter=True)
    dwap_by_symbol = {s.symbol: s for s in dwap_signals}

    # Get momentum rankings
    momentum_rankings = scanner_service.rank_stocks_momentum(apply_market_filter=True)
    momentum_by_symbol = {
        r.symbol: {'rank': i + 1, 'data': r}
        for i, r in enumerate(momentum_rankings[:momentum_top_n])
    }

    # Find intersection (double signals)
    double_signals = []
    fresh_signals = []

    for symbol in dwap_by_symbol:
        if symbol in momentum_by_symbol:
            dwap = dwap_by_symbol[symbol]
            mom = momentum_by_symbol[symbol]
            mom_data = mom['data']
            mom_rank = mom['rank']

            # Ensemble score: combine DWAP strength (0-100) with inverse rank (20-1 -> 1-20 points)
            dwap_score = min(dwap.pct_above_dwap * 10, 50)  # Max 50 points from DWAP
            rank_score = (momentum_top_n - mom_rank + 1) * 2.5  # Max 50 points from rank
            ensemble_score = dwap_score + rank_score

            # Find when the DWAP crossover occurred
            crossover_date, days_since = find_dwap_crossover_date(symbol)

            # Determine if this is a fresh signal (actionable buy)
            is_fresh = days_since is not None and days_since <= fresh_days

            signal = DoubleSignalItem(
                symbol=symbol,
                price=dwap.price,
                dwap=dwap.dwap,
                pct_above_dwap=dwap.pct_above_dwap,
                volume=dwap.volume,
                volume_ratio=dwap.volume_ratio,
                is_strong=dwap.is_strong,
                momentum_rank=mom_rank,
                momentum_score=round(mom_data.composite_score, 2),
                short_momentum=round(mom_data.short_momentum, 2),
                long_momentum=round(mom_data.long_momentum, 2),
                ensemble_score=round(ensemble_score, 1),
                dwap_crossover_date=crossover_date,
                days_since_crossover=days_since,
                is_fresh=is_fresh
            )

            double_signals.append(signal)
            if is_fresh:
                fresh_signals.append(signal)

    # Sort: fresh signals first (by recency), then stale by ensemble score
    double_signals.sort(key=lambda x: (
        0 if x.is_fresh else 1,  # Fresh first
        x.days_since_crossover if x.days_since_crossover else 999,  # Then by recency
        -x.ensemble_score  # Then by score
    ))

    # Sort fresh signals by recency then score
    fresh_signals.sort(key=lambda x: (
        x.days_since_crossover if x.days_since_crossover else 0,
        -x.ensemble_score
    ))

    return DoubleSignalsResponse(
        signals=double_signals,
        fresh_signals=fresh_signals,
        dwap_only_count=len(dwap_by_symbol) - len(double_signals),
        momentum_only_count=len(momentum_by_symbol) - len(double_signals),
        fresh_count=len(fresh_signals),
        stale_count=len(double_signals) - len(fresh_signals),
        fresh_threshold_days=fresh_days,
        market_filter_active=settings.MARKET_FILTER_ENABLED,
        generated_at=datetime.now().isoformat()
    )

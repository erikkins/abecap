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
    is_strong: bool
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

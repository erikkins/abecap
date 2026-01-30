"""
Stocker API - Complete FastAPI Backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import asyncio
import random

# ============================================================================
# Configuration
# ============================================================================

class Config:
    DWAP_THRESHOLD_PCT = 5.0
    STOP_LOSS_PCT = 8.0
    PROFIT_TARGET_PCT = 20.0
    MAX_POSITIONS = 15
    POSITION_SIZE_PCT = 6.0
    MIN_VOLUME = 500_000
    MIN_PRICE = 20.0

config = Config()

# ============================================================================
# Pydantic Models
# ============================================================================

class Signal(BaseModel):
    symbol: str
    price: float
    dwap: float
    pct_above_dwap: float
    volume: int
    volume_ratio: float
    stop_loss: float
    profit_target: float
    is_strong: bool = False
    timestamp: str

class ScanResponse(BaseModel):
    timestamp: str
    total_signals: int
    strong_signals: int
    signals: List[Signal]

class Position(BaseModel):
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

class PositionsResponse(BaseModel):
    positions: List[Position]
    total_value: float
    total_pnl_pct: float

class EquityPoint(BaseModel):
    date: str
    equity: float

class OpenPositionRequest(BaseModel):
    symbol: str
    shares: Optional[float] = None
    price: Optional[float] = None

# ============================================================================
# Mock Data Storage
# ============================================================================

class Storage:
    def __init__(self):
        self.signals = [
            Signal(symbol='NVDA', price=512.30, dwap=473.43, pct_above_dwap=8.2, 
                   volume=45000000, volume_ratio=2.1, stop_loss=471.32, profit_target=614.76,
                   is_strong=True, timestamp=datetime.now().isoformat()),
            Signal(symbol='AMD', price=158.90, dwap=149.20, pct_above_dwap=6.5,
                   volume=32000000, volume_ratio=1.8, stop_loss=146.19, profit_target=190.68,
                   is_strong=True, timestamp=datetime.now().isoformat()),
            Signal(symbol='AVGO', price=1245.50, dwap=1177.26, pct_above_dwap=5.8,
                   volume=8000000, volume_ratio=1.4, stop_loss=1145.86, profit_target=1494.60,
                   is_strong=False, timestamp=datetime.now().isoformat()),
        ]
        
        self.positions = [
            Position(id=1, symbol='AAPL', shares=55, entry_price=182.50, 
                    entry_date='2025-01-12', current_price=195.90,
                    stop_loss=167.90, profit_target=219.00, pnl_pct=7.34, days_held=18),
            Position(id=2, symbol='META', shares=22, entry_price=485.00,
                    entry_date='2025-01-22', current_price=468.20,
                    stop_loss=446.20, profit_target=582.00, pnl_pct=-3.46, days_held=8),
            Position(id=3, symbol='GOOGL', shares=40, entry_price=141.20,
                    entry_date='2025-01-05', current_price=152.80,
                    stop_loss=129.90, profit_target=169.44, pnl_pct=8.22, days_held=25),
        ]
        
        self.equity_history = []
        for i in range(60):
            date = datetime.now() - timedelta(days=59-i)
            equity = 100000 * (1 + (i * 0.011) + (random.random() - 0.3) * 0.012)
            self.equity_history.append(EquityPoint(date=date.strftime('%Y-%m-%d'), equity=round(equity, 2)))
        
        self.position_id = 3

storage = Storage()

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Stocker API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"name": "Stocker API", "version": "2.0.0", "docs": "/docs"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/signals/scan", response_model=ScanResponse)
async def scan_market(refresh: bool = True):
    if refresh:
        await asyncio.sleep(0.3)
    strong = [s for s in storage.signals if s.is_strong]
    return ScanResponse(
        timestamp=datetime.now().isoformat(),
        total_signals=len(storage.signals),
        strong_signals=len(strong),
        signals=storage.signals
    )

@app.get("/api/signals/latest", response_model=ScanResponse)
async def get_latest_signals(limit: int = 20, strong_only: bool = False):
    signals = [s for s in storage.signals if s.is_strong] if strong_only else storage.signals
    return ScanResponse(
        timestamp=datetime.now().isoformat(),
        total_signals=len(signals[:limit]),
        strong_signals=len([s for s in signals[:limit] if s.is_strong]),
        signals=signals[:limit]
    )

@app.get("/api/portfolio/positions", response_model=PositionsResponse)
async def get_positions():
    total_value = sum(p.shares * p.current_price for p in storage.positions)
    total_cost = sum(p.shares * p.entry_price for p in storage.positions)
    total_pnl_pct = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
    return PositionsResponse(positions=storage.positions, total_value=round(total_value, 2), total_pnl_pct=round(total_pnl_pct, 2))

@app.post("/api/portfolio/positions")
async def open_position(request: OpenPositionRequest):
    storage.position_id += 1
    price = request.price or 100.0
    shares = request.shares or (10000 / price)
    position = Position(
        id=storage.position_id, symbol=request.symbol.upper(), shares=round(shares, 2),
        entry_price=price, entry_date=datetime.now().strftime('%Y-%m-%d'), current_price=price,
        stop_loss=round(price * 0.92, 2), profit_target=round(price * 1.20, 2), pnl_pct=0, days_held=0
    )
    storage.positions.append(position)
    return {"message": f"Opened position in {request.symbol}", "position": position}

@app.delete("/api/portfolio/positions/{position_id}")
async def close_position(position_id: int):
    storage.positions = [p for p in storage.positions if p.id != position_id]
    return {"message": "Position closed"}

@app.get("/api/portfolio/equity")
async def get_equity_curve():
    return {"equity_curve": storage.equity_history}

@app.get("/api/config")
async def get_config():
    return {
        "dwap_threshold_pct": config.DWAP_THRESHOLD_PCT,
        "stop_loss_pct": config.STOP_LOSS_PCT,
        "profit_target_pct": config.PROFIT_TARGET_PCT,
        "max_positions": config.MAX_POSITIONS,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

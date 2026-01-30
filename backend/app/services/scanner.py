"""
Scanner Service - Core trading signal generation

Implements the optimized DWAP strategy:
- Buy: Price > DWAP × 1.05
- Stop: 8%
- Target: 20%
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import asyncio

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from app.core.config import settings, get_universe


@dataclass
class SignalData:
    """Trading signal data"""
    symbol: str
    signal_type: str
    price: float
    dwap: float
    pct_above_dwap: float
    volume: int
    volume_ratio: float
    stop_loss: float
    profit_target: float
    ma_50: float
    ma_200: float
    high_52w: float
    is_strong: bool
    timestamp: str
    
    def to_dict(self):
        return asdict(self)


class ScannerService:
    """
    Stock scanner service
    
    Scans universe for DWAP-based buy signals
    """
    
    def __init__(self):
        self.universe = get_universe()
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.last_scan: Optional[datetime] = None
        self.signals: List[SignalData] = []
    
    # =========================================================================
    # INDICATORS
    # =========================================================================
    
    @staticmethod
    def dwap(prices: pd.Series, volumes: pd.Series, period: int = 200) -> pd.Series:
        """Daily Weighted Average Price"""
        pv = prices * volumes
        return pv.rolling(period, min_periods=50).sum() / volumes.rolling(period, min_periods=50).sum()
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(period, min_periods=1).mean()
    
    @staticmethod
    def high_52w(prices: pd.Series) -> pd.Series:
        """52-week rolling high"""
        return prices.rolling(252, min_periods=1).max()
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    async def fetch_data(self, symbols: List[str] = None, period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data from yfinance
        
        Args:
            symbols: List of tickers (default: full universe)
            period: Data period (1y, 2y, 5y, max)
        
        Returns:
            Dict mapping symbol to DataFrame with indicators
        """
        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance not installed")
        
        symbols = symbols or self.universe
        
        # Run yfinance in thread pool (it's blocking)
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: yf.download(symbols, period=period, progress=False, threads=True)
        )
        
        # Process each symbol
        for symbol in symbols:
            try:
                df = pd.DataFrame({
                    'date': data.index,
                    'open': data['Open'][symbol] if len(symbols) > 1 else data['Open'],
                    'high': data['High'][symbol] if len(symbols) > 1 else data['High'],
                    'low': data['Low'][symbol] if len(symbols) > 1 else data['Low'],
                    'close': data['Close'][symbol] if len(symbols) > 1 else data['Close'],
                    'volume': data['Volume'][symbol] if len(symbols) > 1 else data['Volume'],
                }).dropna()
                
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                
                # Compute indicators
                df['dwap'] = self.dwap(df['close'], df['volume'])
                df['ma_50'] = self.sma(df['close'], 50)
                df['ma_200'] = self.sma(df['close'], 200)
                df['vol_avg'] = self.sma(df['volume'], 200)
                df['high_52w'] = self.high_52w(df['close'])
                
                self.data_cache[symbol] = df
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
        
        return self.data_cache
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def analyze_stock(self, symbol: str) -> Optional[SignalData]:
        """
        Analyze single stock for buy signals
        
        Buy conditions:
        - Price > DWAP × (1 + threshold)
        - Volume > minimum
        - Price > minimum
        
        Strong signal (bonus):
        - Volume > avg × spike_mult
        - Price > MA50 > MA200 (healthy trend)
        """
        if symbol not in self.data_cache:
            return None
        
        df = self.data_cache[symbol]
        if len(df) < 200:
            return None
        
        # Current values
        row = df.iloc[-1]
        price = row['close']
        volume = row['volume']
        current_dwap = row['dwap']
        vol_avg = row['vol_avg']
        ma_50 = row['ma_50']
        ma_200 = row['ma_200']
        h_52w = row['high_52w']
        
        # Skip if DWAP invalid
        if pd.isna(current_dwap) or current_dwap <= 0:
            return None
        
        # Calculate metrics
        pct_above_dwap = (price / current_dwap - 1) * 100
        vol_ratio = volume / vol_avg if vol_avg > 0 else 0
        
        # Check buy conditions
        is_buy = (
            pct_above_dwap >= settings.DWAP_THRESHOLD_PCT and
            volume >= settings.MIN_VOLUME and
            price >= settings.MIN_PRICE
        )
        
        if not is_buy:
            return None
        
        # Strong signal check
        is_strong = (
            vol_ratio >= settings.VOLUME_SPIKE_MULT and
            price > ma_50 > ma_200
        )
        
        # Calculate stops/targets
        stop_loss = price * (1 - settings.STOP_LOSS_PCT / 100)
        profit_target = price * (1 + settings.PROFIT_TARGET_PCT / 100)
        
        return SignalData(
            symbol=symbol,
            signal_type='BUY',
            price=round(price, 2),
            dwap=round(current_dwap, 2),
            pct_above_dwap=round(pct_above_dwap, 1),
            volume=int(volume),
            volume_ratio=round(vol_ratio, 2),
            stop_loss=round(stop_loss, 2),
            profit_target=round(profit_target, 2),
            ma_50=round(ma_50, 2),
            ma_200=round(ma_200, 2),
            high_52w=round(h_52w, 2),
            is_strong=is_strong,
            timestamp=datetime.now().isoformat()
        )
    
    async def scan(self, refresh_data: bool = True) -> List[SignalData]:
        """
        Run full market scan
        
        Args:
            refresh_data: Whether to fetch fresh data
        
        Returns:
            List of SignalData objects sorted by strength
        """
        if refresh_data or not self.data_cache:
            await self.fetch_data()
        
        self.signals = []
        
        for symbol in self.data_cache:
            signal = self.analyze_stock(symbol)
            if signal:
                self.signals.append(signal)
        
        # Sort: strong signals first, then by pct above DWAP
        self.signals.sort(key=lambda s: (-s.is_strong, -s.pct_above_dwap))
        
        self.last_scan = datetime.now()
        
        return self.signals
    
    def get_strong_signals(self) -> List[SignalData]:
        """Get only strong signals"""
        return [s for s in self.signals if s.is_strong]
    
    def get_watchlist(self, threshold: float = 3.0) -> List[SignalData]:
        """Get stocks approaching DWAP threshold"""
        watchlist = []
        
        for symbol, df in self.data_cache.items():
            if len(df) < 200:
                continue
            
            row = df.iloc[-1]
            price = row['close']
            dwap = row['dwap']
            
            if pd.isna(dwap) or dwap <= 0:
                continue
            
            pct_above = (price / dwap - 1) * 100
            
            # Approaching but not yet at threshold
            if threshold <= pct_above < settings.DWAP_THRESHOLD_PCT:
                watchlist.append({
                    'symbol': symbol,
                    'price': round(price, 2),
                    'dwap': round(dwap, 2),
                    'pct_above_dwap': round(pct_above, 1)
                })
        
        return sorted(watchlist, key=lambda x: -x['pct_above_dwap'])


# Singleton instance
scanner_service = ScannerService()

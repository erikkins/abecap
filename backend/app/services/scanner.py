"""
Scanner Service - Core trading signal generation

Implements the optimized DWAP strategy:
- Buy: Price > DWAP × 1.05
- Stop: 8%
- Target: 20%

Supports full NASDAQ + NYSE universe (~6000 stocks)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import asyncio
import logging

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from app.core.config import settings, get_universe

logger = logging.getLogger(__name__)


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
    # New fields for enhanced analysis
    signal_strength: float = 0.0  # 0-100 composite score
    sector: str = ""
    recommendation: str = ""

    def to_dict(self):
        return asdict(self)


class ScannerService:
    """
    Stock scanner service

    Scans universe for DWAP-based buy signals.
    Supports dynamic universe from NASDAQ + NYSE.
    Auto-loads full universe from S3 cache on startup.
    """

    def __init__(self):
        # Start with config universe, will be replaced by full universe on load
        self.universe = get_universe()
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.last_scan: Optional[datetime] = None
        self.signals: List[SignalData] = []
        self.full_universe_loaded = False

    async def load_full_universe(self, max_cache_age_hours: int = 168):
        """
        Load full NASDAQ + NYSE universe from S3 cache or fetch fresh.

        Uses 7-day cache by default since the stock universe doesn't change frequently.
        This replaces the default 80-stock list with all tradeable stocks (~4500).

        Args:
            max_cache_age_hours: Max age of cached universe (default 168 = 7 days)
        """
        try:
            from app.services.stock_universe import stock_universe_service

            # Use ensure_loaded which checks cache first
            symbols = await stock_universe_service.ensure_loaded(max_cache_age_hours=max_cache_age_hours)
            if symbols and len(symbols) > 100:
                self.universe = symbols
                self.full_universe_loaded = True
                logger.info(f"Loaded full universe: {len(self.universe)} symbols")
            else:
                logger.warning(f"Universe load returned only {len(symbols) if symbols else 0} symbols, keeping default")
            return self.universe
        except Exception as e:
            logger.error(f"Failed to load full universe: {e}")
            return self.universe

    async def ensure_universe_loaded(self):
        """
        Ensure the full universe is loaded.
        Called on Lambda cold start.
        """
        if not self.full_universe_loaded or len(self.universe) < 100:
            await self.load_full_universe()
    
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

    async def fetch_data(self, symbols: List[str] = None, period: str = "5y") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data from yfinance with throttled batch requests.
        This fetches FULL historical data - use fetch_incremental() for daily updates.

        Args:
            symbols: List of tickers (default: full universe)
            period: Data period (1y, 2y, 5y, max)

        Returns:
            Dict mapping symbol to DataFrame with indicators
        """
        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance not installed")

        symbols = symbols or self.universe

        # Batch settings to avoid rate limiting
        BATCH_SIZE = 10  # Fetch 10 stocks at a time
        DELAY_BETWEEN_BATCHES = 1.5  # Seconds between batches
        MAX_RETRIES = 2

        total = len(symbols)
        successful = 0
        failed = []

        print(f"📊 Fetching data for {total} symbols in batches of {BATCH_SIZE}...")

        # Process in batches
        for i in range(0, total, BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

            for retry in range(MAX_RETRIES):
                try:
                    # Run yfinance in thread pool (it's blocking)
                    loop = asyncio.get_event_loop()
                    data = await loop.run_in_executor(
                        None,
                        lambda b=batch: yf.download(
                            b,
                            period=period,
                            progress=False,
                            threads=True,
                            timeout=30
                        )
                    )

                    # Process each symbol in batch
                    for symbol in batch:
                        try:
                            if len(batch) > 1:
                                df = pd.DataFrame({
                                    'date': data.index,
                                    'open': data['Open'][symbol],
                                    'high': data['High'][symbol],
                                    'low': data['Low'][symbol],
                                    'close': data['Close'][symbol],
                                    'volume': data['Volume'][symbol],
                                }).dropna()
                            else:
                                df = pd.DataFrame({
                                    'date': data.index,
                                    'open': data['Open'],
                                    'high': data['High'],
                                    'low': data['Low'],
                                    'close': data['Close'],
                                    'volume': data['Volume'],
                                }).dropna()

                            if len(df) < 50:  # Skip if not enough data
                                continue

                            df['date'] = pd.to_datetime(df['date'])
                            df = df.set_index('date').sort_index()

                            # Compute indicators
                            df['dwap'] = self.dwap(df['close'], df['volume'])
                            df['ma_50'] = self.sma(df['close'], 50)
                            df['ma_200'] = self.sma(df['close'], 200)
                            df['vol_avg'] = self.sma(df['volume'], 200)
                            df['high_52w'] = self.high_52w(df['close'])

                            self.data_cache[symbol] = df
                            successful += 1

                        except Exception as e:
                            if symbol not in failed:
                                failed.append(symbol)

                    # Success - break retry loop
                    break

                except Exception as e:
                    if retry < MAX_RETRIES - 1:
                        print(f"  Batch {batch_num} failed, retrying... ({e})")
                        await asyncio.sleep(DELAY_BETWEEN_BATCHES * 2)
                    else:
                        print(f"  Batch {batch_num} failed after {MAX_RETRIES} retries")
                        failed.extend(batch)

            # Progress update every 10 batches
            if batch_num % 10 == 0 or batch_num == total_batches:
                print(f"  Progress: {batch_num}/{total_batches} batches ({successful} symbols loaded)")

            # Delay between batches to avoid rate limiting
            if i + BATCH_SIZE < total:
                await asyncio.sleep(DELAY_BETWEEN_BATCHES)

        print(f"✅ Loaded {successful}/{total} symbols ({len(failed)} failed)")

        return self.data_cache

    async def fetch_incremental(self, symbols: List[str] = None) -> Dict[str, int]:
        """
        Fetch only NEW data since the last cached date for each symbol.
        This is the efficient daily update - only gets today's prices.

        Args:
            symbols: List of tickers (default: all symbols in cache)

        Returns:
            Dict with counts: {updated: N, failed: N, skipped: N}
        """
        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance not installed")

        # Use symbols from cache if not specified
        symbols = symbols or list(self.data_cache.keys())
        if not symbols:
            logger.warning("No symbols to update - cache is empty")
            return {"updated": 0, "failed": 0, "skipped": 0}

        from datetime import timedelta

        BATCH_SIZE = 20  # Can be larger for incremental since less data
        DELAY_BETWEEN_BATCHES = 1.0

        updated = 0
        failed = 0
        skipped = 0

        today = pd.Timestamp.now().normalize()

        logger.info(f"📊 Incremental update for {len(symbols)} symbols...")

        # Process in batches
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]

            # Find the oldest "last date" in this batch to determine fetch period
            oldest_last_date = today
            for symbol in batch:
                if symbol in self.data_cache and len(self.data_cache[symbol]) > 0:
                    last_date = self.data_cache[symbol].index.max()
                    if last_date < oldest_last_date:
                        oldest_last_date = last_date

            # If all data is from today, skip this batch
            if oldest_last_date >= today - timedelta(days=1):
                skipped += len(batch)
                continue

            # Fetch from oldest_last_date to now (add buffer for safety)
            start_date = (oldest_last_date - timedelta(days=5)).strftime('%Y-%m-%d')

            try:
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None,
                    lambda b=batch, s=start_date: yf.download(
                        b,
                        start=s,
                        progress=False,
                        threads=True,
                        timeout=30
                    )
                )

                if data.empty:
                    skipped += len(batch)
                    continue

                # Process each symbol
                for symbol in batch:
                    try:
                        if symbol not in self.data_cache:
                            skipped += 1
                            continue

                        # Extract new data for this symbol
                        if len(batch) > 1:
                            new_df = pd.DataFrame({
                                'date': data.index,
                                'open': data['Open'][symbol],
                                'high': data['High'][symbol],
                                'low': data['Low'][symbol],
                                'close': data['Close'][symbol],
                                'volume': data['Volume'][symbol],
                            }).dropna()
                        else:
                            new_df = pd.DataFrame({
                                'date': data.index,
                                'open': data['Open'],
                                'high': data['High'],
                                'low': data['Low'],
                                'close': data['Close'],
                                'volume': data['Volume'],
                            }).dropna()

                        if new_df.empty:
                            skipped += 1
                            continue

                        new_df['date'] = pd.to_datetime(new_df['date'])
                        new_df = new_df.set_index('date').sort_index()

                        # Get existing data
                        existing_df = self.data_cache[symbol]
                        last_cached_date = existing_df.index.max()

                        # Filter to only truly new rows
                        new_rows = new_df[new_df.index > last_cached_date]

                        if new_rows.empty:
                            skipped += 1
                            continue

                        # Append new rows to existing data
                        combined = pd.concat([existing_df[['open', 'high', 'low', 'close', 'volume']],
                                            new_rows[['open', 'high', 'low', 'close', 'volume']]])
                        combined = combined[~combined.index.duplicated(keep='last')]
                        combined = combined.sort_index()

                        # Recompute indicators on full dataset
                        combined['dwap'] = self.dwap(combined['close'], combined['volume'])
                        combined['ma_50'] = self.sma(combined['close'], 50)
                        combined['ma_200'] = self.sma(combined['close'], 200)
                        combined['vol_avg'] = self.sma(combined['volume'], 200)
                        combined['high_52w'] = self.high_52w(combined['close'])

                        self.data_cache[symbol] = combined
                        updated += 1

                    except Exception as e:
                        logger.debug(f"Failed to update {symbol}: {e}")
                        failed += 1

            except Exception as e:
                logger.error(f"Batch fetch failed: {e}")
                failed += len(batch)

            # Small delay between batches
            if i + BATCH_SIZE < len(symbols):
                await asyncio.sleep(DELAY_BETWEEN_BATCHES)

        logger.info(f"✅ Incremental update complete: {updated} updated, {skipped} skipped, {failed} failed")

        return {"updated": updated, "failed": failed, "skipped": skipped}
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute indicators if missing from DataFrame (lazy computation)"""
        if 'dwap' not in df.columns:
            df = df.copy()
            df['dwap'] = self.dwap(df['close'], df['volume'])
            df['ma_50'] = self.sma(df['close'], 50)
            df['ma_200'] = self.sma(df['close'], 200)
            df['vol_avg'] = self.sma(df['volume'], 200)
            df['high_52w'] = self.high_52w(df['close'])
        return df

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

        # Ensure indicators are computed (lazy computation)
        df = self._ensure_indicators(df)
        self.data_cache[symbol] = df  # Cache the computed indicators

        # Current values
        row = df.iloc[-1]
        price = row['close']
        volume = row['volume']
        current_dwap = row.get('dwap', np.nan)
        vol_avg = row.get('vol_avg', 1)
        ma_50 = row.get('ma_50', 0)
        ma_200 = row.get('ma_200', 0)
        h_52w = row.get('high_52w', price)
        
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
    
    async def scan(
        self,
        refresh_data: bool = True,
        apply_market_filter: bool = True,
        min_signal_strength: float = 0
    ) -> List[SignalData]:
        """
        Run full market scan with market regime awareness

        Args:
            refresh_data: Whether to fetch fresh data
            apply_market_filter: Apply market regime filtering
            min_signal_strength: Minimum signal strength to include (0-100)

        Returns:
            List of SignalData objects sorted by signal strength
        """
        # In Lambda mode, always try to load from S3 cache first (faster than yfinance)
        import os
        is_lambda = os.environ.get("ENVIRONMENT") == "prod"

        print(f"🔍 Scan called: is_lambda={is_lambda}, data_cache_size={len(self.data_cache)}, refresh_data={refresh_data}")

        if not self.data_cache and is_lambda:
            try:
                from app.services.data_export import data_export_service
                print("📥 Lambda cold start - loading price data from S3...")
                cached_data = data_export_service.import_all()
                if cached_data:
                    self.data_cache = cached_data
                    print(f"✅ Loaded {len(cached_data)} symbols from S3 cache")
                else:
                    print("⚠️ S3 import returned empty data")
            except Exception as e:
                import traceback
                print(f"❌ Failed to load from S3: {e}")
                print(traceback.format_exc())

        # In Lambda mode with S3 data, skip yfinance refresh (use cached data)
        # In local mode or if no S3 data, fetch from yfinance
        should_fetch = (refresh_data and not is_lambda) or not self.data_cache
        if should_fetch:
            await self.fetch_data()

        # Update market analysis
        try:
            from app.services.market_analysis import market_analysis_service
            await market_analysis_service.update_market_state()
            await market_analysis_service.update_sector_strength()
            market_available = True
        except Exception as e:
            logger.warning(f"Market analysis unavailable: {e}")
            market_available = False

        self.signals = []

        for symbol in self.data_cache:
            signal = self.analyze_stock(symbol)
            if not signal:
                continue

            # Calculate signal strength
            if market_available:
                signal.signal_strength = market_analysis_service.calculate_signal_strength(
                    pct_above_dwap=signal.pct_above_dwap,
                    volume_ratio=signal.volume_ratio,
                    is_strong=signal.is_strong,
                    sector=signal.sector
                )

                # Apply market regime filter
                if apply_market_filter:
                    should_take, reason = market_analysis_service.should_take_signal(signal.signal_strength)
                    signal.recommendation = reason
                    if not should_take:
                        continue
            else:
                # Default strength calculation without market data
                signal.signal_strength = (
                    min(signal.pct_above_dwap * 5, 30) +
                    min((signal.volume_ratio - 1) * 15, 30) +
                    (25 if signal.is_strong else 0)
                )
                signal.recommendation = "Market data unavailable"

            # Apply minimum strength filter
            if signal.signal_strength < min_signal_strength:
                continue

            self.signals.append(signal)

        # Sort by signal strength (highest first)
        self.signals.sort(key=lambda s: -s.signal_strength)

        self.last_scan = datetime.now()

        logger.info(f"Scan complete: {len(self.signals)} signals found")

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

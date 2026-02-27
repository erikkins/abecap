"""
Market Data Provider — Dual-source abstraction layer

Provides:
- AlpacaProvider: Primary source for historical daily bars (faster, more reliable for bulk)
- YfinanceProvider: Primary for live intraday quotes + index symbols + fallback
- DualSourceProvider: Orchestrator with automatic failover

Alpaca free tier notes:
- 200 req/min rate limit, up to 10k symbols per multi-bar request
- Cannot serve index symbols (^VIX, ^GSPC) — yfinance required
- 15-minute delay on free tier (non-issue for EOD bars fetched after 4 PM)
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Symbols that Alpaca cannot serve (indices use ^ prefix)
INDEX_SYMBOLS = {'^VIX', '^GSPC', '^DJI', '^IXIC', '^RUT', '^TNX'}


@dataclass
class QuoteData:
    """Live quote for a single symbol."""
    symbol: str
    price: float
    prev_close: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    day_high: Optional[float] = None
    timestamp: Optional[str] = None
    source: str = ""


@dataclass
class SourceHealth:
    """Health tracking for a data source."""
    consecutive_failures: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    total_requests: int = 0
    total_failures: int = 0


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    async def fetch_bars(
        self, symbols: List[str], start_date: str, end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical daily OHLCV bars.

        Returns dict mapping symbol -> DataFrame with columns:
        open, high, low, close, volume (DatetimeIndex, tz-naive)
        """
        ...

    @abstractmethod
    async def fetch_quotes(self, symbols: List[str]) -> Dict[str, QuoteData]:
        """Fetch live/current quotes for symbols."""
        ...

    def supports_symbol(self, symbol: str) -> bool:
        """Whether this provider can serve data for a given symbol."""
        return True


class AlpacaProvider(MarketDataProvider):
    """Alpaca Markets data provider (REST API).

    Uses alpaca-py SDK for historical bars.
    Cannot serve index symbols (^VIX, ^GSPC, etc).
    """

    def __init__(self):
        self._client = None
        self._initialized = False

    def _ensure_client(self):
        """Lazy-init Alpaca client to avoid import cost on cold start."""
        if self._initialized:
            return self._client is not None

        self._initialized = True
        try:
            from app.core.config import settings
            if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
                logger.warning("Alpaca credentials not configured")
                return False

            from alpaca.data.historical import StockHistoricalDataClient
            self._client = StockHistoricalDataClient(
                api_key=settings.ALPACA_API_KEY,
                secret_key=settings.ALPACA_SECRET_KEY,
            )
            logger.info("Alpaca client initialized")
            return True
        except ImportError:
            logger.warning("alpaca-py not installed")
            return False
        except Exception as e:
            logger.error(f"Alpaca client init failed: {e}")
            return False

    def supports_symbol(self, symbol: str) -> bool:
        return symbol not in INDEX_SYMBOLS

    async def fetch_bars(
        self, symbols: List[str], start_date: str, end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        if not self._ensure_client():
            return {}

        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        # Filter out unsupported symbols
        alpaca_symbols = [s for s in symbols if self.supports_symbol(s)]
        if not alpaca_symbols:
            return {}

        result: Dict[str, pd.DataFrame] = {}
        BATCH_SIZE = 100  # Alpaca supports large batches

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

        for i in range(0, len(alpaca_symbols), BATCH_SIZE):
            batch = alpaca_symbols[i:i + BATCH_SIZE]

            try:
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=start_dt,
                    end=end_dt,
                )

                loop = asyncio.get_event_loop()
                bars = await loop.run_in_executor(
                    None, self._client.get_stock_bars, request
                )

                # Convert to per-symbol DataFrames
                for symbol in batch:
                    try:
                        symbol_bars = bars[symbol] if symbol in bars else []
                        if not symbol_bars:
                            continue

                        rows = []
                        for bar in symbol_bars:
                            rows.append({
                                'open': float(bar.open),
                                'high': float(bar.high),
                                'low': float(bar.low),
                                'close': float(bar.close),
                                'volume': int(bar.volume),
                                'date': bar.timestamp,
                            })

                        if not rows:
                            continue

                        df = pd.DataFrame(rows)
                        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                        df = df.set_index('date').sort_index()
                        # Remove duplicate dates (keep last)
                        df = df[~df.index.duplicated(keep='last')]
                        result[symbol] = df

                    except Exception as e:
                        logger.debug(f"Alpaca parse error for {symbol}: {e}")

            except Exception as e:
                logger.error(f"Alpaca batch fetch failed ({len(batch)} symbols): {e}")

            # Rate limit: 200 req/min -> conservative 0.4s between batches
            if i + BATCH_SIZE < len(alpaca_symbols):
                await asyncio.sleep(0.4)

        return result

    async def fetch_quotes(self, symbols: List[str]) -> Dict[str, QuoteData]:
        """Fetch latest quotes from Alpaca (IEX exchange only on free tier)."""
        if not self._ensure_client():
            return {}

        from alpaca.data.requests import StockLatestQuoteRequest

        alpaca_symbols = [s for s in symbols if self.supports_symbol(s)]
        if not alpaca_symbols:
            return {}

        result: Dict[str, QuoteData] = {}
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=alpaca_symbols)
            loop = asyncio.get_event_loop()
            quotes = await loop.run_in_executor(
                None, self._client.get_stock_latest_quote, request
            )

            for symbol in alpaca_symbols:
                try:
                    q = quotes.get(symbol)
                    if q and q.ask_price and q.ask_price > 0:
                        # Use midpoint of bid/ask as price
                        price = (q.bid_price + q.ask_price) / 2 if q.bid_price else q.ask_price
                        result[symbol] = QuoteData(
                            symbol=symbol,
                            price=round(price, 2),
                            timestamp=str(q.timestamp) if q.timestamp else None,
                            source="alpaca",
                        )
                except Exception as e:
                    logger.debug(f"Alpaca quote parse error for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Alpaca quotes fetch failed: {e}")

        return result


class YfinanceProvider(MarketDataProvider):
    """yfinance data provider (wraps existing logic)."""

    async def fetch_bars(
        self, symbols: List[str], start_date: str, end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed")
            return {}

        result: Dict[str, pd.DataFrame] = {}
        BATCH_SIZE = 25
        DELAY = 1.0

        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]

            try:
                kwargs = {"start": start_date, "progress": False, "threads": True, "timeout": 30}
                if end_date:
                    kwargs["end"] = end_date

                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None, lambda b=batch, k=kwargs: yf.download(b, **k)
                )

                if data.empty:
                    continue

                # Detect if yfinance returned multi-level columns
                has_multi = isinstance(data.columns, pd.MultiIndex)

                for symbol in batch:
                    try:
                        if has_multi:
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

                        if df.empty:
                            continue

                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date').sort_index()
                        # Strip timezone to keep everything tz-naive
                        if hasattr(df.index, 'tz') and df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        result[symbol] = df

                    except Exception as e:
                        logger.debug(f"yfinance parse error for {symbol}: {e}")

            except Exception as e:
                logger.error(f"yfinance batch fetch failed ({len(batch)} symbols): {e}")

            if i + BATCH_SIZE < len(symbols):
                await asyncio.sleep(DELAY)

        return result

    async def fetch_quotes(self, symbols: List[str]) -> Dict[str, QuoteData]:
        """Fetch live quotes using yfinance fast_info."""
        try:
            import yfinance as yf
        except ImportError:
            return {}

        result: Dict[str, QuoteData] = {}
        try:
            loop = asyncio.get_event_loop()
            tickers = await loop.run_in_executor(
                None, lambda: yf.Tickers(" ".join(symbols))
            )

            for symbol in symbols:
                try:
                    ticker = tickers.tickers.get(symbol)
                    if not ticker:
                        continue
                    info = ticker.fast_info
                    last_price = info.last_price if hasattr(info, 'last_price') else None
                    prev_close = info.previous_close if hasattr(info, 'previous_close') else None
                    day_high = info.day_high if hasattr(info, 'day_high') else None

                    if last_price:
                        change = last_price - prev_close if prev_close else 0
                        change_pct = (change / prev_close * 100) if prev_close else 0

                        result[symbol] = QuoteData(
                            symbol=symbol,
                            price=round(last_price, 2),
                            prev_close=round(prev_close, 2) if prev_close else None,
                            change=round(change, 2),
                            change_pct=round(change_pct, 2),
                            day_high=round(day_high, 2) if day_high else None,
                            timestamp=datetime.now().isoformat(),
                            source="yfinance",
                        )
                except Exception as e:
                    logger.debug(f"yfinance quote error for {symbol}: {e}")

        except Exception as e:
            logger.error(f"yfinance quotes fetch failed: {e}")

        return result


class DualSourceProvider(MarketDataProvider):
    """Orchestrator: Alpaca primary → yfinance fallback for bars.
    yfinance primary → Alpaca fallback for live quotes.
    Index symbols (^VIX) always via yfinance.
    """

    def __init__(self):
        self.alpaca = AlpacaProvider()
        self.yfinance = YfinanceProvider()
        self.health: Dict[str, SourceHealth] = {
            "alpaca": SourceHealth(),
            "yfinance": SourceHealth(),
        }
        self.force_source: Optional[str] = None  # Override for retry logic
        self._last_bars_source: Optional[str] = None
        self._last_quotes_source: Optional[str] = None

    def _record_success(self, source: str):
        h = self.health[source]
        h.consecutive_failures = 0
        h.last_success = datetime.now()
        h.total_requests += 1

    def _record_failure(self, source: str):
        h = self.health[source]
        h.consecutive_failures += 1
        h.last_failure = datetime.now()
        h.total_requests += 1
        h.total_failures += 1

    @property
    def last_bars_source(self) -> Optional[str]:
        return self._last_bars_source

    @property
    def last_quotes_source(self) -> Optional[str]:
        return self._last_quotes_source

    def get_health_summary(self) -> Dict:
        """Return health status for monitoring."""
        summary = {}
        for name, h in self.health.items():
            if h.consecutive_failures > 5:
                status = "red"
            elif h.consecutive_failures > 0:
                status = "yellow"
            else:
                status = "green"
            summary[name] = {
                "status": status,
                "consecutive_failures": h.consecutive_failures,
                "last_success": h.last_success.isoformat() if h.last_success else None,
                "last_failure": h.last_failure.isoformat() if h.last_failure else None,
                "total_requests": h.total_requests,
                "total_failures": h.total_failures,
            }
        return summary

    async def fetch_bars(
        self, symbols: List[str], start_date: str, end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch bars: Alpaca primary (stock symbols) → yfinance fallback.
        Index symbols always go through yfinance.
        """
        # Separate index vs stock symbols
        index_syms = [s for s in symbols if s in INDEX_SYMBOLS]
        stock_syms = [s for s in symbols if s not in INDEX_SYMBOLS]

        result: Dict[str, pd.DataFrame] = {}

        # Index symbols: always yfinance
        if index_syms:
            t0 = time.time()
            index_data = await self.yfinance.fetch_bars(index_syms, start_date, end_date)
            elapsed = time.time() - t0
            result.update(index_data)
            if index_data:
                self._record_success("yfinance")
                logger.info(f"[yfinance] Fetched {len(index_data)}/{len(index_syms)} index symbols in {elapsed:.1f}s")
            else:
                self._record_failure("yfinance")
                logger.warning(f"[yfinance] Failed to fetch index symbols")

        if not stock_syms:
            return result

        # Force source override (for retry logic)
        if self.force_source == "yfinance":
            t0 = time.time()
            yf_data = await self.yfinance.fetch_bars(stock_syms, start_date, end_date)
            elapsed = time.time() - t0
            result.update(yf_data)
            self._last_bars_source = "yfinance"
            logger.info(f"[yfinance-forced] Fetched {len(yf_data)}/{len(stock_syms)} symbols in {elapsed:.1f}s")
            return result

        # Primary: Alpaca
        t0 = time.time()
        alpaca_data = await self.alpaca.fetch_bars(stock_syms, start_date, end_date)
        elapsed = time.time() - t0

        if alpaca_data:
            self._record_success("alpaca")
            result.update(alpaca_data)
            self._last_bars_source = "alpaca"
            logger.info(f"[alpaca] Fetched {len(alpaca_data)}/{len(stock_syms)} symbols in {elapsed:.1f}s")

            # Check for missing symbols — fallback to yfinance for those
            missing = [s for s in stock_syms if s not in alpaca_data]
            if missing:
                logger.info(f"[yfinance-fallback] Fetching {len(missing)} symbols missing from Alpaca...")
                t0 = time.time()
                yf_fallback = await self.yfinance.fetch_bars(missing, start_date, end_date)
                elapsed = time.time() - t0
                result.update(yf_fallback)
                logger.info(f"[yfinance-fallback] Got {len(yf_fallback)}/{len(missing)} in {elapsed:.1f}s")
        else:
            # Alpaca failed entirely — full fallback to yfinance
            self._record_failure("alpaca")
            logger.warning(f"[alpaca] Failed, falling back to yfinance for all {len(stock_syms)} symbols")

            t0 = time.time()
            yf_data = await self.yfinance.fetch_bars(stock_syms, start_date, end_date)
            elapsed = time.time() - t0

            if yf_data:
                self._record_success("yfinance")
                result.update(yf_data)
                self._last_bars_source = "yfinance"
                logger.info(f"[yfinance-fallback] Fetched {len(yf_data)}/{len(stock_syms)} symbols in {elapsed:.1f}s")
            else:
                self._record_failure("yfinance")
                logger.error("[both] Both Alpaca and yfinance failed to fetch bars")

        return result

    async def fetch_quotes(self, symbols: List[str]) -> Dict[str, QuoteData]:
        """Fetch live quotes: yfinance primary → Alpaca fallback.

        yfinance is primary for live quotes because it covers all exchanges,
        while Alpaca free tier only covers IEX.
        """
        # Force source override
        if self.force_source == "alpaca":
            alpaca_data = await self.alpaca.fetch_quotes(symbols)
            self._last_quotes_source = "alpaca"
            return alpaca_data

        # Primary: yfinance (all exchanges)
        t0 = time.time()
        yf_data = await self.yfinance.fetch_quotes(symbols)
        elapsed = time.time() - t0

        if yf_data:
            self._record_success("yfinance")
            self._last_quotes_source = "yfinance"
            logger.debug(f"[yfinance] Fetched {len(yf_data)}/{len(symbols)} quotes in {elapsed:.1f}s")

            # Fill missing from Alpaca
            missing = [s for s in symbols if s not in yf_data and s not in INDEX_SYMBOLS]
            if missing:
                alpaca_quotes = await self.alpaca.fetch_quotes(missing)
                yf_data.update(alpaca_quotes)

            return yf_data
        else:
            # yfinance failed — try Alpaca for non-index symbols
            self._record_failure("yfinance")
            stock_syms = [s for s in symbols if s not in INDEX_SYMBOLS]
            if stock_syms:
                logger.warning(f"[yfinance] Failed, falling back to Alpaca for {len(stock_syms)} quotes")
                alpaca_data = await self.alpaca.fetch_quotes(stock_syms)
                if alpaca_data:
                    self._record_success("alpaca")
                    self._last_quotes_source = "alpaca"
                else:
                    self._record_failure("alpaca")
                return alpaca_data

            return {}


# Singleton instance
market_data_provider = DualSourceProvider()

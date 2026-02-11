#!/usr/bin/env python3
"""
Fetch 5 years of historical price data for ALL stocks and save locally.

This script:
1. Fetches all NASDAQ and NYSE stock tickers
2. Ensures required symbols (SPY, major stocks) are always included
3. Fetches 5 years of OHLCV data for each
4. Computes technical indicators (DWAP, MAs, etc.)
5. Saves to a gzipped pickle file locally

Run locally - takes ~30-60 minutes for full universe.
"""

import os
import sys
import pickle
import gzip
import time
import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

import pandas as pd
import numpy as np
import yfinance as yf

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "backend" / "data" / "prices"
OUTPUT_FILE = OUTPUT_DIR / "all_data_5y_full.pkl.gz"
BATCH_SIZE = 50   # Smaller batches for stability
MAX_WORKERS = 3   # Fewer workers to avoid rate limiting
PERIOD = "5y"

# Required symbols that MUST be included - fetch these first
REQUIRED_SYMBOLS = [
    # Market indices and ETFs
    'SPY', '^VIX', 'QQQ', 'IWM', 'DIA', '^GSPC', '^IXIC', '^DJI',
    # Mega-cap tech
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Major stocks by sector
    'JPM', 'BAC', 'WFC', 'GS', 'MS',  # Financials
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK',  # Healthcare
    'XOM', 'CVX', 'COP',  # Energy
    'HD', 'LOW', 'TGT', 'WMT', 'COST',  # Consumer
    'V', 'MA', 'PYPL',  # Payments
    'DIS', 'NFLX', 'CMCSA',  # Media
    'CRM', 'ORCL', 'ADBE', 'NOW', 'INTU',  # Software
    'AVGO', 'AMD', 'INTC', 'QCOM', 'TXN',  # Semiconductors
    'BA', 'CAT', 'GE', 'HON', 'UPS',  # Industrials
    'KO', 'PEP', 'MCD', 'SBUX',  # Consumer staples
    'T', 'VZ', 'TMUS',  # Telecom
    # Popular retail stocks
    'PLTR', 'SOFI', 'RIVN', 'LCID', 'NIO', 'GME', 'AMC',
]

# Exclusion patterns - leveraged/inverse ETFs, etc.
EXCLUDE_PATTERNS = [
    'TQQQ', 'SQQQ', 'UPRO', 'SPXS', 'UVXY', 'VXX', 'SVXY', 'TVIX',
    'FAS', 'FAZ', 'TNA', 'TZA', 'SOXL', 'SOXS', 'LABU', 'LABD',
    'NUGT', 'DUST', 'JNUG', 'JDST', 'GUSH', 'DRIP', 'ERX', 'ERY',
    'SPXU', 'SDOW', 'UDOW', 'URTY', 'SRTY', 'UCO', 'SCO', 'BOIL', 'KOLD',
    'CURE', 'EDC', 'EDZ', 'RETL', 'RETS', 'WANT', 'PASS',
]

# Thread-safe data cache
data_cache = {}
cache_lock = Lock()


def is_valid_symbol(sym: str) -> bool:
    """Check if a symbol is valid for our universe."""
    if not sym:
        return False
    # Allow symbols up to 5 characters (e.g., GOOGL)
    if len(sym) > 5:
        return False
    # Allow alphanumeric and special symbols like ^VIX
    if sym.startswith('^'):
        return True
    # Must be uppercase letters only (no numbers, dots, etc.)
    if not sym.isalpha():
        return False
    if not sym.isupper():
        return False
    # Exclude leveraged/inverse ETFs
    if sym in EXCLUDE_PATTERNS:
        return False
    return True


def get_all_tickers():
    """Get all stock tickers from NASDAQ and NYSE."""
    print("Fetching stock universe...")
    all_symbols = set()

    # Always include required symbols first
    for sym in REQUIRED_SYMBOLS:
        all_symbols.add(sym)
    print(f"  Added {len(REQUIRED_SYMBOLS)} required symbols")

    try:
        import urllib.request
        import ssl

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        # NASDAQ listed stocks
        nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        print("  Fetching NASDAQ list...")
        nasdaq_response = urllib.request.urlopen(nasdaq_url, context=ctx, timeout=30)
        nasdaq_data = nasdaq_response.read().decode('utf-8')
        nasdaq_lines = nasdaq_data.strip().split('\n')[1:-1]
        nasdaq_count = 0
        for line in nasdaq_lines:
            parts = line.split('|')
            if len(parts) > 1:
                sym = parts[0].strip()
                if is_valid_symbol(sym):
                    all_symbols.add(sym)
                    nasdaq_count += 1
        print(f"  Got {nasdaq_count} valid NASDAQ symbols")

        # NYSE/AMEX listed stocks
        other_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
        print("  Fetching NYSE/AMEX list...")
        other_response = urllib.request.urlopen(other_url, context=ctx, timeout=30)
        other_data = other_response.read().decode('utf-8')
        other_lines = other_data.strip().split('\n')[1:-1]
        other_count = 0
        for line in other_lines:
            parts = line.split('|')
            if len(parts) > 1:
                sym = parts[0].strip()
                if is_valid_symbol(sym):
                    all_symbols.add(sym)
                    other_count += 1
        print(f"  Got {other_count} valid NYSE/AMEX symbols")

    except Exception as e:
        print(f"  Warning: Error fetching from NASDAQ: {e}")
        print("  Continuing with required symbols + fallback list...")

        # Add S&P 500 from Wikipedia as fallback
        try:
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            for sym in sp500['Symbol'].tolist():
                sym = sym.replace('.', '-')  # BRK.B -> BRK-B
                if is_valid_symbol(sym):
                    all_symbols.add(sym)
            print(f"  Added S&P 500 symbols from Wikipedia")
        except Exception as e2:
            print(f"  Warning: Could not fetch S&P 500: {e2}")

    # Filter and dedupe
    filtered = [s for s in all_symbols if s not in EXCLUDE_PATTERNS]
    print(f"Total universe: {len(filtered)} symbols")
    return sorted(filtered)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators on a price DataFrame."""
    if len(df) < 50:
        return df

    df.columns = [c.lower() for c in df.columns]

    # DWAP - Daily Weighted Average Price (200-day)
    pv = df['close'] * df['volume']
    df['dwap'] = pv.rolling(200, min_periods=50).sum() / df['volume'].rolling(200, min_periods=50).sum()

    # Moving Averages
    df['ma_50'] = df['close'].rolling(50, min_periods=1).mean()
    df['ma_200'] = df['close'].rolling(200, min_periods=1).mean()
    df['ma_20'] = df['close'].rolling(20, min_periods=1).mean()

    # Volume Average
    df['vol_avg'] = df['volume'].rolling(200, min_periods=1).mean()

    # 52-week High
    df['high_52w'] = df['close'].rolling(252, min_periods=1).max()

    return df


def fetch_single_symbol(symbol: str, period: str = "5y") -> tuple:
    """Fetch a single symbol's data. Returns (symbol, dataframe) or (symbol, None)."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, auto_adjust=True)

        if df.empty or len(df) < 50:
            return (symbol, None)

        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df = compute_indicators(df)
        return (symbol, df)

    except Exception as e:
        return (symbol, None)


def fetch_batch(symbols: list, period: str = "5y") -> dict:
    """Fetch a batch of symbols using yfinance download."""
    batch_data = {}

    try:
        # Use single-threaded download to avoid dictionary iteration issues
        df = yf.download(
            symbols,
            period=period,
            auto_adjust=True,
            threads=False,  # Single-threaded to avoid race conditions
            progress=False
        )

        if df.empty:
            return batch_data

        # Handle single vs multiple symbols
        if len(symbols) == 1:
            symbol = symbols[0]
            df.columns = [c.lower() for c in df.columns]
            if len(df) >= 50:
                df = compute_indicators(df)
                batch_data[symbol] = df
        else:
            # Multi-symbol download returns multi-index columns
            for symbol in symbols:
                try:
                    if 'Close' in df.columns.get_level_values(0):
                        if symbol in df['Close'].columns:
                            symbol_df = pd.DataFrame({
                                'open': df['Open'][symbol] if symbol in df.get('Open', pd.DataFrame()).columns else np.nan,
                                'high': df['High'][symbol] if symbol in df.get('High', pd.DataFrame()).columns else np.nan,
                                'low': df['Low'][symbol] if symbol in df.get('Low', pd.DataFrame()).columns else np.nan,
                                'close': df['Close'][symbol],
                                'volume': df['Volume'][symbol] if symbol in df.get('Volume', pd.DataFrame()).columns else 0
                            }).dropna(subset=['close'])

                            if len(symbol_df) >= 50:
                                symbol_df = compute_indicators(symbol_df)
                                batch_data[symbol] = symbol_df
                except Exception:
                    pass

    except Exception as e:
        # If batch fails, try individual symbols
        for symbol in symbols:
            sym, df = fetch_single_symbol(symbol, period)
            if df is not None:
                batch_data[sym] = df

    return batch_data


def fetch_all_data(symbols: list) -> dict:
    """Fetch all symbols - required ones first, then batches for the rest."""
    global data_cache

    # Phase 1: Fetch required symbols individually (guaranteed inclusion)
    print(f"\n=== Phase 1: Fetching {len(REQUIRED_SYMBOLS)} required symbols ===")
    required_success = 0
    for i, symbol in enumerate(REQUIRED_SYMBOLS):
        sym, df = fetch_single_symbol(symbol, PERIOD)
        if df is not None:
            with cache_lock:
                data_cache[sym] = df
            required_success += 1
            print(f"  [{i+1}/{len(REQUIRED_SYMBOLS)}] {symbol}: {len(df)} rows")
        else:
            print(f"  [{i+1}/{len(REQUIRED_SYMBOLS)}] {symbol}: FAILED")

    print(f"Required symbols: {required_success}/{len(REQUIRED_SYMBOLS)} succeeded")

    # Phase 2: Fetch remaining symbols in batches
    remaining = [s for s in symbols if s not in REQUIRED_SYMBOLS]
    batches = [remaining[i:i+BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    total_batches = len(batches)

    print(f"\n=== Phase 2: Fetching {len(remaining)} remaining symbols in {total_batches} batches ===")

    start_time = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {executor.submit(fetch_batch, batch, PERIOD): i for i, batch in enumerate(batches)}

        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            completed += 1

            try:
                batch_data = future.result()
                with cache_lock:
                    data_cache.update(batch_data)

                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_batches - completed) / rate if rate > 0 else 0

                print(f"Batch {completed}/{total_batches}: +{len(batch_data)} symbols | Total: {len(data_cache)} | ETA: {eta/60:.1f}m", flush=True)

            except Exception as e:
                print(f"Batch {batch_idx} failed: {e}")

    return data_cache


def save_local(data_cache: dict, filepath: Path) -> tuple:
    """Save data to local gzipped pickle file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    pkl_bytes = pickle.dumps(data_cache)
    gzipped = gzip.compress(pkl_bytes, compresslevel=6)

    with open(filepath, 'wb') as f:
        f.write(gzipped)

    size_mb = len(gzipped) / 1024 / 1024
    print(f"\nSaved to {filepath}")
    print(f"Size: {size_mb:.1f} MB ({len(gzipped):,} bytes)")
    return size_mb, len(gzipped)


def verify_data(data_cache: dict):
    """Verify the data looks correct."""
    print("\n" + "=" * 60)
    print("Data Verification")
    print("=" * 60)

    if not data_cache:
        print("ERROR: No data fetched!")
        return False

    # Check required symbols
    all_good = True
    for sym in REQUIRED_SYMBOLS[:15]:  # Check first 15
        if sym in data_cache:
            df = data_cache[sym]
            print(f"  {sym}: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} rows)")
        else:
            print(f"  {sym}: MISSING!")
            all_good = False

    # Stats
    lengths = [len(df) for df in data_cache.values()]
    print(f"\nTotal symbols: {len(data_cache)}")
    if lengths:
        print(f"Avg rows: {np.mean(lengths):.0f}")
        print(f"Min rows: {min(lengths)}, Max rows: {max(lengths)}")

    # Count symbols by data length
    bins = [0, 250, 500, 750, 1000, 1250, 1500]
    for i in range(len(bins) - 1):
        count = sum(1 for df in data_cache.values() if bins[i] <= len(df) < bins[i+1])
        print(f"  {bins[i]}-{bins[i+1]} rows: {count} symbols")
    count_1500_plus = sum(1 for df in data_cache.values() if len(df) >= 1500)
    print(f"  1500+ rows: {count_1500_plus} symbols")

    return all_good


def main():
    print("=" * 60)
    print("Fetching 5 Years of Historical Price Data - FULL UNIVERSE")
    print("=" * 60)
    print(f"Period: {PERIOD}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Required symbols: {len(REQUIRED_SYMBOLS)}")
    print()

    # Get symbols
    symbols = get_all_tickers()

    # Fetch data
    start_time = datetime.now()
    result_cache = fetch_all_data(symbols)
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nFetch completed in {elapsed/60:.1f} minutes")

    # Verify
    success = verify_data(result_cache)

    # Save locally
    print("\nSaving locally...")
    size_mb, size_bytes = save_local(result_cache, OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Local file: {OUTPUT_FILE}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Symbols: {len(result_cache)}")

    # Check required symbols
    required_found = sum(1 for s in REQUIRED_SYMBOLS if s in result_cache)
    print(f"Required symbols found: {required_found}/{len(REQUIRED_SYMBOLS)}")

    if not success:
        print("\nWARNING: Some required symbols are missing!")

    print()
    print("To upload to S3, run:")
    print(f"  aws s3 cp {OUTPUT_FILE} s3://YOUR-BUCKET/prices/all_data.pkl.gz")
    print()
    print("Or rename and use in place of existing file:")
    print(f"  mv {OUTPUT_FILE} {OUTPUT_FILE.parent / 'all_data.pkl.gz'}")


if __name__ == "__main__":
    main()

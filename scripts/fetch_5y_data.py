#!/usr/bin/env python3
"""
Fetch 5 years of historical price data and upload to S3.

This script:
1. Gets the full stock universe (NASDAQ-100 + S&P 500 selections)
2. Fetches 5 years of OHLCV data from yfinance
3. Computes technical indicators (DWAP, MAs, etc.)
4. Saves to a gzipped pickle file
5. Uploads to S3

Run locally with AWS credentials configured.
"""

import os
import sys
import pickle
import gzip
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
import boto3

# Configuration
S3_BUCKET = "rigacap-prod-prices"
OUTPUT_FILE = Path(__file__).parent.parent / "backend" / "data" / "prices" / "all_data_5y.pkl.gz"
BATCH_SIZE = 50  # Download in batches
PERIOD = "5y"

# Required symbols that must be included
REQUIRED_SYMBOLS = ['SPY', '^VIX']

# Top liquid stocks - NASDAQ-100 + major S&P 500
STOCK_UNIVERSE = [
    # NASDAQ-100
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'COST',
    'ASML', 'PEP', 'CSCO', 'NFLX', 'TMUS', 'AMD', 'ADBE', 'TXN', 'QCOM', 'CMCSA',
    'INTC', 'HON', 'INTU', 'AMGN', 'AMAT', 'ISRG', 'BKNG', 'SBUX', 'VRTX', 'LRCX',
    'ADI', 'MDLZ', 'GILD', 'ADP', 'REGN', 'PANW', 'KLAC', 'SNPS', 'CDNS', 'MU',
    'MELI', 'CSX', 'MAR', 'PYPL', 'CRWD', 'ORLY', 'CHTR', 'ABNB', 'MNST', 'MRVL',
    'FTNT', 'NXPI', 'CTAS', 'WDAY', 'DXCM', 'PCAR', 'ODFL', 'KDP', 'KHC', 'PAYX',
    'AEP', 'ROST', 'CPRT', 'LULU', 'GEHC', 'EXC', 'MCHP', 'AZN', 'FAST', 'IDXX',
    'CTSH', 'VRSK', 'TTD', 'CEG', 'ON', 'CSGP', 'EA', 'FANG', 'BKR', 'DDOG',
    'XEL', 'ANSS', 'TEAM', 'ZS', 'ILMN', 'BIIB', 'WBD', 'DLTR', 'EBAY', 'GFS',
    'WBA', 'ENPH', 'SIRI', 'JD', 'RIVN', 'LCID',
    # Major S&P 500 additions
    'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'XOM', 'CVX',
    'BAC', 'LLY', 'ABBV', 'MRK', 'PFE', 'KO', 'TMO', 'ORCL', 'ACN', 'MCD',
    'DIS', 'ABT', 'DHR', 'NKE', 'VZ', 'CRM', 'PM', 'WFC', 'NEE', 'RTX',
    'UPS', 'IBM', 'LOW', 'CAT', 'GE', 'BA', 'SPGI', 'SCHW', 'DE', 'GS',
    'BLK', 'ANET', 'MS', 'PLD', 'C', 'NOW', 'MDT', 'BMY', 'UBER', 'T',
    'AMT', 'AMZN', 'SYK', 'TJX', 'CB', 'MMC', 'AXP', 'MO', 'VICI', 'SO',
    'LMT', 'ZTS', 'PGR', 'SLB', 'EOG', 'DUK', 'CI', 'CL', 'EQIX', 'BDX',
    'CME', 'APD', 'ITW', 'SNOW', 'ICE', 'MPC', 'WM', 'PSX', 'FDX', 'EMR',
    'SHW', 'NOC', 'SMCI', 'PNC', 'USB', 'AON', 'WELL', 'TGT', 'CCI', 'FCX',
    'MCK', 'TFC', 'PLTR', 'COF', 'NSC', 'GM', 'F', 'CARR', 'OKE', 'SPY'
]


def get_stock_universe():
    """Get the stock universe - hardcoded for reliability"""
    # Remove duplicates and add required symbols
    symbols = list(set(STOCK_UNIVERSE))
    for sym in REQUIRED_SYMBOLS:
        if sym not in symbols:
            symbols.append(sym)

    print(f"Universe: {len(symbols)} symbols")
    return sorted(symbols)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators on a price DataFrame"""
    if len(df) < 50:
        return df

    # Ensure columns are lowercase
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


def fetch_batch(symbols: list, period: str = "5y") -> dict:
    """Fetch a batch of symbols from yfinance"""
    data_cache = {}

    try:
        # Use yfinance download for batch efficiency
        df = yf.download(
            symbols,
            period=period,
            auto_adjust=True,
            threads=True,
            progress=False
        )

        if df.empty:
            return data_cache

        # Handle single vs multiple symbols
        if len(symbols) == 1:
            symbol = symbols[0]
            df.columns = [c.lower() for c in df.columns]
            df = compute_indicators(df)
            data_cache[symbol] = df
        else:
            # Multi-symbol download returns multi-index columns
            for symbol in symbols:
                try:
                    if symbol in df['Close'].columns:
                        symbol_df = pd.DataFrame({
                            'open': df['Open'][symbol],
                            'high': df['High'][symbol],
                            'low': df['Low'][symbol],
                            'close': df['Close'][symbol],
                            'volume': df['Volume'][symbol]
                        }).dropna()

                        if len(symbol_df) >= 50:
                            symbol_df = compute_indicators(symbol_df)
                            data_cache[symbol] = symbol_df
                except Exception as e:
                    print(f"  Error processing {symbol}: {e}")

    except Exception as e:
        print(f"  Batch download failed: {e}")

    return data_cache


def fetch_all_data(symbols: list) -> dict:
    """Fetch all symbols in batches"""
    all_data = {}
    total = len(symbols)

    for i in range(0, total, BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"Batch {batch_num}/{total_batches}: Fetching {len(batch)} symbols...")

        batch_data = fetch_batch(batch, PERIOD)
        all_data.update(batch_data)

        print(f"  Got {len(batch_data)} symbols, total: {len(all_data)}")

    return all_data


def save_local(data_cache: dict, filepath: Path):
    """Save data to local gzipped pickle file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    pkl_bytes = pickle.dumps(data_cache)
    gzipped = gzip.compress(pkl_bytes)

    with open(filepath, 'wb') as f:
        f.write(gzipped)

    size_mb = len(gzipped) / 1024 / 1024
    print(f"Saved to {filepath} ({size_mb:.1f} MB)")
    return size_mb


def upload_to_s3(filepath: Path, bucket: str, key: str = "prices/all_data.pkl.gz"):
    """Upload the pickle file to S3"""
    s3 = boto3.client('s3')

    with open(filepath, 'rb') as f:
        data = f.read()

    print(f"Uploading {len(data) / 1024 / 1024:.1f} MB to s3://{bucket}/{key}...")

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType='application/octet-stream'
    )

    print(f"Uploaded to s3://{bucket}/{key}")


def verify_data(data_cache: dict):
    """Verify the data looks correct"""
    print("\n=== Data Verification ===")

    # Check key symbols
    for sym in ['SPY', 'AAPL', 'MSFT', 'GOOGL']:
        if sym in data_cache:
            df = data_cache[sym]
            print(f"{sym}: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} rows)")
        else:
            print(f"{sym}: MISSING!")

    # Stats
    lengths = [len(df) for df in data_cache.values()]
    print(f"\nTotal symbols: {len(data_cache)}")
    print(f"Avg rows: {np.mean(lengths):.0f}")
    print(f"Min rows: {min(lengths)}, Max rows: {max(lengths)}")

    # Count symbols with 5y+ of data
    five_years = 252 * 5  # ~1260 trading days
    has_5y = sum(1 for df in data_cache.values() if len(df) >= five_years)
    print(f"Symbols with 5y+ data: {has_5y}")


def main():
    print("=" * 60)
    print("Fetching 5 Years of Historical Price Data")
    print("=" * 60)
    print(f"Period: {PERIOD}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"S3 Bucket: {S3_BUCKET}")
    print()

    # Get symbols
    symbols = get_stock_universe()

    # Fetch data
    print("\nFetching data from yfinance...")
    start_time = datetime.now()
    data_cache = fetch_all_data(symbols)
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nFetch completed in {elapsed:.0f} seconds")

    # Verify
    verify_data(data_cache)

    # Save locally
    print("\nSaving locally...")
    size_mb = save_local(data_cache, OUTPUT_FILE)

    # Ask before uploading
    print(f"\nReady to upload {size_mb:.1f} MB to S3.")
    response = input("Upload to S3? [y/N]: ").strip().lower()

    if response == 'y':
        upload_to_s3(OUTPUT_FILE, S3_BUCKET)
        print("\nDone! Data is now available on S3.")
    else:
        print(f"\nSkipped upload. Local file: {OUTPUT_FILE}")
        print(f"To upload manually: aws s3 cp {OUTPUT_FILE} s3://{S3_BUCKET}/prices/all_data.pkl.gz")


if __name__ == "__main__":
    main()

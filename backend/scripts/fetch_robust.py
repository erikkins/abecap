#!/usr/bin/env python3
"""
Robust data fetch - smaller batches, retries, 2-year history only.
Saves progress incrementally to avoid losing work.
"""

import asyncio
import sys
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import gzip

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Settings
BATCH_SIZE = 5  # Smaller batches = more reliable
DELAY = 2.0  # Longer delay between batches
MAX_RETRIES = 3
PERIOD = "2y"  # 2 years only

async def main():
    from app.services.stock_universe import stock_universe_service

    print("=" * 60)
    print("ROBUST DATA FETCH (2-year history)")
    print("=" * 60)

    # Load universe
    print("\n📋 Loading universe...")
    symbols = await stock_universe_service.ensure_loaded()
    print(f"   Universe: {len(symbols)} symbols")

    # Track results
    all_data = {}
    failed = []

    # Process in small batches
    total = len(symbols)
    print(f"\n🔄 Fetching {total} symbols in batches of {BATCH_SIZE}...")
    print(f"   Period: {PERIOD}, Delay: {DELAY}s between batches")
    print()

    for i in range(0, total, BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        success = False
        for retry in range(MAX_RETRIES):
            try:
                data = yf.download(
                    batch,
                    period=PERIOD,
                    progress=False,
                    threads=True,
                    timeout=30
                )

                # Process each symbol
                for symbol in batch:
                    try:
                        if len(batch) == 1:
                            df = data.copy()
                        else:
                            if symbol not in data.columns.get_level_values(1):
                                continue
                            df = data.xs(symbol, level=1, axis=1).copy()

                        if len(df) >= 50:  # Need at least 50 days
                            df = df.dropna()
                            df.columns = [c.lower() for c in df.columns]
                            all_data[symbol] = df
                    except Exception:
                        pass

                success = True
                break

            except Exception as e:
                if retry < MAX_RETRIES - 1:
                    time.sleep(DELAY * 2)
                else:
                    failed.extend(batch)

        # Progress update
        if batch_num % 20 == 0 or batch_num == total_batches:
            print(f"   [{batch_num}/{total_batches}] {len(all_data)} symbols fetched, {len(failed)} failed")

        if success:
            time.sleep(DELAY)

    print(f"\n✅ Fetch complete: {len(all_data)} symbols, {len(failed)} failed")

    if failed:
        print(f"   Failed symbols (sample): {failed[:20]}")

    # Save to consolidated file
    print("\n💾 Saving to consolidated file...")

    dfs = []
    for symbol, df in all_data.items():
        df_copy = df.reset_index()
        df_copy['symbol'] = symbol
        # Rename 'Date' to 'date' if needed
        if 'Date' in df_copy.columns:
            df_copy = df_copy.rename(columns={'Date': 'date'})
        dfs.append(df_copy)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"   Total rows: {len(combined)}")

    # Save locally
    local_path = 'data/prices/all_prices_2y.csv.gz'
    combined.to_csv(local_path, index=False, compression='gzip')
    size_mb = os.path.getsize(local_path) / (1024*1024)
    print(f"   Local file: {local_path} ({size_mb:.1f} MB)")

    # Upload to S3
    print("\n☁️  Uploading to S3...")
    import boto3
    s3 = boto3.client('s3')
    bucket = os.getenv('PRICE_DATA_BUCKET', 'rigacap-prod-price-data-149218244179')

    with open(local_path, 'rb') as f:
        s3.put_object(
            Bucket=bucket,
            Key='prices/all_prices.csv.gz',
            Body=f.read(),
            ContentType='application/gzip'
        )
    print(f"   ✅ Uploaded to s3://{bucket}/prices/all_prices.csv.gz")

    # Update metadata
    metadata = {
        "last_export": datetime.utcnow().isoformat(),
        "symbols_count": len(all_data),
        "period": PERIOD
    }
    import json
    s3.put_object(
        Bucket=bucket,
        Key='prices/_metadata.json',
        Body=json.dumps(metadata),
        ContentType='application/json'
    )
    print(f"   ✅ Metadata updated: {len(all_data)} symbols")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())

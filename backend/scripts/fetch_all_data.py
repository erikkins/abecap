#!/usr/bin/env python3
"""
Fetch all price data locally and upload to S3.

This script fetches data for all symbols in the universe that aren't already cached,
then uploads the complete dataset to S3.

Run from backend directory:
    source venv/bin/activate && python scripts/fetch_all_data.py
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def main():
    from app.services.scanner import scanner_service
    from app.services.stock_universe import stock_universe_service
    from app.services.data_export import data_export_service

    print("=" * 60)
    print("FULL DATA FETCH SCRIPT")
    print("=" * 60)

    # Step 1: Load universe
    print("\n📋 Step 1: Loading universe...")
    symbols = await stock_universe_service.ensure_loaded()
    scanner_service.universe = symbols
    print(f"   Universe: {len(symbols)} symbols")

    # Step 2: Try to load existing data from S3
    print("\n📥 Step 2: Loading existing data from S3...")
    try:
        data = data_export_service.import_all()
        if data:
            scanner_service.data_cache = data
            print(f"   Loaded {len(scanner_service.data_cache)} symbols from S3")
        else:
            print(f"   No existing data found")
    except Exception as e:
        print(f"   Failed to load from S3: {e}")

    # Step 3: Find missing symbols
    cached = set(scanner_service.data_cache.keys())
    universe = set(symbols)
    missing = list(universe - cached)
    print(f"\n📊 Step 3: Status")
    print(f"   Cached: {len(cached)}")
    print(f"   Missing: {len(missing)}")

    if not missing:
        print("\n✅ All symbols already cached!")
        return

    # Step 4: Fetch missing data in batches
    print(f"\n🔄 Step 4: Fetching {len(missing)} missing symbols...")
    print("   This will take a while. Progress will be shown every 100 symbols.")

    BATCH_SIZE = 10
    DELAY = 1.5

    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(missing) + BATCH_SIZE - 1) // BATCH_SIZE

        try:
            import yfinance as yf
            data = yf.download(batch, period="5y", progress=False, threads=True, timeout=30)

            # Process each symbol
            for symbol in batch:
                try:
                    if len(batch) == 1:
                        df = data.copy()
                    else:
                        if symbol not in data.columns.get_level_values(1):
                            continue
                        df = data.xs(symbol, level=1, axis=1).copy()

                    if len(df) > 0:
                        df = df.dropna()
                        df.columns = [c.lower() for c in df.columns]
                        scanner_service.data_cache[symbol] = df
                except Exception as e:
                    pass

            # Progress update every 100 symbols
            if (i + BATCH_SIZE) % 100 == 0 or i + BATCH_SIZE >= len(missing):
                print(f"   Progress: {min(i + BATCH_SIZE, len(missing))}/{len(missing)} "
                      f"({len(scanner_service.data_cache)} total cached)")

            await asyncio.sleep(DELAY)

        except Exception as e:
            print(f"   Batch {batch_num} error: {e}")
            await asyncio.sleep(DELAY * 2)

    print(f"\n✅ Fetch complete: {len(scanner_service.data_cache)} symbols cached")

    # Step 5: Save to S3
    print("\n💾 Step 5: Saving to S3...")
    export_result = data_export_service.export_consolidated(scanner_service.data_cache)
    if export_result.get("success"):
        print(f"   ✅ Saved {export_result.get('count')} symbols to S3")
        print(f"   Size: {export_result.get('total_size_mb', 0):.1f} MB")
    else:
        print(f"   ❌ Export failed: {export_result.get('message')}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())

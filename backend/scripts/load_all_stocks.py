#!/usr/bin/env python3
"""
Bulk load all stocks and save to S3.

Run this locally (not in Lambda) to build up the full dataset.
It will load stocks in batches and save progress after each batch.

Usage:
    python scripts/load_all_stocks.py

Environment:
    Set PRICE_DATA_BUCKET to your S3 bucket name.
    AWS credentials should be configured (aws configure).
"""

import asyncio
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set S3 bucket (simulating Lambda environment)
os.environ["PRICE_DATA_BUCKET"] = os.environ.get("PRICE_DATA_BUCKET", "rigacap-prod-price-data")

from app.services.scanner import scanner_service
from app.services.data_export import data_export_service


async def main():
    batch_size = 50
    save_every = 100  # Save to S3 every N stocks

    print("=" * 60)
    print("Stock Data Loader")
    print("=" * 60)

    # Load universe
    print("\n1. Loading stock universe...")
    await scanner_service.ensure_universe_loaded()
    print(f"   Universe: {len(scanner_service.universe)} symbols")

    # Load existing data from S3
    print("\n2. Loading existing data from S3...")
    try:
        cached_data = data_export_service.import_all()
        if cached_data:
            scanner_service.data_cache = cached_data
            print(f"   Loaded: {len(cached_data)} symbols from S3")
        else:
            print("   No existing data found")
    except Exception as e:
        print(f"   Error loading from S3: {e}")

    # Find missing symbols
    cached = set(scanner_service.data_cache.keys())
    all_symbols = set(scanner_service.universe)
    missing = list(all_symbols - cached)

    print(f"\n3. Status:")
    print(f"   Cached: {len(cached)}")
    print(f"   Missing: {len(missing)}")
    print(f"   Progress: {len(cached) / len(all_symbols) * 100:.1f}%")

    if not missing:
        print("\n✅ All stocks already loaded!")
        return

    # Load in batches
    print(f"\n4. Loading {len(missing)} missing stocks in batches of {batch_size}...")
    total_loaded = 0
    start_time = time.time()

    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(missing) + batch_size - 1) // batch_size

        print(f"\n   Batch {batch_num}/{total_batches}: {len(batch)} symbols...", end=" ", flush=True)

        try:
            await scanner_service.fetch_data(batch)
            newly_loaded = len([s for s in batch if s in scanner_service.data_cache])
            total_loaded += newly_loaded
            print(f"✓ {newly_loaded} loaded")

            # Save progress periodically
            if total_loaded % save_every < batch_size or i + batch_size >= len(missing):
                print(f"   Saving to S3...", end=" ", flush=True)
                result = data_export_service.export_consolidated(scanner_service.data_cache)
                if result.get("success"):
                    print(f"✓ {result.get('count')} symbols ({result.get('total_size_mb')} MB)")
                else:
                    print(f"✗ {result.get('message')}")

        except Exception as e:
            print(f"✗ Error: {e}")
            # Continue with next batch
            continue

        # Progress estimate
        elapsed = time.time() - start_time
        rate = total_loaded / elapsed if elapsed > 0 else 0
        remaining = len(missing) - (i + batch_size)
        eta = remaining / rate / 60 if rate > 0 else 0
        print(f"   Progress: {len(scanner_service.data_cache)}/{len(all_symbols)} ({len(scanner_service.data_cache) / len(all_symbols) * 100:.1f}%) | ETA: {eta:.0f} min")

    # Final save
    print("\n5. Final save to S3...")
    result = data_export_service.export_consolidated(scanner_service.data_cache)
    if result.get("success"):
        print(f"   ✅ Saved {result.get('count')} symbols ({result.get('total_size_mb')} MB)")
    else:
        print(f"   ✗ Error: {result.get('message')}")

    elapsed = time.time() - start_time
    print(f"\n✅ Complete! Loaded {total_loaded} new stocks in {elapsed / 60:.1f} minutes")
    print(f"   Total cached: {len(scanner_service.data_cache)} / {len(all_symbols)}")


if __name__ == "__main__":
    asyncio.run(main())

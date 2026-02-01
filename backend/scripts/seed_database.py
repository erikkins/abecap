#!/usr/bin/env python3
"""
Database Seeding Script

Seeds the database with historical price data from parquet files.
Useful for new deployments or restoring data.

Usage:
    python -m scripts.seed_database
    python -m scripts.seed_database --clear  # Clear existing data first
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import argparse
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.data_export import data_export_service
from app.services.scanner import scanner_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def seed_database(clear_first: bool = False):
    """
    Seed the database with data from parquet files
    """
    logger.info("Starting database seeding...")

    # Check for parquet files
    status = data_export_service.get_status()
    files_count = status.get('files_count', 0)

    if files_count == 0:
        logger.error("No parquet files found to seed from.")
        logger.info("First, export data using: POST /api/data/export")
        return False

    logger.info(f"Found {files_count} parquet files ({status.get('total_size_mb', 0):.1f} MB)")

    if clear_first:
        logger.warning("Clearing existing price data...")
        cleared = data_export_service.clear_all()
        logger.info(f"Cleared {cleared} files")

    # Import all parquet files
    logger.info("Importing parquet files into memory...")
    data_cache = data_export_service.import_all()

    if not data_cache:
        logger.error("No data loaded from parquet files")
        return False

    # Update scanner service cache
    scanner_service.data_cache = data_cache

    logger.info(f"Loaded {len(data_cache)} symbols into scanner cache")

    # Print some stats
    total_rows = sum(len(df) for df in data_cache.values())
    logger.info(f"Total data points: {total_rows:,}")

    # Sample of loaded symbols
    sample = list(data_cache.keys())[:10]
    logger.info(f"Sample symbols: {', '.join(sample)}")

    logger.info("Database seeding complete!")
    return True


async def run_initial_scan():
    """
    Run an initial scan after seeding to generate signals
    """
    logger.info("Running initial market scan...")

    signals = await scanner_service.scan(refresh_data=False)
    strong = [s for s in signals if s.is_strong]

    logger.info(f"Found {len(signals)} signals ({len(strong)} strong)")

    # Print top signals
    for sig in strong[:5]:
        logger.info(f"  {sig.symbol}: ${sig.price:.2f} (+{sig.pct_above_dwap:.1f}% DWAP)")


def main():
    parser = argparse.ArgumentParser(description='Seed database with historical price data')
    parser.add_argument('--clear', action='store_true', help='Clear existing data before seeding')
    parser.add_argument('--scan', action='store_true', help='Run initial scan after seeding')
    args = parser.parse_args()

    # Run async seeding
    success = asyncio.run(seed_database(clear_first=args.clear))

    if success and args.scan:
        asyncio.run(run_initial_scan())


if __name__ == "__main__":
    main()

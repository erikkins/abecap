"""
Data Export Service - Persist historical price data to S3 or local files

Exports scanner data cache to Parquet files for:
- Permanent storage of historical prices (never need to re-fetch)
- Fast loading on startup
- Database seeding for new deployments

In Lambda: Uses S3 bucket for persistent storage
Locally: Uses backend/data/prices directory
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging
import json
import os
import io
import gzip

logger = logging.getLogger(__name__)

# Check if running in Lambda with S3 bucket configured
S3_BUCKET = os.environ.get("PRICE_DATA_BUCKET")
IS_LAMBDA = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

# Local data directory (for development)
LOCAL_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prices"


class DataExportService:
    """
    Manages export and import of historical price data.
    Uses S3 in Lambda, local filesystem in development.
    """

    def __init__(self):
        self.last_export: Optional[datetime] = None
        self.exported_symbols: List[str] = []
        self._s3_client = None

        if not IS_LAMBDA:
            self._ensure_local_dir()

    def _get_s3_client(self):
        """Get boto3 S3 client (lazy initialization)"""
        if self._s3_client is None:
            import boto3
            self._s3_client = boto3.client('s3')
        return self._s3_client

    def _ensure_local_dir(self):
        """Create local data directory if it doesn't exist"""
        LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _use_s3(self) -> bool:
        """Check if we should use S3 storage"""
        return IS_LAMBDA and S3_BUCKET is not None

    def export_all(self, data_cache: Dict[str, pd.DataFrame]) -> Dict:
        """
        Export all cached data to Parquet files (S3 or local)

        Args:
            data_cache: Scanner's data cache (symbol -> DataFrame)

        Returns:
            Export summary with file count and size
        """
        if not data_cache:
            return {"success": False, "message": "No data to export", "count": 0}

        exported = 0
        total_size = 0
        failed = []

        for symbol, df in data_cache.items():
            try:
                if len(df) < 50:  # Skip if not enough data
                    continue

                # Reset index to include date as column
                df_export = df.reset_index()

                # Ensure date column is proper datetime
                if 'date' in df_export.columns:
                    df_export['date'] = pd.to_datetime(df_export['date'])

                if self._use_s3():
                    # Export to S3 as CSV (pyarrow not available in Lambda)
                    buffer = io.StringIO()
                    df_export.to_csv(buffer, index=False)
                    csv_bytes = buffer.getvalue().encode('utf-8')

                    s3 = self._get_s3_client()
                    s3.put_object(
                        Bucket=S3_BUCKET,
                        Key=f"prices/{symbol}.csv",
                        Body=csv_bytes,
                        ContentType='text/csv'
                    )
                    file_size = len(csv_bytes)
                else:
                    # Export to local file as parquet (pyarrow available locally)
                    filepath = LOCAL_DATA_DIR / f"{symbol}.parquet"
                    df_export.to_parquet(filepath, index=False, compression='snappy')
                    file_size = filepath.stat().st_size

                total_size += file_size
                exported += 1

            except Exception as e:
                logger.error(f"Failed to export {symbol}: {e}")
                failed.append(symbol)

        # Save metadata
        self.last_export = datetime.now()
        self.exported_symbols = list(data_cache.keys())
        self._save_metadata()

        storage_type = "S3" if self._use_s3() else "local"
        logger.info(f"Exported {exported} symbols to {storage_type} ({total_size / 1024 / 1024:.1f} MB)")

        return {
            "success": True,
            "count": exported,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "failed": failed,
            "storage": storage_type,
            "bucket": S3_BUCKET if self._use_s3() else None,
            "timestamp": self.last_export.isoformat()
        }

    def import_all(self) -> Dict[str, pd.DataFrame]:
        """
        Import all Parquet files into memory (from S3 or local)

        Returns:
            Dict mapping symbol to DataFrame
        """
        data_cache = {}

        if self._use_s3():
            data_cache = self._import_from_s3()
        else:
            data_cache = self._import_from_local()

        return data_cache

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators on a price DataFrame"""
        if len(df) < 50:
            return df

        # DWAP - Daily Weighted Average Price (200-day)
        pv = df['close'] * df['volume']
        df['dwap'] = pv.rolling(200, min_periods=50).sum() / df['volume'].rolling(200, min_periods=50).sum()

        # Moving Averages
        df['ma_50'] = df['close'].rolling(50, min_periods=1).mean()
        df['ma_200'] = df['close'].rolling(200, min_periods=1).mean()

        # Volume Average
        df['vol_avg'] = df['volume'].rolling(200, min_periods=1).mean()

        # 52-week High
        df['high_52w'] = df['close'].rolling(252, min_periods=1).max()

        return df

    def _import_from_s3(self) -> Dict[str, pd.DataFrame]:
        """Import price data from S3 - uses consolidated file for speed"""
        data_cache = {}

        try:
            s3 = self._get_s3_client()

            # Try consolidated file first (single download, much faster)
            try:
                logger.info("Loading consolidated price data from S3...")
                response = s3.get_object(Bucket=S3_BUCKET, Key='prices/all_prices.csv.gz')
                csv_bytes = gzip.decompress(response['Body'].read())
                csv_content = csv_bytes.decode('utf-8')
                buffer = io.StringIO(csv_content)

                # Read the consolidated CSV with symbol column
                df_all = pd.read_csv(buffer)
                df_all['date'] = pd.to_datetime(df_all['date'])

                # Split by symbol (indicators are computed lazily by scanner)
                for symbol, group in df_all.groupby('symbol'):
                    df = group.drop(columns=['symbol']).set_index('date').sort_index()
                    data_cache[symbol] = df

                logger.info(f"Loaded {len(data_cache)} symbols from consolidated S3 file")
                return data_cache

            except s3.exceptions.NoSuchKey:
                logger.info("No consolidated file found, falling back to individual files")
            except Exception as e:
                logger.warning(f"Consolidated load failed ({e}), falling back to individual files")

            # Fallback: load individual CSV files
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=S3_BUCKET, Prefix='prices/')

            csv_keys = []
            for page in pages:
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.csv') and not obj['Key'].endswith('all_prices.csv.gz'):
                        csv_keys.append(obj['Key'])

            logger.info(f"Found {len(csv_keys)} CSV files in S3")

            for key in csv_keys:
                try:
                    symbol = key.split('/')[-1].replace('.csv', '')

                    response = s3.get_object(Bucket=S3_BUCKET, Key=key)
                    csv_content = response['Body'].read().decode('utf-8')
                    buffer = io.StringIO(csv_content)

                    df = pd.read_csv(buffer)

                    # Set date as index
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date').sort_index()

                    data_cache[symbol] = df

                except Exception as e:
                    logger.error(f"Failed to import {key} from S3: {e}")

            logger.info(f"Imported {len(data_cache)} symbols from S3")

        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")

        return data_cache

    def export_consolidated(self, data_cache: Dict[str, pd.DataFrame]) -> Dict:
        """
        Export all cached data to a single consolidated gzipped CSV file.
        Much faster to load than individual files (single S3 GET vs 100+).
        """
        if not data_cache:
            return {"success": False, "message": "No data to export", "count": 0}

        try:
            # Combine all DataFrames into one with symbol column
            dfs = []
            for symbol, df in data_cache.items():
                if len(df) < 50:
                    continue
                df_copy = df.reset_index()
                df_copy['symbol'] = symbol
                dfs.append(df_copy)

            if not dfs:
                return {"success": False, "message": "No valid data to export", "count": 0}

            combined = pd.concat(dfs, ignore_index=True)

            # Ensure date column is proper datetime
            if 'date' in combined.columns:
                combined['date'] = pd.to_datetime(combined['date'])

            if self._use_s3():
                # Export to S3 as gzipped CSV
                buffer = io.StringIO()
                combined.to_csv(buffer, index=False)
                csv_bytes = buffer.getvalue().encode('utf-8')
                gzipped = gzip.compress(csv_bytes)

                s3 = self._get_s3_client()
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key='prices/all_prices.csv.gz',
                    Body=gzipped,
                    ContentType='application/gzip'
                )
                file_size = len(gzipped)
                storage_type = "S3"
            else:
                # Export to local file
                filepath = LOCAL_DATA_DIR / "all_prices.csv.gz"
                combined.to_csv(filepath, index=False, compression='gzip')
                file_size = filepath.stat().st_size
                storage_type = "local"

            # Save metadata
            self.last_export = datetime.now()
            self.exported_symbols = list(data_cache.keys())
            self._save_metadata()

            num_symbols = combined['symbol'].nunique()
            logger.info(f"Exported consolidated file with {num_symbols} symbols ({file_size / 1024 / 1024:.1f} MB)")

            return {
                "success": True,
                "count": num_symbols,
                "total_size_mb": round(file_size / 1024 / 1024, 2),
                "storage": storage_type,
                "bucket": S3_BUCKET if self._use_s3() else None,
                "timestamp": self.last_export.isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to export consolidated data: {e}")
            return {"success": False, "message": str(e), "count": 0}

    def _import_from_local(self) -> Dict[str, pd.DataFrame]:
        """Import parquet files from local filesystem"""
        data_cache = {}

        if not LOCAL_DATA_DIR.exists():
            logger.info("No local data directory found")
            return data_cache

        parquet_files = list(LOCAL_DATA_DIR.glob("*.parquet"))
        logger.info(f"Found {len(parquet_files)} parquet files locally")

        for filepath in parquet_files:
            try:
                symbol = filepath.stem  # filename without extension

                df = pd.read_parquet(filepath)

                # Set date as index
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date').sort_index()

                # Compute indicators if not present
                if 'dwap' not in df.columns:
                    df = self._compute_indicators(df)

                data_cache[symbol] = df

            except Exception as e:
                logger.error(f"Failed to import {filepath.name}: {e}")

        logger.info(f"Imported {len(data_cache)} symbols from local files (with indicators)")
        return data_cache

    def get_last_date(self, symbol: str) -> Optional[datetime]:
        """
        Get the last date we have data for a symbol

        Used to determine what new data to fetch from yfinance.
        """
        try:
            if self._use_s3():
                s3 = self._get_s3_client()
                response = s3.get_object(Bucket=S3_BUCKET, Key=f"prices/{symbol}.csv")
                csv_content = response['Body'].read().decode('utf-8')
                df = pd.read_csv(io.StringIO(csv_content))
            else:
                filepath = LOCAL_DATA_DIR / f"{symbol}.parquet"
                if not filepath.exists():
                    return None
                df = pd.read_parquet(filepath)

            if 'date' in df.columns:
                return pd.to_datetime(df['date'].max())
            return None

        except Exception:
            return None

    def get_symbols_with_data(self) -> List[str]:
        """Get list of symbols that have saved data"""
        if self._use_s3():
            try:
                s3 = self._get_s3_client()
                paginator = s3.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=S3_BUCKET, Prefix='prices/')

                symbols = []
                for page in pages:
                    for obj in page.get('Contents', []):
                        if obj['Key'].endswith('.csv'):
                            symbol = obj['Key'].split('/')[-1].replace('.csv', '')
                            symbols.append(symbol)
                return symbols
            except Exception as e:
                logger.error(f"Failed to list S3 symbols: {e}")
                return []
        else:
            if not LOCAL_DATA_DIR.exists():
                return []
            return [f.stem for f in LOCAL_DATA_DIR.glob("*.parquet")]

    def get_status(self) -> Dict:
        """Get export status and statistics"""
        self._load_metadata()

        symbols = self.get_symbols_with_data()

        if self._use_s3():
            return {
                "storage": "s3",
                "bucket": S3_BUCKET,
                "files_count": len(symbols),
                "last_export": self.last_export.isoformat() if self.last_export else None,
                "symbols": symbols[:50]  # First 50 for display
            }
        else:
            parquet_files = list(LOCAL_DATA_DIR.glob("*.parquet")) if LOCAL_DATA_DIR.exists() else []
            total_size = sum(f.stat().st_size for f in parquet_files)

            return {
                "storage": "local",
                "data_dir": str(LOCAL_DATA_DIR),
                "files_count": len(parquet_files),
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "last_export": self.last_export.isoformat() if self.last_export else None,
                "symbols": symbols[:50]
            }

    def _save_metadata(self):
        """Save export metadata"""
        metadata = {
            "last_export": self.last_export.isoformat() if self.last_export else None,
            "symbols_count": len(self.exported_symbols),
            "exported_at": datetime.now().isoformat()
        }

        if self._use_s3():
            try:
                s3 = self._get_s3_client()
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key="prices/_metadata.json",
                    Body=json.dumps(metadata),
                    ContentType='application/json'
                )
            except Exception as e:
                logger.error(f"Failed to save metadata to S3: {e}")
        else:
            try:
                metadata_file = LOCAL_DATA_DIR / "_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f)
            except Exception as e:
                logger.error(f"Failed to save metadata locally: {e}")

    def _load_metadata(self):
        """Load export metadata"""
        try:
            if self._use_s3():
                s3 = self._get_s3_client()
                response = s3.get_object(Bucket=S3_BUCKET, Key="prices/_metadata.json")
                metadata = json.loads(response['Body'].read().decode('utf-8'))
            else:
                metadata_file = LOCAL_DATA_DIR / "_metadata.json"
                if not metadata_file.exists():
                    return
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

            if metadata.get("last_export"):
                self.last_export = datetime.fromisoformat(metadata["last_export"])
        except Exception:
            pass

    def delete_symbol(self, symbol: str) -> bool:
        """Delete saved data for a symbol"""
        try:
            if self._use_s3():
                s3 = self._get_s3_client()
                s3.delete_object(Bucket=S3_BUCKET, Key=f"prices/{symbol}.csv")
                return True
            else:
                filepath = LOCAL_DATA_DIR / f"{symbol}.parquet"
                if filepath.exists():
                    filepath.unlink()
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete {symbol}: {e}")
            return False

    def clear_all(self) -> int:
        """Delete all saved price data (use with caution!)"""
        count = 0

        if self._use_s3():
            try:
                s3 = self._get_s3_client()
                paginator = s3.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=S3_BUCKET, Prefix='prices/')

                for page in pages:
                    for obj in page.get('Contents', []):
                        s3.delete_object(Bucket=S3_BUCKET, Key=obj['Key'])
                        count += 1
            except Exception as e:
                logger.error(f"Failed to clear S3 data: {e}")
        else:
            for filepath in LOCAL_DATA_DIR.glob("*.parquet"):
                filepath.unlink()
                count += 1
            metadata_file = LOCAL_DATA_DIR / "_metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()

        return count


    def export_signals_json(self, signals: list) -> Dict:
        """
        Export signals to a static JSON file on S3 for CDN delivery.

        This file is publicly accessible and cached by CloudFront,
        so the frontend can load signals instantly without API calls.

        Args:
            signals: List of SignalData objects from scanner

        Returns:
            Export result with URL
        """
        if not signals:
            return {"success": False, "message": "No signals to export", "count": 0}

        try:
            # Convert signals to JSON-serializable format
            signals_data = {
                "signals": [s.to_dict() if hasattr(s, 'to_dict') else s for s in signals],
                "generated_at": datetime.now().isoformat(),
                "count": len(signals)
            }

            json_content = json.dumps(signals_data, default=str)

            if self._use_s3():
                s3 = self._get_s3_client()

                # Upload to signals/ prefix (separate from price data)
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key='signals/latest.json',
                    Body=json_content.encode('utf-8'),
                    ContentType='application/json',
                    CacheControl='public, max-age=300'  # 5 min cache
                )

                # Also save timestamped version for history
                timestamp = datetime.now().strftime('%Y-%m-%d')
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key=f'signals/{timestamp}.json',
                    Body=json_content.encode('utf-8'),
                    ContentType='application/json'
                )

                logger.info(f"Exported {len(signals)} signals to S3")

                return {
                    "success": True,
                    "count": len(signals),
                    "bucket": S3_BUCKET,
                    "key": "signals/latest.json",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Local development - save to data directory
                signals_dir = LOCAL_DATA_DIR.parent / "signals"
                signals_dir.mkdir(parents=True, exist_ok=True)

                filepath = signals_dir / "latest.json"
                with open(filepath, 'w') as f:
                    f.write(json_content)

                logger.info(f"Exported {len(signals)} signals to {filepath}")

                return {
                    "success": True,
                    "count": len(signals),
                    "path": str(filepath),
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to export signals: {e}")
            return {"success": False, "message": str(e), "count": 0}

    def get_signals_url(self) -> str:
        """
        Get the public URL for the latest signals JSON.

        In production, this should be the CloudFront URL.
        """
        if self._use_s3():
            # Return S3 URL - in production this should be the CloudFront URL
            return f"https://{S3_BUCKET}.s3.amazonaws.com/signals/latest.json"
        else:
            return "/api/signals/latest.json"


# Singleton instance
data_export_service = DataExportService()

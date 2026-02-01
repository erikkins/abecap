"""AWS Lambda handler for RigaCap API."""
import asyncio
import logging
from mangum import Mangum
from main import app
from app.core.database import init_db

logger = logging.getLogger(__name__)

# Track initialization state
_initialized = False

async def _cold_start_init():
    """
    Initialize everything on Lambda cold start:
    1. Database tables
    2. Full stock universe (from S3 cache)
    3. Price data (from S3 cache)
    """
    global _initialized
    if _initialized:
        return

    print("🚀 Lambda cold start initialization...")

    # 1. Initialize database tables
    try:
        await init_db()
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️ DB init error (continuing): {e}")

    # 2. Load full stock universe from S3 cache
    try:
        from app.services.scanner import scanner_service
        await scanner_service.ensure_universe_loaded()
        print(f"✅ Universe loaded: {len(scanner_service.universe)} symbols")
    except Exception as e:
        print(f"⚠️ Universe load error (continuing): {e}")

    # 3. Load price data from S3 cache
    try:
        from app.services.data_export import data_export_service
        cached_data = data_export_service.import_all()
        if cached_data:
            from app.services.scanner import scanner_service
            scanner_service.data_cache = cached_data
            print(f"✅ Price data loaded: {len(cached_data)} symbols from S3")
        else:
            print("💡 No cached price data found in S3")
    except Exception as e:
        print(f"⚠️ Price data load error (continuing): {e}")

    _initialized = True
    print("🎯 Cold start initialization complete")

# Run initialization on module load (cold start)
asyncio.get_event_loop().run_until_complete(_cold_start_init())

# Mangum adapter wraps FastAPI app for Lambda
_mangum_handler = Mangum(app, lifespan="off")


def handler(event, context):
    """
    Lambda handler that supports:
    1. Warmer events (from EventBridge scheduled warmer)
    2. API Gateway HTTP API events (via Mangum)
    """
    # Handle warmer events - just return success to keep Lambda warm
    if event.get("warmer"):
        print("🔥 Warmer ping received - Lambda is warm")
        return {
            "statusCode": 200,
            "body": '{"status": "warm"}'
        }

    # For API Gateway events, use Mangum
    return _mangum_handler(event, context)

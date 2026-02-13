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


async def _run_walk_forward_job(job_config: dict):
    """Run walk-forward simulation job asynchronously."""
    import json
    from datetime import datetime
    from app.core.database import async_session
    from app.services.walk_forward_service import walk_forward_service
    from sqlalchemy import select
    from app.core.database import WalkForwardSimulation

    job_id = job_config["job_id"]
    print(f"[ASYNC-WF] Starting walk-forward job {job_id}")

    async with async_session() as db:
        try:
            # Update status to running
            result = await db.execute(
                select(WalkForwardSimulation).where(WalkForwardSimulation.id == job_id)
            )
            job = result.scalar_one_or_none()
            if job:
                job.status = "running"
                await db.commit()

            # Run the simulation
            start = datetime.strptime(job_config["start_date"], "%Y-%m-%d")
            end = datetime.strptime(job_config["end_date"], "%Y-%m-%d")

            sim_result = await walk_forward_service.run_walk_forward_simulation(
                db=db,
                start_date=start,
                end_date=end,
                reoptimization_frequency=job_config["frequency"],
                min_score_diff=job_config["min_score_diff"],
                enable_ai_optimization=job_config["enable_ai"],
                max_symbols=job_config["max_symbols"],
                existing_job_id=job_id
            )

            print(f"[ASYNC-WF] Job {job_id} completed: return={sim_result.total_return_pct}%")
            return {"status": "completed", "job_id": job_id}

        except Exception as e:
            import traceback
            print(f"[ASYNC-WF] Job {job_id} failed: {e}")
            print(traceback.format_exc())

            # Update job status to failed
            try:
                result = await db.execute(
                    select(WalkForwardSimulation).where(WalkForwardSimulation.id == job_id)
                )
                job = result.scalar_one_or_none()
                if job:
                    job.status = "failed"
                    job.switch_history_json = json.dumps({"error": str(e)})
                    await db.commit()
            except Exception:
                pass

            return {"status": "failed", "job_id": job_id, "error": str(e)}


async def _export_dashboard_cache():
    """Export pre-computed dashboard data to S3."""
    from app.core.database import async_session
    from app.api.signals import compute_shared_dashboard_data
    from app.services.data_export import data_export_service

    try:
        async with async_session() as db:
            data = await compute_shared_dashboard_data(db)
            result = data_export_service.export_dashboard_json(data)
            print(f"📦 Dashboard cache export: {result}")
            return {"status": "success", **result}
    except Exception as e:
        import traceback
        print(f"❌ Dashboard cache export failed: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


def handler(event, context):
    """
    Lambda handler that supports:
    1. Warmer events (from EventBridge scheduled warmer)
    2. Walk-forward async jobs (from async Lambda invocation)
    3. API Gateway HTTP API events (via Mangum)
    """
    # Handle warmer events - just return success to keep Lambda warm
    if event.get("warmer"):
        print("🔥 Warmer ping received - Lambda is warm")
        return {
            "statusCode": 200,
            "body": '{"status": "warm"}'
        }

    # Handle dashboard cache export
    if event.get("export_dashboard_cache"):
        print("📦 Dashboard cache export requested")
        result = asyncio.get_event_loop().run_until_complete(_export_dashboard_cache())
        return result

    # Handle async walk-forward jobs
    if event.get("walk_forward_job"):
        print("📊 Walk-forward async job received")
        job_config = event["walk_forward_job"]
        result = asyncio.get_event_loop().run_until_complete(_run_walk_forward_job(job_config))
        return result

    # For API Gateway events, use Mangum
    return _mangum_handler(event, context)

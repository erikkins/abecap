"""
Scheduler Service - Daily market data updates

Runs after market close (4:30 PM ET) on trading days to:
1. Fetch fresh price data from yfinance
2. Run market scan for new signals
3. Store signals in database
4. Check open positions for stop/target hits
"""

import asyncio
from datetime import datetime, time
from typing import Optional, Callable, List
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from app.core.config import settings
from app.services.scanner import scanner_service
from app.services.email_service import email_service
from app.services.data_export import data_export_service

logger = logging.getLogger(__name__)

# US Eastern timezone for market hours
ET = pytz.timezone('US/Eastern')


class SchedulerService:
    """
    Manages scheduled jobs for market data updates
    """

    def __init__(self):
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.is_running = False
        self.last_run: Optional[datetime] = None
        self.last_run_status: Optional[str] = None
        self.run_count = 0
        self.callbacks: List[Callable] = []

    def add_callback(self, callback: Callable):
        """Add a callback to be called after each scheduled run"""
        self.callbacks.append(callback)

    async def daily_update(self):
        """
        Main daily update job

        Runs after market close to:
        1. Fetch fresh data
        2. Generate signals
        3. Log results
        """
        start_time = datetime.now(ET)
        logger.info(f"🕐 Starting daily update at {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        try:
            # Check if market was open today (skip weekends/holidays)
            if not self._is_trading_day(start_time):
                logger.info("📅 Not a trading day, skipping update")
                self.last_run_status = "skipped_non_trading_day"
                return

            # Ensure universe is loaded
            await scanner_service.ensure_universe_loaded()
            universe_size = len(scanner_service.universe)
            cache_size = len(scanner_service.data_cache)
            coverage = cache_size / universe_size if universe_size > 0 else 0

            # Check if we need to fill gaps or just do incremental update
            COVERAGE_THRESHOLD = 0.9  # 90% coverage required for incremental mode

            if coverage < COVERAGE_THRESHOLD:
                # Below threshold - need to fill gaps with full historical fetch
                logger.info(f"📊 Cache coverage {coverage:.1%} below {COVERAGE_THRESHOLD:.0%} threshold")
                logger.info(f"📊 Fetching full historical data to fill gaps...")
                await scanner_service.fetch_data(period="5y")
                symbols_loaded = len(scanner_service.data_cache)
                new_coverage = symbols_loaded / universe_size if universe_size > 0 else 0
                logger.info(f"✅ Full fetch complete: {symbols_loaded} symbols ({new_coverage:.1%} coverage)")
            else:
                # Good coverage - just get today's prices (fast incremental update)
                logger.info(f"📊 Cache coverage {coverage:.1%} OK - fetching today's prices (incremental)...")
                fetch_result = await scanner_service.fetch_incremental()
                symbols_loaded = len(scanner_service.data_cache)
                logger.info(f"✅ Incremental update: {fetch_result.get('updated', 0)} updated, "
                           f"{fetch_result.get('skipped', 0)} skipped, {symbols_loaded} total symbols")

            # Auto-save to S3/local after fetching new data
            logger.info("💾 Saving price data to persistent storage...")
            export_result = data_export_service.export_consolidated(scanner_service.data_cache)
            if export_result.get("success"):
                logger.info(f"✅ Saved {export_result.get('count', 0)} symbols to {export_result.get('storage', 'storage')}")
            else:
                logger.warning(f"⚠️ Data export failed: {export_result.get('message', 'unknown error')}")

            # Run scan
            logger.info("🔍 Running market scan...")
            signals = await scanner_service.scan(refresh_data=False)
            strong_signals = [s for s in signals if s.is_strong]

            logger.info(f"📈 Found {len(signals)} signals ({len(strong_signals)} strong)")

            # Log signal details
            for sig in signals[:5]:  # Log top 5
                logger.info(
                    f"   {'🔥' if sig.is_strong else '📊'} {sig.symbol}: "
                    f"${sig.price:.2f} ({sig.pct_above_dwap:+.1f}% > DWAP)"
                )

            # Run callbacks (e.g., store to DB, send alerts)
            for callback in self.callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(signals)
                    else:
                        callback(signals)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            # Update status
            self.last_run = datetime.now(ET)
            self.last_run_status = "success"
            self.run_count += 1

            elapsed = (datetime.now(ET) - start_time).total_seconds()
            logger.info(f"✅ Daily update complete in {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"❌ Daily update failed: {e}")
            self.last_run_status = f"error: {str(e)}"
            raise

    def _is_trading_day(self, dt: datetime) -> bool:
        """
        Check if the given date is a trading day

        Basic check: weekday only (Mon-Fri)
        TODO: Add holiday calendar (NYSE holidays)
        """
        return dt.weekday() < 5  # 0-4 = Mon-Fri

    def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler already running")
            return

        self.scheduler = AsyncIOScheduler(timezone=ET)

        # Schedule daily update at 4:30 PM ET (after market close)
        self.scheduler.add_job(
            self.daily_update,
            CronTrigger(
                day_of_week='mon-fri',
                hour=16,
                minute=30,
                timezone=ET
            ),
            id='daily_market_update',
            name='Daily Market Update',
            replace_existing=True
        )

        # Also run a pre-market update at 9:00 AM ET
        self.scheduler.add_job(
            self._premarket_update,
            CronTrigger(
                day_of_week='mon-fri',
                hour=9,
                minute=0,
                timezone=ET
            ),
            id='premarket_update',
            name='Pre-Market Data Refresh',
            replace_existing=True
        )

        # Send daily email summary at 6:00 PM ET (dinnertime)
        self.scheduler.add_job(
            self.send_daily_emails,
            CronTrigger(
                day_of_week='mon-fri',
                hour=18,
                minute=0,
                timezone=ET
            ),
            id='daily_email',
            name='Daily Email Summary',
            replace_existing=True
        )

        self.scheduler.start()
        self.is_running = True

        next_run = self.scheduler.get_job('daily_market_update').next_run_time
        logger.info(f"📅 Scheduler started. Next daily update: {next_run}")

    async def _premarket_update(self):
        """
        Pre-market update - just refresh data, don't scan

        Useful to have fresh data available when market opens.
        Uses incremental fetch if coverage is good, otherwise fills gaps.
        """
        logger.info("🌅 Pre-market data refresh starting...")
        try:
            # Ensure universe is loaded
            await scanner_service.ensure_universe_loaded()
            universe_size = len(scanner_service.universe)
            cache_size = len(scanner_service.data_cache)
            coverage = cache_size / universe_size if universe_size > 0 else 0

            COVERAGE_THRESHOLD = 0.9

            if coverage < COVERAGE_THRESHOLD:
                # Fill gaps with full fetch
                logger.info(f"📊 Coverage {coverage:.1%} below threshold - filling gaps...")
                await scanner_service.fetch_data(period="5y")
                logger.info(f"✅ Full fetch complete: {len(scanner_service.data_cache)} symbols")
            else:
                # Incremental update
                fetch_result = await scanner_service.fetch_incremental()
                logger.info(f"✅ Pre-market refresh complete: {fetch_result.get('updated', 0)} updated, "
                           f"{len(scanner_service.data_cache)} total symbols")

            # Auto-save to S3/local
            export_result = data_export_service.export_consolidated(scanner_service.data_cache)
            if export_result.get("success"):
                logger.info(f"💾 Saved to {export_result.get('storage', 'storage')}")
        except Exception as e:
            logger.error(f"❌ Pre-market refresh failed: {e}")

    async def send_daily_emails(self):
        """
        Send daily summary emails to subscribers

        Runs at 6 PM ET (dinnertime) on trading days
        """
        logger.info("📧 Starting daily email job...")

        try:
            # Check if this is a trading day
            now = datetime.now(ET)
            if not self._is_trading_day(now):
                logger.info("📅 Not a trading day, skipping emails")
                return

            # Get today's signals
            signals = await scanner_service.scan(refresh_data=False)
            signal_dicts = [s.to_dict() for s in signals]

            # Get market regime
            from app.services.market_analysis import market_analysis_service
            regime = market_analysis_service.get_market_regime()

            # TODO: Get subscribers from database
            # For now, just log what we would send
            subscribers = []  # Would come from database User table

            if not subscribers:
                logger.info("📧 No subscribers configured, skipping email send")
                # Log a sample of what the email would look like
                strong_count = len([s for s in signals if s.is_strong])
                logger.info(f"📧 Would have sent email with {len(signals)} signals ({strong_count} strong)")
                return

            # Send emails
            result = await email_service.send_bulk_daily_summary(
                subscribers=subscribers,
                signals=signal_dicts,
                market_regime=regime
            )

            logger.info(f"📧 Daily emails complete: {result['sent']} sent, {result['failed']} failed")

        except Exception as e:
            logger.error(f"❌ Daily email job failed: {e}")

    def stop(self):
        """Stop the scheduler"""
        if self.scheduler and self.is_running:
            self.scheduler.shutdown(wait=False)
            self.is_running = False
            logger.info("🛑 Scheduler stopped")

    def get_status(self) -> dict:
        """Get scheduler status"""
        status = {
            "is_running": self.is_running,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_run_status": self.last_run_status,
            "run_count": self.run_count,
            "next_runs": []
        }

        if self.scheduler and self.is_running:
            jobs = self.scheduler.get_jobs()
            status["next_runs"] = [
                {
                    "job_id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None
                }
                for job in jobs
            ]

        return status

    async def run_now(self):
        """Manually trigger the daily update (for testing)"""
        logger.info("🚀 Manual trigger: running daily update now")
        await self.daily_update()


# Singleton instance
scheduler_service = SchedulerService()

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
from app.services.email_service import email_service, admin_email_service
from app.services.data_export import data_export_service
from app.services.stock_universe import MUST_INCLUDE

logger = logging.getLogger(__name__)

# Admin email for alerts
ADMIN_EMAIL = "erik@rigacap.com"

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
        # Track alerted double signals to avoid duplicate alerts
        self._alerted_double_signals: set = set()

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

            # Run daily walk-forward simulation for dashboard stats
            try:
                logger.info("📊 Running daily walk-forward simulation...")
                await self._run_daily_walk_forward()
                logger.info("✅ Walk-forward simulation complete")
            except Exception as wf_err:
                logger.error(f"⚠️ Walk-forward simulation failed: {wf_err}")
                # Don't fail the whole update if walk-forward fails

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

    async def _run_daily_walk_forward(self):
        """
        Run a 1-year walk-forward simulation and cache results for dashboard.

        Uses biweekly reoptimization without AI to keep it fast (~30 seconds).
        Results are stored in WalkForwardSimulation table with is_daily_cache=True.
        """
        from datetime import timedelta
        from app.core.database import async_session, WalkForwardSimulation
        from app.services.walk_forward_service import walk_forward_service
        from sqlalchemy import select, delete

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year lookback

        async with async_session() as db:
            try:
                # Delete old daily cache entries (keep only last one)
                await db.execute(
                    delete(WalkForwardSimulation).where(
                        WalkForwardSimulation.is_daily_cache == True
                    )
                )

                # Create new job record
                job = WalkForwardSimulation(
                    simulation_date=datetime.utcnow(),
                    start_date=start_date,
                    end_date=end_date,
                    reoptimization_frequency="biweekly",
                    status="running",
                    is_daily_cache=True,  # Mark as dashboard cache
                    total_return_pct=0,
                    sharpe_ratio=0,
                    max_drawdown_pct=0,
                    num_strategy_switches=0,
                    benchmark_return_pct=0,
                )
                db.add(job)
                await db.commit()
                await db.refresh(job)
                job_id = job.id

                logger.info(f"[DAILY-WF] Starting 1-year walk-forward (job {job_id})")

                # Run walk-forward without AI optimization (faster)
                result = await walk_forward_service.run_walk_forward_simulation(
                    db=db,
                    start_date=start_date,
                    end_date=end_date,
                    reoptimization_frequency="biweekly",
                    min_score_diff=10.0,
                    enable_ai_optimization=False,  # No AI for speed
                    max_symbols=100,  # Smaller universe for speed
                    existing_job_id=job_id
                )

                logger.info(f"[DAILY-WF] Complete: {result.total_return_pct:.1f}% return, "
                           f"{result.num_strategy_switches} switches, "
                           f"benchmark {result.benchmark_return_pct:.1f}%")

            except Exception as e:
                logger.error(f"[DAILY-WF] Failed: {e}")
                # Update job status to failed
                result = await db.execute(
                    select(WalkForwardSimulation).where(WalkForwardSimulation.id == job_id)
                )
                job = result.scalar_one_or_none()
                if job:
                    job.status = "failed"
                    await db.commit()
                raise

    async def _run_nightly_walk_forward(self):
        """
        Nightly walk-forward simulation for missed opportunities + social content.

        Runs at 8 PM ET (after daily emails at 6 PM) on trading days.
        1. Runs 90-day rolling WF simulation with ensemble strategy
        2. Stores results with is_nightly_missed_opps=True for dashboard
        3. Generates social media content from best trades
        """
        from datetime import timedelta
        from app.core.database import async_session, WalkForwardSimulation
        from app.services.walk_forward_service import walk_forward_service
        from sqlalchemy import select, delete

        now = datetime.now(ET)
        if not self._is_trading_day(now):
            logger.info("📅 Not a trading day, skipping nightly walk-forward")
            return

        logger.info("🌙 Starting nightly walk-forward simulation...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        async with async_session() as db:
            try:
                # Delete old nightly missed opps cache
                await db.execute(
                    delete(WalkForwardSimulation).where(
                        WalkForwardSimulation.is_nightly_missed_opps == True
                    )
                )

                # Create new job record
                job = WalkForwardSimulation(
                    simulation_date=datetime.utcnow(),
                    start_date=start_date,
                    end_date=end_date,
                    reoptimization_frequency="biweekly",
                    status="running",
                    is_nightly_missed_opps=True,
                    total_return_pct=0,
                    sharpe_ratio=0,
                    max_drawdown_pct=0,
                    num_strategy_switches=0,
                    benchmark_return_pct=0,
                )
                db.add(job)
                await db.commit()
                await db.refresh(job)
                job_id = job.id

                logger.info(f"[NIGHTLY-WF] Starting 90-day walk-forward (job {job_id})")

                # Run walk-forward with ensemble strategy, no AI
                result = await walk_forward_service.run_walk_forward_simulation(
                    db=db,
                    start_date=start_date,
                    end_date=end_date,
                    reoptimization_frequency="biweekly",
                    min_score_diff=10.0,
                    enable_ai_optimization=False,
                    max_symbols=100,
                    existing_job_id=job_id,
                    fixed_strategy_id=5,  # Ensemble strategy
                )

                logger.info(f"[NIGHTLY-WF] Complete: {result.total_return_pct:.1f}% return, "
                           f"{result.num_strategy_switches} switches")

                # Generate social content from trades
                try:
                    from app.services.social_content_service import social_content_service
                    posts = await social_content_service.generate_from_nightly_wf(db, job_id)
                    logger.info(f"[NIGHTLY-WF] Generated {len(posts)} social posts")

                    # On Fridays, also generate weekly recap
                    if now.weekday() == 4:  # Friday
                        recap_posts = await social_content_service.generate_weekly_recap(db)
                        logger.info(f"[NIGHTLY-WF] Generated {len(recap_posts)} weekly recap posts")
                except Exception as social_err:
                    logger.error(f"[NIGHTLY-WF] Social content generation failed: {social_err}")

            except Exception as e:
                logger.error(f"[NIGHTLY-WF] Failed: {e}")
                import traceback
                traceback.print_exc()
                # Update job status to failed
                try:
                    result = await db.execute(
                        select(WalkForwardSimulation).where(WalkForwardSimulation.id == job_id)
                    )
                    job = result.scalar_one_or_none()
                    if job:
                        job.status = "failed"
                        await db.commit()
                except Exception:
                    pass

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

        # Check for new double signals at 5:00 PM ET (after daily update)
        # Sends alert email when momentum stocks cross DWAP +5%
        self.scheduler.add_job(
            self.check_double_signal_alerts,
            CronTrigger(
                day_of_week='mon-fri',
                hour=17,
                minute=0,
                timezone=ET
            ),
            id='double_signal_alert',
            name='Double Signal Alert Check',
            replace_existing=True
        )

        # Ticker health check at 7:00 AM ET (before market open)
        # Checks open positions and must-include symbols for data issues
        self.scheduler.add_job(
            self.check_ticker_health,
            CronTrigger(
                day_of_week='mon-fri',
                hour=7,
                minute=0,
                timezone=ET
            ),
            id='ticker_health_check',
            name='Ticker Health Check',
            replace_existing=True
        )

        # Nightly walk-forward + social content at 8 PM ET
        self.scheduler.add_job(
            self._run_nightly_walk_forward,
            CronTrigger(
                day_of_week='mon-fri',
                hour=20,
                minute=0,
                timezone=ET
            ),
            id='nightly_walk_forward',
            name='Nightly Walk-Forward + Social Content',
            replace_existing=True
        )

        # Strategy auto-analysis every other Friday at 6 PM ET
        # Runs biweekly strategy analysis and potential auto-switch
        self.scheduler.add_job(
            self._strategy_auto_analysis,
            CronTrigger(
                day_of_week='fri',
                hour=18,
                minute=30,
                week='*/2',  # Every 2 weeks
                timezone=ET
            ),
            id='strategy_auto_analysis',
            name='Strategy Auto-Analysis',
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

    async def check_ticker_health(self):
        """
        Daily health check for ticker issues.

        Runs at 7 AM ET to detect:
        - Open positions with missing/stale data
        - Must-include symbols that fail to resolve

        Sends alert email if issues found.
        """
        logger.info("🏥 Starting daily ticker health check...")

        try:
            issues = []

            # Import database components
            try:
                from app.core.database import async_session, db_available
                from app.core.database import Position as DBPosition
                from sqlalchemy import select
            except ImportError as e:
                logger.warning(f"Database not available for health check: {e}")
                db_available = False

            # Check 1: Open positions
            if db_available:
                try:
                    async with async_session() as session:
                        result = await session.execute(
                            select(DBPosition).where(DBPosition.status == "open")
                        )
                        positions = result.scalars().all()

                        for pos in positions:
                            symbol = pos.symbol
                            issue = await self._check_symbol_health(symbol)
                            if issue:
                                issue['last_price'] = f"{pos.entry_price:.2f}" if pos.entry_price else "N/A"
                                issue['last_date'] = pos.entry_date.strftime('%Y-%m-%d') if pos.entry_date else "N/A"
                                issue['suggestion'] = f"You have an open position in {symbol}. Research if ticker changed or company was acquired."
                                issues.append(issue)

                        logger.info(f"✅ Checked {len(positions)} open positions")
                except Exception as e:
                    logger.error(f"Failed to check positions: {e}")

            # Check 2: Must-include symbols (sample check - top 10)
            must_check = MUST_INCLUDE[:10]  # Check first 10 must-includes
            for symbol in must_check:
                # Skip if already in issues
                if any(i['symbol'] == symbol for i in issues):
                    continue

                issue = await self._check_symbol_health(symbol)
                if issue:
                    issue['suggestion'] = f"{symbol} is in MUST_INCLUDE list. Check for ticker change and update stock_universe.py."
                    issues.append(issue)

            logger.info(f"✅ Checked {len(must_check)} must-include symbols")

            # Send alert if issues found
            if issues:
                logger.warning(f"⚠️ Found {len(issues)} ticker issues!")
                for issue in issues:
                    logger.warning(f"   • {issue['symbol']}: {issue['issue']}")

                # Send alert email
                await admin_email_service.send_ticker_alert(
                    to_email=ADMIN_EMAIL,
                    issues=issues,
                    check_type="position"
                )
                logger.info(f"📧 Alert email sent to {ADMIN_EMAIL}")
            else:
                logger.info("✅ All tickers healthy - no issues found")

        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            import traceback
            traceback.print_exc()

    async def _check_symbol_health(self, symbol: str) -> dict:
        """
        Check if a symbol can return valid data.

        Returns:
            dict with issue details if problem found, None if healthy
        """
        try:
            import yfinance as yf

            # First check cache
            if symbol in scanner_service.data_cache:
                df = scanner_service.data_cache[symbol]
                if not df.empty:
                    # Check if data is recent (within 7 days)
                    last_date = df.index[-1]
                    days_old = (datetime.now() - last_date.to_pydatetime().replace(tzinfo=None)).days
                    if days_old <= 7:
                        return None  # Healthy

            # Try to fetch fresh data
            stock = yf.Ticker(symbol)
            hist = stock.history(period="5d")

            if hist.empty:
                # Check if ticker info exists at all
                info = stock.info
                if not info or not info.get('regularMarketPrice'):
                    return {
                        'symbol': symbol,
                        'issue': 'No data available - possibly delisted or ticker changed',
                        'last_price': 'N/A',
                        'last_date': 'N/A'
                    }

            return None  # Healthy

        except Exception as e:
            return {
                'symbol': symbol,
                'issue': f'Failed to fetch data: {str(e)[:50]}',
                'last_price': 'N/A',
                'last_date': 'N/A'
            }

    async def check_double_signal_alerts(self):
        """
        Check for new double signals and send email alerts.

        Runs at 5:00 PM ET to detect momentum stocks that just crossed DWAP +5%.
        Only alerts on NEW crossovers (not previously alerted).
        """
        logger.info("⚡ Checking for new double signal alerts...")

        try:
            # Check if this is a trading day
            now = datetime.now(ET)
            if not self._is_trading_day(now):
                logger.info("📅 Not a trading day, skipping double signal check")
                return

            import pandas as pd
            from app.api.signals import find_dwap_crossover_date

            # Get momentum rankings (top 20)
            momentum_rankings = scanner_service.rank_stocks_momentum(apply_market_filter=True)
            top_momentum = {
                r.symbol: {'rank': i + 1, 'data': r}
                for i, r in enumerate(momentum_rankings[:20])
            }

            # Find stocks that crossed DWAP +5% recently (within last 3 trading days)
            new_signals = []
            approaching = []
            today_str = now.strftime('%Y-%m-%d')

            for symbol, mom in top_momentum.items():
                df = scanner_service.data_cache.get(symbol)
                if df is None or len(df) < 1:
                    continue

                row = df.iloc[-1]
                price = row['close']
                dwap = row.get('dwap')

                if pd.isna(dwap) or dwap <= 0:
                    continue

                pct_above = (price / dwap - 1) * 100

                # Check for approaching trigger (3-5%)
                if 3.0 <= pct_above < 5.0:
                    distance = 5.0 - pct_above
                    approaching.append({
                        'symbol': symbol,
                        'price': float(price),
                        'pct_above_dwap': pct_above,
                        'distance_to_trigger': distance,
                        'momentum_rank': mom['rank'],
                    })

                # Check for double signals (>= 5% above DWAP)
                elif pct_above >= 5.0:
                    # Find crossover date
                    crossover_date, days_since = find_dwap_crossover_date(symbol, threshold_pct=5.0, lookback_days=5)

                    # Only alert if crossover was recent (within last 3 days) and not already alerted
                    alert_key = f"{symbol}_{crossover_date or today_str}"

                    if days_since is not None and days_since <= 3 and alert_key not in self._alerted_double_signals:
                        mom_data = mom['data']
                        new_signals.append({
                            'symbol': symbol,
                            'price': float(price),
                            'dwap': float(dwap),
                            'pct_above_dwap': pct_above,
                            'momentum_rank': mom['rank'],
                            'momentum_score': mom_data.composite_score,
                            'short_momentum': mom_data.short_momentum,
                            'long_momentum': mom_data.long_momentum,
                            'dwap_crossover_date': crossover_date or today_str,
                            'days_since_crossover': days_since or 0,
                        })
                        # Mark as alerted
                        self._alerted_double_signals.add(alert_key)

            # Clean up old alert keys (keep last 100)
            if len(self._alerted_double_signals) > 100:
                # Convert to list, sort, keep recent ones
                self._alerted_double_signals = set(list(self._alerted_double_signals)[-50:])

            # Send alert email if new signals found
            if new_signals:
                logger.info(f"⚡ Found {len(new_signals)} new double signal(s)!")
                for sig in new_signals:
                    logger.info(f"   • {sig['symbol']}: ${sig['price']:.2f} (+{sig['pct_above_dwap']:.1f}%) - Mom #{sig['momentum_rank']}")

                # Get market regime for context
                from app.services.market_analysis import market_analysis_service
                regime = market_analysis_service.get_market_regime()

                # Send email alert
                success = await email_service.send_double_signal_alert(
                    to_email=ADMIN_EMAIL,
                    new_signals=new_signals,
                    approaching=approaching,
                    market_regime=regime
                )

                if success:
                    logger.info(f"📧 Double signal alert sent to {ADMIN_EMAIL}")
                else:
                    logger.warning("⚠️ Failed to send double signal alert email")
            else:
                logger.info(f"✅ No new double signals (checked {len(top_momentum)} momentum stocks)")
                if approaching:
                    logger.info(f"   👀 {len(approaching)} stocks approaching trigger")

        except Exception as e:
            logger.error(f"❌ Double signal alert check failed: {e}")
            import traceback
            traceback.print_exc()

    async def send_daily_emails(self):
        """
        Send daily summary emails to subscribers

        Runs at 6 PM ET (dinnertime) on trading days.
        Builds ensemble signals (same logic as dashboard) with freshness tracking.
        """
        logger.info("📧 Starting daily email job...")

        try:
            # Check if this is a trading day
            now = datetime.now(ET)
            if not self._is_trading_day(now):
                logger.info("📅 Not a trading day, skipping emails")
                return

            from app.api.signals import find_dwap_crossover_date

            # Build ensemble signals (same as dashboard endpoint)
            dwap_signals = await scanner_service.scan(refresh_data=False, apply_market_filter=True)
            dwap_by_symbol = {s.symbol: s for s in dwap_signals}

            momentum_rankings = scanner_service.rank_stocks_momentum(apply_market_filter=True)
            momentum_top_n = 20
            fresh_days = 5
            momentum_by_symbol = {
                r.symbol: {'rank': i + 1, 'data': r}
                for i, r in enumerate(momentum_rankings[:momentum_top_n])
            }

            # Build ensemble buy signals with freshness
            buy_signals = []
            for symbol in dwap_by_symbol:
                if symbol in momentum_by_symbol:
                    dwap = dwap_by_symbol[symbol]
                    mom = momentum_by_symbol[symbol]
                    mom_data = mom['data']
                    mom_rank = mom['rank']

                    dwap_score = min(dwap.pct_above_dwap * 10, 50)
                    rank_score = (momentum_top_n - mom_rank + 1) * 2.5
                    ensemble_score = dwap_score + rank_score

                    crossover_date, days_since = find_dwap_crossover_date(symbol)
                    is_fresh = days_since is not None and days_since <= fresh_days

                    buy_signals.append({
                        'symbol': symbol,
                        'price': float(dwap.price),
                        'pct_above_dwap': float(dwap.pct_above_dwap),
                        'is_strong': bool(dwap.is_strong),
                        'momentum_rank': int(mom_rank),
                        'ensemble_score': round(float(ensemble_score), 1),
                        'dwap_crossover_date': crossover_date,
                        'days_since_crossover': int(days_since) if days_since is not None else None,
                        'is_fresh': bool(is_fresh),
                    })

            buy_signals.sort(key=lambda x: (
                0 if x['is_fresh'] else 1,
                x.get('days_since_crossover') or 999,
                -x['ensemble_score']
            ))

            # Build watchlist (approaching trigger)
            watchlist = []
            for r in momentum_rankings[:momentum_top_n]:
                if r.symbol not in dwap_by_symbol:
                    df = scanner_service.data_cache.get(r.symbol)
                    if df is not None and len(df) >= 200:
                        price = float(df['close'].iloc[-1])
                        dwap_val = float(df['dwap'].iloc[-1]) if 'dwap' in df.columns else 0
                        if dwap_val > 0:
                            pct_above = (price / dwap_val - 1) * 100
                            distance = 5.0 - pct_above
                            if 0 < distance <= 3.0:
                                watchlist.append({
                                    'symbol': r.symbol,
                                    'price': price,
                                    'pct_above_dwap': round(pct_above, 1),
                                    'distance_to_trigger': round(distance, 1),
                                })

            watchlist.sort(key=lambda x: x['distance_to_trigger'])

            # Get market regime
            from app.services.market_analysis import market_analysis_service
            regime = market_analysis_service.get_market_regime()

            # TODO: Get subscribers from database
            # For now, just log what we would send
            subscribers = []  # Would come from database User table

            if not subscribers:
                logger.info("📧 No subscribers configured, skipping email send")
                fresh_count = len([s for s in buy_signals if s.get('is_fresh')])
                logger.info(f"📧 Would have sent email with {len(buy_signals)} ensemble signals ({fresh_count} fresh), {len(watchlist)} watchlist")
                return

            # Send emails to each subscriber
            sent = 0
            failed = 0
            for sub_email in subscribers:
                try:
                    success = await email_service.send_daily_summary(
                        to_email=sub_email,
                        signals=buy_signals,
                        market_regime=regime,
                        watchlist=watchlist,
                    )
                    if success:
                        sent += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Failed to send to {sub_email}: {e}")
                    failed += 1

            logger.info(f"📧 Daily emails complete: {sent} sent, {failed} failed")

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

    async def _strategy_auto_analysis(self):
        """
        Biweekly strategy analysis and potential auto-switch.

        This job runs every other Friday to:
        1. Analyze all strategies with 90-day rolling backtest
        2. Check if a switch is recommended
        3. Execute switch if safeguards pass and auto-switch is enabled
        4. Send email notifications
        """
        logger.info("📊 Starting biweekly strategy auto-analysis...")

        try:
            from app.services.auto_switch_service import auto_switch_service
            await auto_switch_service.scheduled_analysis_job()
            logger.info("✅ Strategy auto-analysis complete")
        except Exception as e:
            logger.error(f"❌ Strategy auto-analysis failed: {e}")
            import traceback
            traceback.print_exc()


# Singleton instance
scheduler_service = SchedulerService()

"""
Model Portfolio Service — Dual live + walk-forward portfolio tracking.

Two parallel model portfolios run forward from launch:
1. Live Portfolio: Intraday monitoring (5 min), trailing stop/regime exits, no forced rebalancing.
2. Walk-Forward Portfolio: Biweekly rebalancing (Feb 1 canonical dates), daily close checks,
   force-close at period boundaries.

Both generate "We Called It" social content on profitable exits.
"""

import json
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import ModelPosition, ModelPortfolioState, ModelPortfolioSnapshot

logger = logging.getLogger(__name__)

# Constants
PORTFOLIO_TYPES = ("live", "walkforward")
MAX_POSITIONS = 6
POSITION_SIZE_PCT = 0.15  # 15% of available cash per position
TRAILING_STOP_PCT = 12.0
STARTING_CAPITAL = 100_000.0
WF_ANCHOR_DATE = date(2026, 2, 1)  # Canonical biweekly boundaries
WF_PERIOD_DAYS = 14


class ModelPortfolioService:
    """Track dual model portfolios: live (intraday) and walk-forward (biweekly)."""

    async def _get_or_create_state(
        self, db: AsyncSession, portfolio_type: str
    ) -> ModelPortfolioState:
        """Lazy-init portfolio state row with starting capital."""
        result = await db.execute(
            select(ModelPortfolioState).where(
                ModelPortfolioState.portfolio_type == portfolio_type
            )
        )
        state = result.scalar_one_or_none()
        if not state:
            state = ModelPortfolioState(
                portfolio_type=portfolio_type,
                starting_capital=STARTING_CAPITAL,
                current_cash=STARTING_CAPITAL,
                total_trades=0,
                winning_trades=0,
                total_pnl=0.0,
            )
            db.add(state)
            await db.flush()
            logger.info(f"[MODEL-{portfolio_type.upper()}] Initialized with ${STARTING_CAPITAL:,.0f}")
        return state

    async def _get_open_positions(
        self, db: AsyncSession, portfolio_type: str
    ) -> List[ModelPosition]:
        result = await db.execute(
            select(ModelPosition).where(
                ModelPosition.portfolio_type == portfolio_type,
                ModelPosition.status == "open",
            )
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    async def process_entries(
        self, db: AsyncSession, portfolio_type: str
    ) -> dict:
        """
        After daily scan, enter fresh ensemble signals.

        Reads buy signals from dashboard cache (S3 JSON). Filters to is_fresh=True,
        skips symbols already held. Sorts by ensemble_score, enters up to
        MAX_POSITIONS - open_count.

        For WF portfolio: only enters at biweekly boundaries.
        """
        if portfolio_type not in PORTFOLIO_TYPES:
            return {"error": f"Invalid portfolio type: {portfolio_type}"}

        # WF portfolio only enters at period boundaries
        today = date.today()
        if portfolio_type == "walkforward" and not self._is_wf_period_boundary(today):
            return {"entries": 0, "reason": "Not a WF period boundary"}

        state = await self._get_or_create_state(db, portfolio_type)
        open_positions = await self._get_open_positions(db, portfolio_type)
        open_count = len(open_positions)

        if open_count >= MAX_POSITIONS:
            return {"entries": 0, "reason": "Max positions reached"}

        held_symbols = {p.symbol for p in open_positions}
        slots = MAX_POSITIONS - open_count

        # Read fresh signals from dashboard cache
        from app.services.data_export import data_export_service

        dashboard = data_export_service.read_dashboard_json()
        if not dashboard:
            return {"entries": 0, "reason": "No dashboard cache available"}

        buy_signals = dashboard.get("buy_signals", [])
        fresh_signals = [
            s for s in buy_signals
            if s.get("is_fresh") and s["symbol"] not in held_symbols
        ]
        fresh_signals.sort(key=lambda x: -x.get("ensemble_score", 0))

        entries = 0
        for sig in fresh_signals[:slots]:
            symbol = sig["symbol"]
            price = sig.get("price", 0)
            if price <= 0:
                continue

            # Position sizing: 15% of current cash
            alloc = state.current_cash * POSITION_SIZE_PCT
            if alloc < 100:  # Minimum allocation
                break

            shares = alloc / price

            pos = ModelPosition(
                portfolio_type=portfolio_type,
                symbol=symbol,
                entry_date=datetime.utcnow(),
                entry_price=price,
                shares=shares,
                cost_basis=alloc,
                highest_price=price,
                status="open",
                signal_data_json=json.dumps(sig),
            )
            db.add(pos)
            state.current_cash -= alloc
            entries += 1

            logger.info(
                f"[MODEL-{portfolio_type.upper()}] Entered {symbol} @ ${price:.2f} "
                f"x {shares:.1f} shares (${alloc:,.0f})"
            )

        if entries:
            state.updated_at = datetime.utcnow()
            await db.commit()

        return {"entries": entries, "cash_remaining": round(state.current_cash, 2)}

    # ------------------------------------------------------------------
    # Exit logic — Live (intraday)
    # ------------------------------------------------------------------

    async def process_live_exits(
        self,
        db: AsyncSession,
        live_prices: Dict[str, float],
        regime_forecast: Optional[dict] = None,
    ) -> List[dict]:
        """
        Called by intraday monitor every 5 min.
        Checks all open live positions for trailing stop (12% from HWM) and regime exit.
        Updates highest_price if new high.
        """
        positions = await self._get_open_positions(db, "live")
        if not positions:
            return []

        closed = []
        for pos in positions:
            price = live_prices.get(pos.symbol)
            if price is None:
                continue

            # Update HWM
            if price > (pos.highest_price or pos.entry_price):
                pos.highest_price = price

            hwm = pos.highest_price or pos.entry_price
            trailing_stop_level = hwm * (1 - TRAILING_STOP_PCT / 100)

            exit_reason = None

            # Check regime exit
            if regime_forecast:
                rec = regime_forecast.get("recommended_action", "stay_invested")
                if rec == "go_to_cash":
                    exit_reason = "regime_exit"

            # Check trailing stop (overrides regime)
            if price <= trailing_stop_level:
                exit_reason = "trailing_stop"

            if exit_reason:
                result = await self._close_position(db, pos, price, exit_reason)
                closed.append(result)

        if closed:
            await db.commit()

        return closed

    # ------------------------------------------------------------------
    # Exit logic — Walk-Forward (daily close)
    # ------------------------------------------------------------------

    async def process_wf_exits(self, db: AsyncSession) -> List[dict]:
        """
        Called once daily after market close.
        Checks all open WF positions using daily close prices.
        Force-closes all if today is a biweekly boundary (rebalance_exit).
        """
        positions = await self._get_open_positions(db, "walkforward")
        if not positions:
            return []

        today = date.today()
        is_boundary = self._is_wf_period_boundary(today)

        # Get daily close prices from scanner cache
        from app.services.scanner import scanner_service

        closed = []
        for pos in positions:
            df = scanner_service.data_cache.get(pos.symbol)
            if df is None or df.empty:
                continue

            close_price = float(df["close"].iloc[-1])

            # Update HWM
            if close_price > (pos.highest_price or pos.entry_price):
                pos.highest_price = close_price

            exit_reason = None

            # Period boundary → force close
            if is_boundary:
                exit_reason = "rebalance_exit"
            else:
                # Check trailing stop
                hwm = pos.highest_price or pos.entry_price
                trailing_stop_level = hwm * (1 - TRAILING_STOP_PCT / 100)
                if close_price <= trailing_stop_level:
                    exit_reason = "trailing_stop"

            if exit_reason:
                result = await self._close_position(db, pos, close_price, exit_reason)
                closed.append(result)

        if closed:
            await db.commit()

        return closed

    # ------------------------------------------------------------------
    # Shared close logic
    # ------------------------------------------------------------------

    async def _close_position(
        self,
        db: AsyncSession,
        position: ModelPosition,
        exit_price: float,
        exit_reason: str,
    ) -> dict:
        """Close a model position: set exit fields, calculate P&L, update state."""
        pnl_dollars = (exit_price - position.entry_price) * position.shares
        pnl_pct = ((exit_price / position.entry_price) - 1) * 100

        position.exit_date = datetime.utcnow()
        position.exit_price = exit_price
        position.exit_reason = exit_reason
        position.pnl_dollars = round(pnl_dollars, 2)
        position.pnl_pct = round(pnl_pct, 2)
        position.status = "closed"

        # Update portfolio state
        state = await self._get_or_create_state(db, position.portfolio_type)
        state.current_cash += position.cost_basis + pnl_dollars
        state.total_trades += 1
        state.total_pnl += pnl_dollars
        if pnl_pct > 0:
            state.winning_trades += 1
        state.updated_at = datetime.utcnow()

        logger.info(
            f"[MODEL-{position.portfolio_type.upper()}] Closed {position.symbol} "
            f"@ ${exit_price:.2f} ({pnl_pct:+.1f}%) — {exit_reason}"
        )

        # Generate social content for profitable exits
        if pnl_pct >= 5 and not position.social_post_generated:
            try:
                await self._generate_exit_content(db, position)
                position.social_post_generated = True
            except Exception as e:
                logger.error(f"[MODEL-PORTFOLIO] Social content generation failed: {e}")

        return {
            "symbol": position.symbol,
            "portfolio_type": position.portfolio_type,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "pnl_pct": round(pnl_pct, 2),
            "pnl_dollars": round(pnl_dollars, 2),
            "exit_reason": exit_reason,
        }

    # ------------------------------------------------------------------
    # Social content generation on exit
    # ------------------------------------------------------------------

    async def _generate_exit_content(
        self, db: AsyncSession, position: ModelPosition
    ) -> None:
        """
        Generate social posts for profitable model portfolio exits.
        trade_result for >5%, we_called_it for >10%.
        """
        from app.services.ai_content_service import ai_content_service
        from app.services.post_scheduler_service import post_scheduler_service

        trade = {
            "symbol": position.symbol,
            "entry_price": position.entry_price,
            "exit_price": position.exit_price,
            "entry_date": position.entry_date.isoformat() if position.entry_date else "",
            "exit_date": position.exit_date.isoformat() if position.exit_date else "",
            "pnl_pct": position.pnl_pct,
            "exit_reason": position.exit_reason,
        }

        post_type = "we_called_it" if position.pnl_pct >= 10 else "trade_result"

        for platform in ("twitter", "instagram"):
            post = await ai_content_service.generate_post(
                trade=trade,
                post_type=post_type,
                platform=platform,
            )
            if post:
                db.add(post)
                await db.flush()
                # Auto-schedule the draft
                await post_scheduler_service.auto_schedule_drafts(db)

        logger.info(
            f"[MODEL-PORTFOLIO] Generated {post_type} content for "
            f"{position.symbol} ({position.pnl_pct:+.1f}%)"
        )

    # ------------------------------------------------------------------
    # Portfolio summary
    # ------------------------------------------------------------------

    async def get_portfolio_summary(
        self, db: AsyncSession, portfolio_type: Optional[str] = None
    ) -> dict:
        """
        Returns current state: capital, cash, open positions, realized/unrealized P&L,
        win rate, recent trades. If portfolio_type=None, returns both side by side.
        """
        types = [portfolio_type] if portfolio_type else list(PORTFOLIO_TYPES)
        result = {}

        for ptype in types:
            state = await self._get_or_create_state(db, ptype)
            open_positions = await self._get_open_positions(db, ptype)

            # Get live prices for unrealized P&L
            from app.services.scanner import scanner_service

            open_data = []
            unrealized_pnl = 0.0
            for pos in open_positions:
                df = scanner_service.data_cache.get(pos.symbol)
                current_price = float(df["close"].iloc[-1]) if df is not None and not df.empty else pos.entry_price
                pos_pnl = (current_price - pos.entry_price) * pos.shares
                unrealized_pnl += pos_pnl
                open_data.append({
                    "symbol": pos.symbol,
                    "entry_date": pos.entry_date.isoformat() if pos.entry_date else None,
                    "entry_price": pos.entry_price,
                    "shares": round(pos.shares, 2),
                    "current_price": round(current_price, 2),
                    "pnl_pct": round(((current_price / pos.entry_price) - 1) * 100, 2),
                    "pnl_dollars": round(pos_pnl, 2),
                    "highest_price": pos.highest_price,
                })

            # Recent closed trades
            recent_result = await db.execute(
                select(ModelPosition)
                .where(
                    ModelPosition.portfolio_type == ptype,
                    ModelPosition.status == "closed",
                )
                .order_by(ModelPosition.exit_date.desc())
                .limit(10)
            )
            recent_trades = [
                {
                    "id": t.id,
                    "symbol": t.symbol,
                    "entry_date": t.entry_date.isoformat() if t.entry_date else None,
                    "exit_date": t.exit_date.isoformat() if t.exit_date else None,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl_pct": t.pnl_pct,
                    "pnl_dollars": t.pnl_dollars,
                    "exit_reason": t.exit_reason,
                }
                for t in recent_result.scalars().all()
            ]

            positions_value = sum(p["current_price"] * p["shares"] for p in open_data)
            total_value = state.current_cash + positions_value

            win_rate = (
                (state.winning_trades / state.total_trades * 100)
                if state.total_trades > 0
                else 0
            )

            result[ptype] = {
                "starting_capital": state.starting_capital,
                "current_cash": round(state.current_cash, 2),
                "total_value": round(total_value, 2),
                "total_return_pct": round(
                    ((total_value / state.starting_capital) - 1) * 100, 2
                ),
                "realized_pnl": round(state.total_pnl, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "total_trades": state.total_trades,
                "winning_trades": state.winning_trades,
                "win_rate": round(win_rate, 1),
                "open_positions": open_data,
                "recent_trades": recent_trades,
            }

        return result if not portfolio_type else result.get(portfolio_type, {})

    # ------------------------------------------------------------------
    # Walk-forward biweekly boundary logic
    # ------------------------------------------------------------------

    def _is_wf_period_boundary(self, today: date) -> bool:
        """Check if today is the first trading day of a new biweekly period.

        Boundaries fall on Sundays (Feb 1 2026 anchor + 14-day periods),
        so we check if today is Monday (or the first weekday) after a boundary.
        Specifically: the boundary that *just passed* (within 0-2 days ago)
        should not already have been the boundary for a previous trading day.
        """
        days_since = (today - WF_ANCHOR_DATE).days
        if days_since < 0:
            return False
        # Check if a boundary falls on today or within the last 2 days
        # (covers Sat/Sun boundaries triggering on Monday)
        for offset in range(3):
            check_date = today - timedelta(days=offset)
            ds = (check_date - WF_ANCHOR_DATE).days
            if ds >= 0 and ds % WF_PERIOD_DAYS == 0:
                # Boundary is on check_date. Today triggers if it's the first
                # weekday on or after check_date.
                first_weekday = check_date
                while first_weekday.weekday() >= 5:  # Sat=5, Sun=6
                    first_weekday += timedelta(days=1)
                return today == first_weekday
        return False

    # ------------------------------------------------------------------
    # Reset (admin/testing)
    # ------------------------------------------------------------------

    async def reset_portfolio(
        self, db: AsyncSession, portfolio_type: Optional[str] = None
    ) -> dict:
        """Reset one or both portfolios. Deletes all positions and resets state."""
        from sqlalchemy import delete

        types = [portfolio_type] if portfolio_type else list(PORTFOLIO_TYPES)
        deleted = 0

        for ptype in types:
            result = await db.execute(
                delete(ModelPosition).where(
                    ModelPosition.portfolio_type == ptype
                )
            )
            deleted += result.rowcount

            await db.execute(
                delete(ModelPortfolioState).where(
                    ModelPortfolioState.portfolio_type == ptype
                )
            )

            # Also clear snapshots
            await db.execute(
                delete(ModelPortfolioSnapshot).where(
                    ModelPortfolioSnapshot.portfolio_type == ptype
                )
            )

        await db.commit()
        return {"deleted_positions": deleted, "reset_types": types}

    # ------------------------------------------------------------------
    # Backfill walk-forward portfolio from a historical date
    # ------------------------------------------------------------------

    async def backfill_from_date(
        self,
        db: AsyncSession,
        as_of_date: str = "2026-02-01",
        force: bool = False,
    ) -> dict:
        """
        Backfill the WF portfolio from a historical date through today.

        Steps:
        1. Reset WF portfolio if force=True
        2. Load signals from snapshot or live computation for the start date
        3. Enter top fresh ensemble signals
        4. Walk forward day by day: update HWM, check trailing stop, rebalance at boundaries
        5. Take daily snapshots for equity curve
        """
        import pandas as pd
        from sqlalchemy import delete

        portfolio_type = "walkforward"

        if force:
            await self.reset_portfolio(db, portfolio_type)
            logger.info("[BACKFILL] Reset WF portfolio")

        # Get scanner data for price lookups
        from app.services.scanner import scanner_service

        spy_df = scanner_service.data_cache.get("SPY")
        if spy_df is None or spy_df.empty:
            return {"error": "SPY data not in cache — run a scan first"}

        # Build trading calendar from SPY index
        start = pd.Timestamp(as_of_date).normalize()
        today = pd.Timestamp(date.today()).normalize()

        # Normalize SPY index for comparison
        spy_index = spy_df.index.normalize()
        trading_days = sorted([d for d in spy_index if start <= d <= today])
        if not trading_days:
            return {"error": f"No trading days found between {as_of_date} and today"}

        # Initialize state
        state = await self._get_or_create_state(db, portfolio_type)
        summary = {
            "start_date": str(start.date()),
            "end_date": str(today.date()),
            "trading_days": len(trading_days),
            "entries": 0,
            "exits": 0,
            "rebalances": 0,
            "snapshots": 0,
        }

        # Get WF boundary dates in range
        boundaries = set()
        d = WF_ANCHOR_DATE
        while d <= today.date():
            if d >= start.date():
                boundaries.add(pd.Timestamp(d).normalize())
            d += timedelta(days=WF_PERIOD_DAYS)

        logger.info(
            f"[BACKFILL] {len(trading_days)} trading days, "
            f"{len(boundaries)} WF boundaries"
        )

        for day in trading_days:
            day_date = day.date()
            is_boundary = day in boundaries

            # Get current open positions
            open_positions = await self._get_open_positions(db, portfolio_type)

            # --- Check exits ---
            for pos in open_positions:
                df = scanner_service.data_cache.get(pos.symbol)
                if df is None or df.empty:
                    continue

                # Get close price for this day
                df_norm = df.index.normalize()
                day_mask = df_norm == day
                if not day_mask.any():
                    continue

                close_price = float(df.loc[day_mask, "close"].iloc[-1])

                # Update HWM
                if close_price > (pos.highest_price or pos.entry_price):
                    pos.highest_price = close_price

                exit_reason = None
                if is_boundary:
                    exit_reason = "rebalance_exit"
                else:
                    hwm = pos.highest_price or pos.entry_price
                    stop_level = hwm * (1 - TRAILING_STOP_PCT / 100)
                    if close_price <= stop_level:
                        exit_reason = "trailing_stop"

                if exit_reason:
                    pnl_dollars = (close_price - pos.entry_price) * pos.shares
                    pnl_pct = ((close_price / pos.entry_price) - 1) * 100

                    pos.exit_date = datetime.combine(day_date, datetime.min.time())
                    pos.exit_price = close_price
                    pos.exit_reason = exit_reason
                    pos.pnl_dollars = round(pnl_dollars, 2)
                    pos.pnl_pct = round(pnl_pct, 2)
                    pos.status = "closed"

                    state.current_cash += pos.cost_basis + pnl_dollars
                    state.total_trades += 1
                    state.total_pnl += pnl_dollars
                    if pnl_pct > 0:
                        state.winning_trades += 1

                    summary["exits"] += 1

            # --- Enter new positions at boundaries or first day ---
            if is_boundary or day == trading_days[0]:
                if is_boundary and day != trading_days[0]:
                    summary["rebalances"] += 1

                # Load signals for this date
                signals = await self._get_signals_for_date(day_date)
                if signals:
                    open_positions = await self._get_open_positions(db, portfolio_type)
                    held = {p.symbol for p in open_positions}
                    slots = MAX_POSITIONS - len(open_positions)

                    fresh = [
                        s for s in signals
                        if s.get("is_fresh") and s["symbol"] not in held
                    ]
                    fresh.sort(key=lambda x: -x.get("ensemble_score", 0))

                    for sig in fresh[:slots]:
                        symbol = sig["symbol"]
                        # Get close price for entry day
                        df = scanner_service.data_cache.get(symbol)
                        if df is None or df.empty:
                            continue

                        df_norm = df.index.normalize()
                        day_mask = df_norm == day
                        if not day_mask.any():
                            continue

                        price = float(df.loc[day_mask, "close"].iloc[-1])
                        if price <= 0:
                            continue

                        alloc = state.current_cash * POSITION_SIZE_PCT
                        if alloc < 100:
                            break

                        shares = alloc / price
                        pos = ModelPosition(
                            portfolio_type=portfolio_type,
                            symbol=symbol,
                            entry_date=datetime.combine(day_date, datetime.min.time()),
                            entry_price=price,
                            shares=shares,
                            cost_basis=alloc,
                            highest_price=price,
                            status="open",
                            signal_data_json=json.dumps(sig),
                        )
                        db.add(pos)
                        state.current_cash -= alloc
                        summary["entries"] += 1

            # --- Take daily snapshot ---
            open_positions = await self._get_open_positions(db, portfolio_type)
            positions_value = 0.0
            for pos in open_positions:
                df = scanner_service.data_cache.get(pos.symbol)
                if df is None or df.empty:
                    continue
                df_norm = df.index.normalize()
                day_mask = df_norm == day
                if day_mask.any():
                    close_price = float(df.loc[day_mask, "close"].iloc[-1])
                    positions_value += close_price * pos.shares

            spy_mask = spy_index == day
            spy_close = float(spy_df.loc[spy_mask, "close"].iloc[-1]) if spy_mask.any() else None

            snapshot = ModelPortfolioSnapshot(
                portfolio_type=portfolio_type,
                snapshot_date=datetime.combine(day_date, datetime.min.time()),
                total_value=round(state.current_cash + positions_value, 2),
                cash=round(state.current_cash, 2),
                positions_value=round(positions_value, 2),
                num_positions=len(open_positions),
                spy_close=spy_close,
            )
            db.add(snapshot)
            summary["snapshots"] += 1

            # Flush periodically to keep session clean
            if summary["snapshots"] % 10 == 0:
                await db.flush()

        state.updated_at = datetime.utcnow()
        await db.commit()

        logger.info(
            f"[BACKFILL] Complete: {summary['entries']} entries, "
            f"{summary['exits']} exits, {summary['rebalances']} rebalances, "
            f"{summary['snapshots']} snapshots"
        )
        return summary

    async def _get_signals_for_date(self, target_date: date) -> Optional[List[dict]]:
        """Load ensemble signals for a given date from snapshot or live computation."""
        from app.services.data_export import data_export_service

        date_str = target_date.isoformat()

        # Try snapshot first
        snapshot = data_export_service.read_snapshot(date_str)
        if snapshot:
            return snapshot.get("buy_signals", [])

        # No snapshot available — skip (can't time-travel without cached data)
        logger.debug(f"[BACKFILL] No snapshot for {date_str}")
        return None

    # ------------------------------------------------------------------
    # Daily snapshot for equity curve
    # ------------------------------------------------------------------

    async def take_daily_snapshot(
        self, db: AsyncSession, snapshot_date: Optional[date] = None
    ) -> dict:
        """
        Take a snapshot of each portfolio's value for the equity curve.
        Uses scanner cache for current prices + SPY close.
        Upserts to avoid duplicates.
        """
        from app.services.scanner import scanner_service

        if snapshot_date is None:
            snapshot_date = date.today()

        snapshot_dt = datetime.combine(snapshot_date, datetime.min.time())
        results = {}

        for ptype in PORTFOLIO_TYPES:
            state = await self._get_or_create_state(db, ptype)
            open_positions = await self._get_open_positions(db, ptype)

            positions_value = 0.0
            for pos in open_positions:
                df = scanner_service.data_cache.get(pos.symbol)
                if df is not None and not df.empty:
                    positions_value += float(df["close"].iloc[-1]) * pos.shares

            spy_df = scanner_service.data_cache.get("SPY")
            spy_close = float(spy_df["close"].iloc[-1]) if spy_df is not None and not spy_df.empty else None

            total_value = state.current_cash + positions_value

            # Upsert: check existing
            existing = await db.execute(
                select(ModelPortfolioSnapshot).where(
                    ModelPortfolioSnapshot.portfolio_type == ptype,
                    ModelPortfolioSnapshot.snapshot_date == snapshot_dt,
                )
            )
            snap = existing.scalar_one_or_none()

            if snap:
                snap.total_value = round(total_value, 2)
                snap.cash = round(state.current_cash, 2)
                snap.positions_value = round(positions_value, 2)
                snap.num_positions = len(open_positions)
                snap.spy_close = spy_close
            else:
                snap = ModelPortfolioSnapshot(
                    portfolio_type=ptype,
                    snapshot_date=snapshot_dt,
                    total_value=round(total_value, 2),
                    cash=round(state.current_cash, 2),
                    positions_value=round(positions_value, 2),
                    num_positions=len(open_positions),
                    spy_close=spy_close,
                )
                db.add(snap)

            results[ptype] = {
                "total_value": round(total_value, 2),
                "positions_value": round(positions_value, 2),
                "num_positions": len(open_positions),
            }

        await db.commit()
        logger.info(f"[SNAPSHOT] Taken for {snapshot_date}: {results}")
        return results

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    async def get_equity_curve(
        self, db: AsyncSession, portfolio_type: Optional[str] = None
    ) -> List[dict]:
        """
        Return equity curve data points for charting.
        Normalizes SPY to $100K starting value for comparison.
        """
        from sqlalchemy import asc

        query = select(ModelPortfolioSnapshot).order_by(
            asc(ModelPortfolioSnapshot.snapshot_date)
        )
        if portfolio_type:
            query = query.where(ModelPortfolioSnapshot.portfolio_type == portfolio_type)

        result = await db.execute(query)
        snapshots = result.scalars().all()

        if not snapshots:
            return []

        # Group by date
        by_date: Dict[str, dict] = {}
        first_spy = None

        for s in snapshots:
            date_str = s.snapshot_date.strftime("%Y-%m-%d") if s.snapshot_date else ""
            if date_str not in by_date:
                by_date[date_str] = {"date": date_str}

            key = f"{s.portfolio_type}_value"
            by_date[date_str][key] = s.total_value

            if s.spy_close and first_spy is None:
                first_spy = s.spy_close

            if s.spy_close and first_spy:
                by_date[date_str]["spy_value"] = round(
                    STARTING_CAPITAL * (s.spy_close / first_spy), 2
                )

        return sorted(by_date.values(), key=lambda x: x["date"])

    # ------------------------------------------------------------------
    # Trade journal
    # ------------------------------------------------------------------

    async def get_all_trades(
        self,
        db: AsyncSession,
        portfolio_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        """Return all trades (closed + open) ordered by entry_date desc."""
        query = select(ModelPosition).order_by(ModelPosition.entry_date.desc())
        if portfolio_type:
            query = query.where(ModelPosition.portfolio_type == portfolio_type)
        query = query.limit(limit)

        result = await db.execute(query)
        trades = []
        for t in result.scalars().all():
            sig = {}
            if t.signal_data_json:
                try:
                    sig = json.loads(t.signal_data_json)
                except (json.JSONDecodeError, TypeError):
                    pass

            days_held = None
            if t.entry_date:
                end = t.exit_date or datetime.utcnow()
                days_held = (end - t.entry_date).days

            trades.append({
                "id": t.id,
                "portfolio_type": t.portfolio_type,
                "symbol": t.symbol,
                "status": t.status,
                "entry_date": t.entry_date.isoformat() if t.entry_date else None,
                "exit_date": t.exit_date.isoformat() if t.exit_date else None,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "shares": round(t.shares, 2) if t.shares else None,
                "cost_basis": round(t.cost_basis, 2) if t.cost_basis else None,
                "pnl_pct": t.pnl_pct,
                "pnl_dollars": t.pnl_dollars,
                "exit_reason": t.exit_reason,
                "days_held": days_held,
                "ensemble_score": sig.get("ensemble_score"),
                "momentum_rank": sig.get("momentum_rank"),
            })

        return trades

    async def get_trade_detail(self, db: AsyncSession, trade_id: int) -> Optional[dict]:
        """Return full detail for a single trade including signal replay."""
        result = await db.execute(
            select(ModelPosition).where(ModelPosition.id == trade_id)
        )
        t = result.scalar_one_or_none()
        if not t:
            return None

        sig = {}
        if t.signal_data_json:
            try:
                sig = json.loads(t.signal_data_json)
            except (json.JSONDecodeError, TypeError):
                pass

        days_held = None
        if t.entry_date:
            end = t.exit_date or datetime.utcnow()
            days_held = (end - t.entry_date).days

        # Calculate max gain during hold
        max_gain_pct = None
        if t.highest_price and t.entry_price:
            max_gain_pct = round(((t.highest_price / t.entry_price) - 1) * 100, 2)

        return {
            "id": t.id,
            "portfolio_type": t.portfolio_type,
            "symbol": t.symbol,
            "status": t.status,
            "entry_date": t.entry_date.isoformat() if t.entry_date else None,
            "exit_date": t.exit_date.isoformat() if t.exit_date else None,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "shares": round(t.shares, 2) if t.shares else None,
            "cost_basis": round(t.cost_basis, 2) if t.cost_basis else None,
            "highest_price": t.highest_price,
            "pnl_pct": t.pnl_pct,
            "pnl_dollars": t.pnl_dollars,
            "exit_reason": t.exit_reason,
            "days_held": days_held,
            "max_gain_pct": max_gain_pct,
            # Signal replay
            "ensemble_score": sig.get("ensemble_score"),
            "momentum_rank": sig.get("momentum_rank"),
            "pct_above_dwap": sig.get("pct_above_dwap"),
            "sector": sig.get("sector"),
            "short_momentum": sig.get("short_momentum"),
            "long_momentum": sig.get("long_momentum"),
            "volatility": sig.get("volatility"),
            "dwap_crossover_date": sig.get("dwap_crossover_date"),
            "ensemble_entry_date": sig.get("ensemble_entry_date"),
        }

    # ------------------------------------------------------------------
    # Subscriber preview
    # ------------------------------------------------------------------

    async def get_subscriber_view(self, db: AsyncSession) -> dict:
        """
        Return a subscriber-safe preview of the WF portfolio.
        Shows positions (symbol + P&L only), recent winners, and aggregate stats.
        """
        # Prefer WF portfolio (more history after backfill)
        state = await self._get_or_create_state(db, "walkforward")
        open_positions = await self._get_open_positions(db, "walkforward")

        from app.services.scanner import scanner_service

        # Open positions: symbol + P&L only (no shares/sizes)
        positions = []
        for pos in open_positions:
            df = scanner_service.data_cache.get(pos.symbol)
            current_price = float(df["close"].iloc[-1]) if df is not None and not df.empty else pos.entry_price
            pnl_pct = ((current_price / pos.entry_price) - 1) * 100
            positions.append({
                "symbol": pos.symbol,
                "pnl_pct": round(pnl_pct, 2),
            })

        # Recent winners (last 5 profitable closed trades)
        winners_result = await db.execute(
            select(ModelPosition)
            .where(
                ModelPosition.portfolio_type == "walkforward",
                ModelPosition.status == "closed",
                ModelPosition.pnl_pct > 0,
            )
            .order_by(ModelPosition.exit_date.desc())
            .limit(5)
        )
        recent_winners = [
            {
                "symbol": t.symbol,
                "pnl_pct": t.pnl_pct,
                "exit_date": t.exit_date.isoformat() if t.exit_date else None,
            }
            for t in winners_result.scalars().all()
        ]

        # Aggregate stats
        positions_value = 0.0
        for pos in open_positions:
            df = scanner_service.data_cache.get(pos.symbol)
            if df is not None and not df.empty:
                positions_value += float(df["close"].iloc[-1]) * pos.shares

        total_value = state.current_cash + positions_value
        portfolio_return_pct = ((total_value / state.starting_capital) - 1) * 100 if state.starting_capital else 0

        win_rate = (
            (state.winning_trades / state.total_trades * 100)
            if state.total_trades > 0
            else 0
        )

        # Find inception date from earliest position
        earliest = await db.execute(
            select(ModelPosition)
            .where(ModelPosition.portfolio_type == "walkforward")
            .order_by(ModelPosition.entry_date.asc())
            .limit(1)
        )
        first_pos = earliest.scalar_one_or_none()
        inception_date = first_pos.entry_date.date().isoformat() if first_pos and first_pos.entry_date else None
        active_days = (date.today() - first_pos.entry_date.date()).days if first_pos and first_pos.entry_date else 0

        return {
            "open_positions": positions,
            "recent_winners": recent_winners,
            "portfolio_return_pct": round(portfolio_return_pct, 2),
            "win_rate": round(win_rate, 1),
            "total_trades": state.total_trades,
            "inception_date": inception_date,
            "active_since_days": active_days,
        }


# Singleton
model_portfolio_service = ModelPortfolioService()

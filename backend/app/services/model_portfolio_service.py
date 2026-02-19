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

from app.core.database import ModelPosition, ModelPortfolioState

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
        """Check if today is a biweekly boundary from WF_ANCHOR_DATE."""
        days_since = (today - WF_ANCHOR_DATE).days
        return days_since >= 0 and days_since % WF_PERIOD_DAYS == 0

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

        await db.commit()
        return {"deleted_positions": deleted, "reset_types": types}


# Singleton
model_portfolio_service = ModelPortfolioService()

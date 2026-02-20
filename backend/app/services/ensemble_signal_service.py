"""
Ensemble Signal Persistence Service

Persists ensemble buy signals to the database for:
1. Audit trail — prove what signals fired when
2. Dashboard/email consistency — email reads from DB instead of regenerating
3. Signal accuracy tracking — link signals to outcomes
4. Invalidation tracking — know when signals dropped out
"""

import logging
from datetime import date, datetime
from typing import List, Optional, Set

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import EnsembleSignal

logger = logging.getLogger(__name__)


class EnsembleSignalService:

    async def persist_signals(
        self, db: AsyncSession, signals: List[dict], signal_date: date
    ) -> dict:
        """
        Upsert ensemble signals for a given date.

        Uses INSERT ... ON CONFLICT (signal_date, symbol) DO UPDATE
        so re-running the same day overwrites with latest data.

        Returns: {inserted: int, updated: int}
        """
        inserted = 0
        updated = 0

        for sig in signals:
            # Parse date strings to date objects
            dwap_crossover = None
            if sig.get("dwap_crossover_date"):
                try:
                    dwap_crossover = date.fromisoformat(str(sig["dwap_crossover_date"])[:10])
                except (ValueError, TypeError):
                    pass

            ensemble_entry = None
            if sig.get("ensemble_entry_date"):
                try:
                    ensemble_entry = date.fromisoformat(str(sig["ensemble_entry_date"])[:10])
                except (ValueError, TypeError):
                    pass

            values = {
                "signal_date": signal_date,
                "symbol": sig["symbol"],
                "price": sig.get("price"),
                "dwap": sig.get("dwap"),
                "pct_above_dwap": sig.get("pct_above_dwap"),
                "volume": sig.get("volume"),
                "volume_ratio": sig.get("volume_ratio"),
                "momentum_rank": sig.get("momentum_rank"),
                "momentum_score": sig.get("momentum_score"),
                "short_momentum": sig.get("short_momentum"),
                "long_momentum": sig.get("long_momentum"),
                "ensemble_score": sig.get("ensemble_score"),
                "dwap_crossover_date": dwap_crossover,
                "ensemble_entry_date": ensemble_entry,
                "days_since_crossover": sig.get("days_since_crossover"),
                "days_since_entry": sig.get("days_since_entry"),
                "is_fresh": sig.get("is_fresh", False),
                "is_strong": sig.get("is_strong", False),
                "sector": sig.get("sector", ""),
                "status": "active",
            }

            stmt = pg_insert(EnsembleSignal).values(**values)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_ensemble_signal_date_symbol",
                set_={
                    "price": stmt.excluded.price,
                    "dwap": stmt.excluded.dwap,
                    "pct_above_dwap": stmt.excluded.pct_above_dwap,
                    "volume": stmt.excluded.volume,
                    "volume_ratio": stmt.excluded.volume_ratio,
                    "momentum_rank": stmt.excluded.momentum_rank,
                    "momentum_score": stmt.excluded.momentum_score,
                    "short_momentum": stmt.excluded.short_momentum,
                    "long_momentum": stmt.excluded.long_momentum,
                    "ensemble_score": stmt.excluded.ensemble_score,
                    "dwap_crossover_date": stmt.excluded.dwap_crossover_date,
                    "ensemble_entry_date": stmt.excluded.ensemble_entry_date,
                    "days_since_crossover": stmt.excluded.days_since_crossover,
                    "days_since_entry": stmt.excluded.days_since_entry,
                    "is_fresh": stmt.excluded.is_fresh,
                    "is_strong": stmt.excluded.is_strong,
                    "sector": stmt.excluded.sector,
                    "status": stmt.excluded.status,
                },
            )

            result = await db.execute(stmt)
            # rowcount == 1 for both insert and update in PostgreSQL upsert
            # We can't distinguish perfectly, but track total
            if result.rowcount:
                inserted += 1

        await db.commit()
        return {"inserted": inserted, "updated": updated}

    async def invalidate_stale_signals(
        self, db: AsyncSession, signal_date: date, current_symbols: Set[str]
    ) -> int:
        """
        Mark today's signals as 'invalidated' if they were active but are no longer
        in the current signal set. This happens when a stock drops out of the
        ensemble between scans.

        Returns: number of signals invalidated
        """
        # Find active signals for today that are NOT in current_symbols
        result = await db.execute(
            select(EnsembleSignal).where(
                EnsembleSignal.signal_date == signal_date,
                EnsembleSignal.status == "active",
                EnsembleSignal.symbol.notin_(current_symbols) if current_symbols else True,
            )
        )
        stale = result.scalars().all()

        count = 0
        for sig in stale:
            sig.status = "invalidated"
            sig.invalidated_at = datetime.utcnow()
            sig.invalidation_reason = "dropped_from_ensemble"
            count += 1

        if count:
            await db.commit()

        return count

    async def get_signals_for_date(
        self, db: AsyncSession, signal_date: date
    ) -> List[EnsembleSignal]:
        """
        Get active signals for a given date. Used by email job to read
        the same signals that were computed at 4 PM.
        """
        result = await db.execute(
            select(EnsembleSignal)
            .where(
                EnsembleSignal.signal_date == signal_date,
                EnsembleSignal.status == "active",
            )
            .order_by(
                EnsembleSignal.is_fresh.desc(),
                EnsembleSignal.days_since_entry.asc().nullslast(),
                EnsembleSignal.ensemble_score.desc(),
            )
        )
        return result.scalars().all()

    async def query_signals(
        self,
        db: AsyncSession,
        symbol: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[EnsembleSignal]:
        """
        Flexible query for signal history. Used by admin history endpoint.
        """
        query = select(EnsembleSignal)

        if symbol:
            query = query.where(EnsembleSignal.symbol == symbol.upper())
        if start_date:
            query = query.where(EnsembleSignal.signal_date >= start_date)
        if end_date:
            query = query.where(EnsembleSignal.signal_date <= end_date)
        if status:
            query = query.where(EnsembleSignal.status == status)

        query = query.order_by(
            EnsembleSignal.signal_date.desc(), EnsembleSignal.ensemble_score.desc()
        ).limit(limit)

        result = await db.execute(query)
        return result.scalars().all()


# Singleton
ensemble_signal_service = EnsembleSignalService()

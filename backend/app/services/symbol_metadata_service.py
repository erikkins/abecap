"""
Symbol metadata + corporate-actions service.

Handles:
- Per-symbol identity tracking (Alpaca asset_id UUID → detect ticker reuse)
- Nightly corporate-actions poll (splits, dividends, spinoffs, mergers)
- Quarantine + audit-log writes

Called from the nightly data-hygiene Lambda handler. Idempotent — safe
to re-run.
"""
import json
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select, and_, or_

from app.core.config import settings
from app.core.database import async_session, SymbolMetadata, SymbolMetadataEvent

logger = logging.getLogger(__name__)


class SymbolMetadataService:
    """Asset-ID integrity + corp-actions pipeline."""

    def _get_trading_client(self):
        """Alpaca TradingClient for asset metadata (separate from price data)."""
        from alpaca.trading.client import TradingClient
        return TradingClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper=False,
        )

    def _get_corp_actions_client(self):
        """Alpaca CorporateActionsClient for splits/dividends/etc."""
        from alpaca.data.historical.corporate_actions import CorporateActionsClient
        return CorporateActionsClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
        )

    # ─────────────────────────── Asset-ID integrity ───────────────────────────

    async def verify_asset_ids(
        self, symbols: List[str], record_events: bool = True, concurrency: int = 10
    ) -> Dict[str, Dict]:
        """
        For each symbol in `symbols`, compare the stored asset_id against
        Alpaca's current asset_id. Records SymbolMetadataEvent entries for
        any mismatches and returns a summary.

        PARALLELIZED (Apr 15 2026): uses asyncio.gather with a semaphore
        (default 10 concurrent) to fetch asset records via a thread pool.
        4500 symbols drops from ~15 min (serial) to ~2-3 min (10-way).
        Alpaca Pro SIP handles 200 req/min — 10 concurrent stays well under.

        On first run (no stored asset_id yet), populates the record.

        Returns dict keyed by symbol: {
            "status": "ok" | "new" | "reused" | "missing_in_alpaca",
            "stored_asset_id": str | None,
            "current_asset_id": str | None,
        }
        """
        import asyncio as _asyncio
        client = self._get_trading_client()
        summary: Dict[str, Dict] = {}

        # Pre-fetch stored metadata for all symbols in one DB query
        async with async_session() as db:
            result = await db.execute(
                select(SymbolMetadata).where(SymbolMetadata.symbol.in_(symbols))
            )
            stored = {row.symbol: row for row in result.scalars().all()}

        # Fetch all asset records in parallel via thread pool
        sem = _asyncio.Semaphore(concurrency)
        loop = _asyncio.get_event_loop()

        async def _fetch_one(sym: str):
            async with sem:
                try:
                    asset = await loop.run_in_executor(None, client.get_asset, sym)
                    return sym, asset, None
                except Exception as e:
                    return sym, None, e

        fetch_tasks = [_fetch_one(s) for s in symbols]
        fetch_results = await _asyncio.gather(*fetch_tasks)

        # Now process the results in a DB transaction (single session)
        async with async_session() as db:
            # Re-load stored metadata in the new session (for ORM operations)
            result = await db.execute(
                select(SymbolMetadata).where(SymbolMetadata.symbol.in_(symbols))
            )
            stored = {row.symbol: row for row in result.scalars().all()}

            for symbol, asset, err in fetch_results:
                stored_meta = stored.get(symbol)
                if err is not None:
                    summary[symbol] = {
                        "status": "missing_in_alpaca",
                        "stored_asset_id": stored_meta.asset_id if stored_meta else None,
                        "current_asset_id": None,
                        "error": str(err)[:200],
                    }
                    if record_events:
                        db.add(SymbolMetadataEvent(
                            symbol=symbol,
                            event_type="missing_in_alpaca",
                            details_json=json.dumps({"error": str(err)[:200]}),
                        ))
                    continue

                try:
                    current_id = str(asset.id) if asset and asset.id else None
                except Exception as e:
                    summary[symbol] = {
                        "status": "missing_in_alpaca",
                        "stored_asset_id": stored_meta.asset_id if stored_meta else None,
                        "current_asset_id": None,
                        "error": str(e)[:200],
                    }
                    continue

                # No prior record — create one (first verification)
                if stored_meta is None:
                    db.add(SymbolMetadata(
                        symbol=symbol,
                        asset_id=current_id,
                        status="active",
                        last_verified_at=datetime.utcnow(),
                    ))
                    summary[symbol] = {
                        "status": "new",
                        "stored_asset_id": None,
                        "current_asset_id": current_id,
                    }
                    continue

                # Existing record — check for asset_id drift (ticker reuse!)
                if stored_meta.asset_id and stored_meta.asset_id != current_id:
                    summary[symbol] = {
                        "status": "reused",
                        "stored_asset_id": stored_meta.asset_id,
                        "current_asset_id": current_id,
                    }
                    if record_events:
                        db.add(SymbolMetadataEvent(
                            symbol=symbol,
                            event_type="asset_id_changed",
                            details_json=json.dumps({
                                "old_asset_id": stored_meta.asset_id,
                                "new_asset_id": current_id,
                            }),
                        ))
                    # Auto-quarantine for admin review
                    stored_meta.status = "quarantined"
                    stored_meta.quarantine_reason = "asset_id_changed"
                    stored_meta.quarantined_at = datetime.utcnow()
                    continue

                # Healthy: update asset_id (if unset) and last_verified_at
                if not stored_meta.asset_id:
                    stored_meta.asset_id = current_id
                stored_meta.last_verified_at = datetime.utcnow()
                summary[symbol] = {
                    "status": "ok",
                    "stored_asset_id": stored_meta.asset_id,
                    "current_asset_id": current_id,
                }

            await db.commit()

        return summary

    # ─────────────────────────── Corp-actions poll ───────────────────────────

    async def poll_corp_actions(
        self, since_hours: int = 36
    ) -> List[Dict]:
        """
        Query Alpaca's corporate-actions endpoint for events in the past
        N hours. Records each into SymbolMetadataEvent and returns a list
        of event dicts.

        Default window 36h covers a 24h nightly cadence with buffer.
        """
        from alpaca.data.requests import CorporateActionsRequest

        client = self._get_corp_actions_client()
        end = date.today()
        start = end - timedelta(days=max(2, since_hours // 24 + 1))

        try:
            req = CorporateActionsRequest(start=start, end=end)
            result = client.get_corporate_actions(req)
        except Exception as e:
            logger.error(f"corp-actions poll failed: {e}")
            return [{"error": str(e)[:300]}]

        events: List[Dict] = []
        raw_data = getattr(result, "data", {}) or {}

        async with async_session() as db:
            for action_type, actions in raw_data.items():
                for action in (actions or []):
                    # action is a pydantic model; extract fields generically
                    ad = getattr(action, "model_dump", lambda: dict(action))()
                    symbol = ad.get("symbol") or ad.get("target_symbol") or ad.get("initiating_symbol")
                    event_date_val = ad.get("ex_date") or ad.get("effective_date") or ad.get("process_date")
                    event = {
                        "symbol": symbol,
                        "event_type": str(action_type),
                        "event_date": event_date_val.isoformat() if hasattr(event_date_val, "isoformat") else event_date_val,
                        "details": ad,
                    }
                    events.append(event)
                    if symbol:
                        db.add(SymbolMetadataEvent(
                            symbol=symbol,
                            event_type=str(action_type),
                            event_date=event_date_val if hasattr(event_date_val, "year") else None,
                            details_json=json.dumps(ad, default=str)[:4000],
                        ))
            await db.commit()

        logger.info(f"corp-actions poll: {len(events)} events across {len(raw_data)} types")
        return events

    # ─────────────────────────── Admin utilities ───────────────────────────

    async def get_quarantined_symbols(self) -> List[Dict]:
        """Return all symbols currently in 'quarantined' status."""
        async with async_session() as db:
            result = await db.execute(
                select(SymbolMetadata).where(SymbolMetadata.status == "quarantined")
            )
            rows = result.scalars().all()
            return [{
                "symbol": r.symbol,
                "asset_id": r.asset_id,
                "reason": r.quarantine_reason,
                "quarantined_at": r.quarantined_at.isoformat() if r.quarantined_at else None,
            } for r in rows]

    async def get_excluded_symbols(self) -> List[str]:
        """List of symbols to exclude from signal generation (quarantined + inactive)."""
        async with async_session() as db:
            result = await db.execute(
                select(SymbolMetadata.symbol).where(
                    SymbolMetadata.status.in_(["quarantined", "inactive"])
                )
            )
            return [r[0] for r in result.all()]

    async def set_status(self, symbol: str, status: str, reason: Optional[str] = None) -> bool:
        """Manually set a symbol's status (admin override)."""
        assert status in {"active", "inactive", "quarantined"}
        async with async_session() as db:
            result = await db.execute(
                select(SymbolMetadata).where(SymbolMetadata.symbol == symbol)
            )
            row = result.scalar_one_or_none()
            if not row:
                return False
            row.status = status
            if status == "quarantined":
                row.quarantine_reason = reason or "manual"
                row.quarantined_at = datetime.utcnow()
            else:
                row.quarantine_reason = None
                row.quarantined_at = None
            db.add(SymbolMetadataEvent(
                symbol=symbol,
                event_type="manual_override",
                details_json=json.dumps({"new_status": status, "reason": reason}),
            ))
            await db.commit()
            return True


symbol_metadata_service = SymbolMetadataService()

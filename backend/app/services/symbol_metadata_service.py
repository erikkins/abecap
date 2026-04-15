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
        self, symbols: List[str], record_events: bool = True
    ) -> Dict[str, Dict]:
        """
        For each symbol in `symbols`, compare the stored asset_id against
        Alpaca's current asset_id. Records SymbolMetadataEvent entries for
        any mismatches and returns a summary.

        On first run (no stored asset_id yet), populates the record.

        Returns dict keyed by symbol: {
            "status": "ok" | "new" | "reused" | "missing_in_alpaca",
            "stored_asset_id": str | None,
            "current_asset_id": str | None,
            "first_listing_date": date | None,
        }
        """
        client = self._get_trading_client()
        # Alpaca's get_asset(symbol_or_asset_id) is per-symbol; no bulk-by-symbol API.
        # We batch via list comprehension but each call is a network round trip.
        # For 4500 symbols that's slow — cache-aware caller should batch over days.

        summary: Dict[str, Dict] = {}

        async with async_session() as db:
            # Load all stored metadata in one query
            result = await db.execute(
                select(SymbolMetadata).where(SymbolMetadata.symbol.in_(symbols))
            )
            stored = {row.symbol: row for row in result.scalars().all()}

            for symbol in symbols:
                stored_meta = stored.get(symbol)
                try:
                    asset = client.get_asset(symbol)
                    current_id = str(asset.id) if asset and asset.id else None
                    current_first_date = (
                        asset.attributes.get("first_trade_date") if asset and hasattr(asset, "attributes") else None
                    )
                except Exception as e:
                    # Asset not found in Alpaca (delisted or never listed)
                    summary[symbol] = {
                        "status": "missing_in_alpaca",
                        "stored_asset_id": stored_meta.asset_id if stored_meta else None,
                        "current_asset_id": None,
                        "error": str(e)[:200],
                    }
                    if record_events:
                        db.add(SymbolMetadataEvent(
                            symbol=symbol,
                            event_type="missing_in_alpaca",
                            details_json=json.dumps({"error": str(e)[:200]}),
                        ))
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

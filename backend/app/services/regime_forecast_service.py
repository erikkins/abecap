"""
Regime Forecast Service — Daily regime forecast storage, accuracy tracking, and heatmap data.

Persists daily snapshots from MarketRegimeService.predict_transitions() for
historical analysis, forecast accuracy measurement, and visualization.
"""

import json
import logging
from datetime import datetime, date, timedelta
from typing import List, Optional

from sqlalchemy import select, asc, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import RegimeForecastSnapshot

logger = logging.getLogger(__name__)


class RegimeForecastService:
    """Persist and query regime forecast snapshots."""

    async def take_snapshot(self, db: AsyncSession) -> dict:
        """
        Call predict_transitions() and store today's forecast.
        Called daily after scan completes.
        """
        from app.services.market_regime import market_regime_service
        from app.services.scanner import scanner_service

        spy_df = scanner_service.data_cache.get("SPY")
        if spy_df is None or spy_df.empty:
            return {"error": "SPY data not in cache"}

        vix_df = scanner_service.data_cache.get("^VIX") or scanner_service.data_cache.get("VIX")

        try:
            forecast = market_regime_service.predict_transitions(
                spy_df, scanner_service.data_cache, vix_df
            )
        except Exception as e:
            logger.error(f"Regime forecast failed: {e}")
            return {"error": str(e)}

        today_dt = datetime.combine(date.today(), datetime.min.time())

        # Get SPY/VIX close
        spy_close = float(spy_df["close"].iloc[-1]) if not spy_df.empty else None
        vix_close = None
        if vix_df is not None and not vix_df.empty:
            vix_close = float(vix_df["close"].iloc[-1])

        # Upsert
        existing = await db.execute(
            select(RegimeForecastSnapshot).where(
                RegimeForecastSnapshot.snapshot_date == today_dt
            )
        )
        snap = existing.scalar_one_or_none()

        forecast_dict = forecast.to_dict() if hasattr(forecast, "to_dict") else {}
        probabilities = forecast_dict.get("transition_probabilities", {})

        if snap:
            snap.current_regime = forecast_dict.get("current_regime", "unknown")
            snap.probabilities_json = json.dumps(probabilities)
            snap.outlook = forecast_dict.get("outlook")
            snap.recommended_action = forecast_dict.get("recommended_action")
            snap.risk_change = forecast_dict.get("risk_change")
            snap.spy_close = spy_close
            snap.vix_close = vix_close
        else:
            snap = RegimeForecastSnapshot(
                snapshot_date=today_dt,
                current_regime=forecast_dict.get("current_regime", "unknown"),
                probabilities_json=json.dumps(probabilities),
                outlook=forecast_dict.get("outlook"),
                recommended_action=forecast_dict.get("recommended_action"),
                risk_change=forecast_dict.get("risk_change"),
                spy_close=spy_close,
                vix_close=vix_close,
            )
            db.add(snap)

        await db.commit()
        logger.info(f"[REGIME-FORECAST] Snapshot taken: {forecast_dict.get('current_regime')}")
        return {
            "date": str(date.today()),
            "regime": forecast_dict.get("current_regime"),
            "outlook": forecast_dict.get("outlook"),
            "recommended_action": forecast_dict.get("recommended_action"),
        }

    async def get_forecast_history(
        self, db: AsyncSession, days: int = 90
    ) -> List[dict]:
        """Return forecast snapshots ordered by date."""
        cutoff = datetime.combine(
            date.today() - timedelta(days=days), datetime.min.time()
        )
        result = await db.execute(
            select(RegimeForecastSnapshot)
            .where(RegimeForecastSnapshot.snapshot_date >= cutoff)
            .order_by(asc(RegimeForecastSnapshot.snapshot_date))
        )
        snapshots = result.scalars().all()

        return [
            {
                "date": s.snapshot_date.strftime("%Y-%m-%d") if s.snapshot_date else "",
                "regime": s.current_regime,
                "probabilities": json.loads(s.probabilities_json) if s.probabilities_json else {},
                "outlook": s.outlook,
                "recommended_action": s.recommended_action,
                "risk_change": s.risk_change,
                "spy_close": s.spy_close,
                "vix_close": s.vix_close,
            }
            for s in snapshots
        ]

    async def get_forecast_accuracy(
        self, db: AsyncSession, days: int = 90
    ) -> dict:
        """
        Compare each forecast's predicted regime vs the actual regime N days later.
        Returns accuracy % and a confusion matrix.
        """
        cutoff = datetime.combine(
            date.today() - timedelta(days=days), datetime.min.time()
        )
        result = await db.execute(
            select(RegimeForecastSnapshot)
            .where(RegimeForecastSnapshot.snapshot_date >= cutoff)
            .order_by(asc(RegimeForecastSnapshot.snapshot_date))
        )
        snapshots = list(result.scalars().all())

        if len(snapshots) < 2:
            return {"accuracy_pct": None, "total_forecasts": len(snapshots), "note": "Not enough data"}

        # Build date→regime lookup
        date_regime = {
            s.snapshot_date.strftime("%Y-%m-%d"): s.current_regime
            for s in snapshots
        }

        correct = 0
        total = 0
        confusion = {}  # {predicted: {actual: count}}

        for snap in snapshots:
            # Get the regime that had the highest probability
            probs = json.loads(snap.probabilities_json) if snap.probabilities_json else {}
            if not probs:
                continue

            predicted_regime = max(probs, key=probs.get)

            # Check what the actual regime was the next day
            next_date = snap.snapshot_date + timedelta(days=1)
            actual_regime = date_regime.get(next_date.strftime("%Y-%m-%d"))
            if not actual_regime:
                # Try +2 days (weekend/holiday)
                next_date = snap.snapshot_date + timedelta(days=2)
                actual_regime = date_regime.get(next_date.strftime("%Y-%m-%d"))
            if not actual_regime:
                next_date = snap.snapshot_date + timedelta(days=3)
                actual_regime = date_regime.get(next_date.strftime("%Y-%m-%d"))
            if not actual_regime:
                continue

            total += 1
            if predicted_regime == actual_regime:
                correct += 1

            if predicted_regime not in confusion:
                confusion[predicted_regime] = {}
            confusion[predicted_regime][actual_regime] = confusion[predicted_regime].get(actual_regime, 0) + 1

        accuracy = round((correct / total * 100), 1) if total > 0 else None

        return {
            "accuracy_pct": accuracy,
            "correct": correct,
            "total_forecasts": total,
            "confusion_matrix": confusion,
        }

    async def get_transition_heatmap(
        self, db: AsyncSession, days: int = 60
    ) -> List[dict]:
        """Return daily probability arrays for heatmap visualization."""
        cutoff = datetime.combine(
            date.today() - timedelta(days=days), datetime.min.time()
        )
        result = await db.execute(
            select(RegimeForecastSnapshot)
            .where(RegimeForecastSnapshot.snapshot_date >= cutoff)
            .order_by(asc(RegimeForecastSnapshot.snapshot_date))
        )
        snapshots = result.scalars().all()

        return [
            {
                "date": s.snapshot_date.strftime("%Y-%m-%d") if s.snapshot_date else "",
                "regime": s.current_regime,
                "probabilities": json.loads(s.probabilities_json) if s.probabilities_json else {},
            }
            for s in snapshots
        ]


# Singleton
regime_forecast_service = RegimeForecastService()

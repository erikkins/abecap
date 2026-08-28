"""
Market Analysis Service - Regime Detection & Sector Analysis

Provides:
- Bull/Bear market regime detection
- Sector strength ranking
- Signal quality scoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    STRONG_BULL = "strong_bull"      # SPY > 200 MA, trending up strongly
    BULL = "bull"                     # SPY > 200 MA
    NEUTRAL = "neutral"               # SPY near 200 MA
    BEAR = "bear"                     # SPY < 200 MA
    STRONG_BEAR = "strong_bear"       # SPY < 200 MA, trending down strongly


def _safe_float(value, default=0.0):
    """Convert value to JSON-safe float (handle NaN/Inf)"""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return default
    return float(value)


@dataclass
class MarketState:
    """Current market state"""
    regime: MarketRegime
    spy_price: float
    spy_ma_200: float
    spy_ma_50: float
    spy_pct_from_high: float
    vix_level: float
    trend_strength: float  # -1 to 1
    recommendation: str
    updated: str

    def to_dict(self):
        spy = _safe_float(self.spy_price)
        ma200 = _safe_float(self.spy_ma_200)
        return {
            "regime": self.regime.value,
            "spy_price": spy,
            "spy_ma_200": ma200,
            "spy_ma_50": _safe_float(self.spy_ma_50),
            "spy_pct_from_high": _safe_float(self.spy_pct_from_high),
            "spy_above_200ma": spy > ma200 if spy and ma200 else True,
            "vix_level": _safe_float(self.vix_level, 20.0),
            "trend_strength": _safe_float(self.trend_strength),
            "recommendation": self.recommendation,
            "updated": self.updated
        }


class MarketAnalysisService:
    """
    Analyzes market conditions (legacy 5-regime; used by the legacy DWAP scan
    and admin market endpoints only — not the live momentum book or Maximizer).
    """

    def __init__(self):
        self.market_state: Optional[MarketState] = None
        self.last_updated: Optional[datetime] = None

    async def update_market_state(self) -> MarketState:
        """
        Update market regime detection

        Uses SPY and VIX to determine bull/bear market conditions.
        SPY via DualSourceProvider (Alpaca primary), ^VIX always via yfinance.
        """
        try:
            from app.services.market_data_provider import market_data_provider
            from datetime import timedelta

            start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
            bars = await market_data_provider.fetch_bars(["SPY", "^VIX"], start_date)

            spy_df = bars.get("SPY")
            vix_df = bars.get("^VIX")

            if spy_df is None or spy_df.empty:
                raise ValueError("Could not fetch SPY data")

            spy_close = spy_df['close'].dropna()
            vix_close = vix_df['close'].dropna() if vix_df is not None and not vix_df.empty else pd.Series(dtype=float)

            if len(spy_close) < 200:
                raise ValueError("Not enough SPY data")

            # Calculate indicators
            spy_price = float(spy_close.iloc[-1])
            spy_ma_200 = float(spy_close.rolling(200).mean().iloc[-1])
            spy_ma_50 = float(spy_close.rolling(50).mean().iloc[-1])
            spy_high_52w = float(spy_close.rolling(252).max().iloc[-1])
            spy_pct_from_high = (spy_price - spy_high_52w) / spy_high_52w * 100
            vix_level = float(vix_close.iloc[-1]) if len(vix_close) > 0 else 20

            # Calculate trend strength (-1 to 1)
            # Based on price position relative to moving averages
            ma_spread = (spy_ma_50 - spy_ma_200) / spy_ma_200 * 100
            price_vs_ma = (spy_price - spy_ma_200) / spy_ma_200 * 100
            trend_strength = np.clip((price_vs_ma + ma_spread) / 20, -1, 1)

            # Determine regime
            if spy_price > spy_ma_200:
                if spy_ma_50 > spy_ma_200 and price_vs_ma > 5 and vix_level < 20:
                    regime = MarketRegime.STRONG_BULL
                else:
                    regime = MarketRegime.BULL
            elif spy_price < spy_ma_200:
                if spy_ma_50 < spy_ma_200 and price_vs_ma < -5 and vix_level > 25:
                    regime = MarketRegime.STRONG_BEAR
                else:
                    regime = MarketRegime.BEAR
            else:
                regime = MarketRegime.NEUTRAL

            # Generate recommendation
            if regime == MarketRegime.STRONG_BULL:
                recommendation = "Full exposure. Favor growth sectors. Take all strong signals."
            elif regime == MarketRegime.BULL:
                recommendation = "Normal exposure. Mix of growth and value. Be selective with signals."
            elif regime == MarketRegime.NEUTRAL:
                recommendation = "Reduced exposure. Focus on strongest signals only. Tighter stops."
            elif regime == MarketRegime.BEAR:
                recommendation = "Minimal exposure. Only take exceptional signals. Consider cash."
            else:  # STRONG_BEAR
                recommendation = "Avoid new positions. Preserve capital. Wait for regime change."

            self.market_state = MarketState(
                regime=regime,
                spy_price=round(spy_price, 2),
                spy_ma_200=round(spy_ma_200, 2),
                spy_ma_50=round(spy_ma_50, 2),
                spy_pct_from_high=round(spy_pct_from_high, 1),
                vix_level=round(vix_level, 1),
                trend_strength=round(trend_strength, 2),
                recommendation=recommendation,
                updated=datetime.now().isoformat()
            )

            self.last_updated = datetime.now()
            logger.info(f"Market regime: {regime.value}, VIX: {vix_level:.1f}")

            return self.market_state

        except Exception as e:
            logger.error(f"Failed to update market state: {e}")
            # Return a default neutral state
            return MarketState(
                regime=MarketRegime.NEUTRAL,
                spy_price=0,
                spy_ma_200=0,
                spy_ma_50=0,
                spy_pct_from_high=0,
                vix_level=20,
                trend_strength=0,
                recommendation="Market data unavailable. Trade with caution.",
                updated=datetime.now().isoformat()
            )

    def get_market_regime(self) -> Dict:
        """
        Get current market regime as a dictionary.
        Returns cached state or a neutral default if not initialized.
        """
        if self.market_state:
            return self.market_state.to_dict()
        # Return default neutral state if not initialized
        return {
            "regime": "neutral",
            "spy_price": 0,
            "spy_ma_200": 0,
            "spy_ma_50": 0,
            "spy_pct_from_high": 0,
            "vix_level": 20,
            "trend_strength": 0,
            "recommendation": "Market data not yet loaded",
            "updated": None
        }

    def get_sector_for_stock(self, symbol: str, symbol_info: dict = None) -> Optional[str]:
        """Get sector for a stock symbol"""
        if symbol_info and "sector" in symbol_info:
            return symbol_info.get("sector")
        return None

    def should_take_signal(self, signal_strength: float) -> Tuple[bool, str]:
        """
        Determine if a signal should be taken based on market regime

        Args:
            signal_strength: Signal quality score (0-100)

        Returns:
            (should_take, reason)
        """
        if not self.market_state:
            return True, "No market data - using default rules"

        regime = self.market_state.regime

        if regime == MarketRegime.STRONG_BULL:
            # Take almost any signal
            if signal_strength >= 30:
                return True, "Strong bull market - signal accepted"
            return False, "Signal too weak even for bull market"

        elif regime == MarketRegime.BULL:
            # Be somewhat selective
            if signal_strength >= 50:
                return True, "Bull market - good signal accepted"
            return False, "Signal below threshold for current market"

        elif regime == MarketRegime.NEUTRAL:
            # Be selective
            if signal_strength >= 65:
                return True, "Neutral market - strong signal accepted"
            return False, "Only taking strong signals in neutral market"

        elif regime == MarketRegime.BEAR:
            # Very selective
            if signal_strength >= 80:
                return True, "Bear market - exceptional signal accepted"
            return False, "Avoiding most signals in bear market"

        else:  # STRONG_BEAR
            # Avoid almost everything
            if signal_strength >= 90:
                return True, "Strong bear - only taking exceptional signals"
            return False, "Avoiding signals in strong bear market"

    def calculate_signal_strength(
        self,
        pct_above_dwap: float,
        volume_ratio: float,
        is_strong: bool,
    ) -> float:
        """
        Calculate composite signal strength score (0-100)

        Factors:
        - % above DWAP (higher = stronger)
        - Volume ratio (higher = stronger)
        - Strong signal flag
        """
        score = 0

        # DWAP component (0-30 points)
        # 5% above = 15 points, 10% above = 30 points
        dwap_score = min(pct_above_dwap * 3, 30)
        score += dwap_score

        # Volume component (0-25 points)
        # 1.5x = 12.5 points, 3x = 25 points
        vol_score = min((volume_ratio - 1) * 12.5, 25)
        score += max(vol_score, 0)

        # Strong signal bonus (0-20 points)
        if is_strong:
            score += 20

        # Market regime adjustment (0-10 points bonus in bull, -10 in bear)
        if self.market_state:
            if self.market_state.regime == MarketRegime.STRONG_BULL:
                score += 10
            elif self.market_state.regime == MarketRegime.BULL:
                score += 5
            elif self.market_state.regime == MarketRegime.BEAR:
                score -= 5
            elif self.market_state.regime == MarketRegime.STRONG_BEAR:
                score -= 10

        return round(np.clip(score, 0, 100), 1)


# Singleton instance
market_analysis_service = MarketAnalysisService()

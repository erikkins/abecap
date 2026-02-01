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
        return {
            "regime": self.regime.value,
            "spy_price": _safe_float(self.spy_price),
            "spy_ma_200": _safe_float(self.spy_ma_200),
            "spy_ma_50": _safe_float(self.spy_ma_50),
            "spy_pct_from_high": _safe_float(self.spy_pct_from_high),
            "vix_level": _safe_float(self.vix_level, 20.0),
            "trend_strength": _safe_float(self.trend_strength),
            "recommendation": self.recommendation,
            "updated": self.updated
        }


# Sector ETFs for sector strength analysis
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# Strong sectors typically outperform in bull markets
GROWTH_SECTORS = ["Technology", "Consumer Discretionary", "Communication Services"]
# Defensive sectors typically outperform in bear markets
DEFENSIVE_SECTORS = ["Utilities", "Consumer Staples", "Healthcare"]


class MarketAnalysisService:
    """
    Analyzes market conditions and sector strength
    """

    def __init__(self):
        self.market_state: Optional[MarketState] = None
        self.sector_strength: Dict[str, float] = {}
        self.sector_data: Dict[str, pd.DataFrame] = {}
        self.last_updated: Optional[datetime] = None

    async def update_market_state(self) -> MarketState:
        """
        Update market regime detection

        Uses SPY and VIX to determine bull/bear market conditions.
        """
        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance not installed")

        try:
            # Fetch SPY and VIX data
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: yf.download(["SPY", "^VIX"], period="1y", progress=False)
            )

            spy_close = data['Close']['SPY'].dropna()
            vix_close = data['Close']['^VIX'].dropna()

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

    async def update_sector_strength(self) -> Dict[str, float]:
        """
        Calculate relative strength for each sector

        Returns dict of sector -> strength score (0-100)
        """
        if not YFINANCE_AVAILABLE:
            return {}

        try:
            etfs = list(SECTOR_ETFS.values())

            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: yf.download(etfs, period="3mo", progress=False)
            )

            strength_scores = {}

            for sector, etf in SECTOR_ETFS.items():
                try:
                    close = data['Close'][etf].dropna()
                    if len(close) < 20:
                        continue

                    # Calculate momentum metrics
                    returns_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100
                    returns_3m = (close.iloc[-1] / close.iloc[0] - 1) * 100
                    ma_20 = close.rolling(20).mean().iloc[-1]
                    price_vs_ma = (close.iloc[-1] - ma_20) / ma_20 * 100

                    # Composite strength score (0-100)
                    score = 50 + (returns_1m * 2) + (returns_3m * 1) + (price_vs_ma * 3)
                    score = np.clip(score, 0, 100)

                    strength_scores[sector] = round(score, 1)
                    self.sector_data[sector] = close

                except Exception as e:
                    logger.debug(f"Failed to calculate {sector} strength: {e}")

            # Rank sectors
            self.sector_strength = dict(
                sorted(strength_scores.items(), key=lambda x: x[1], reverse=True)
            )

            logger.info(f"Sector strength updated: Top sector = {list(self.sector_strength.keys())[0] if self.sector_strength else 'N/A'}")

            return self.sector_strength

        except Exception as e:
            logger.error(f"Failed to update sector strength: {e}")
            return {}

    def get_sector_for_stock(self, symbol: str, symbol_info: dict = None) -> Optional[str]:
        """Get sector for a stock symbol"""
        if symbol_info and "sector" in symbol_info:
            return symbol_info.get("sector")
        return None

    def is_sector_strong(self, sector: str, threshold: float = 50) -> bool:
        """Check if a sector is above the strength threshold"""
        return self.sector_strength.get(sector, 50) >= threshold

    def get_weak_sectors(self, threshold: float = 40) -> List[str]:
        """Get list of sectors below the strength threshold"""
        return [s for s, score in self.sector_strength.items() if score < threshold]

    def get_strong_sectors(self, threshold: float = 60) -> List[str]:
        """Get list of sectors above the strength threshold"""
        return [s for s, score in self.sector_strength.items() if score >= threshold]

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
        sector: Optional[str] = None
    ) -> float:
        """
        Calculate composite signal strength score (0-100)

        Factors:
        - % above DWAP (higher = stronger)
        - Volume ratio (higher = stronger)
        - Strong signal flag
        - Sector strength
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

        # Sector component (0-15 points)
        if sector and sector in self.sector_strength:
            sector_score = (self.sector_strength[sector] - 50) / 50 * 15
            score += max(sector_score, 0)

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

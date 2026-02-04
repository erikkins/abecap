"""
Market Regime Detection Service

Multi-factor market regime clustering using:
- SPY trend (50MA, 200MA)
- VIX level and percentile
- Trend strength (momentum/ADX proxy)
- Market breadth

Six distinct regimes with regime-specific parameter adjustments and scoring weights.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np

from app.services.scanner import scanner_service


@dataclass
class MarketConditions:
    """Raw market condition indicators"""
    spy_vs_200ma: float      # % above/below 200MA
    spy_vs_50ma: float       # % above/below 50MA
    vix_level: float         # Current VIX
    vix_percentile: float    # VIX vs 1-year range (0-100)
    trend_strength: float    # Momentum-based trend strength (0-100)
    breadth_pct: float       # % of stocks above 50MA


@dataclass
class MarketRegime:
    """Classified market regime with recommendations"""
    name: str                # "strong_bull", "panic_crash", etc.
    risk_level: str          # "low", "medium", "high", "extreme"
    confidence: float        # 0-1 confidence in classification
    conditions: MarketConditions
    param_adjustments: Dict[str, float]
    scoring_weights: Dict[str, float]


# Six market regime definitions
MARKET_REGIMES = {
    "strong_bull": {
        "conditions": {
            "spy_vs_200ma": (5, 100),      # 5%+ above 200MA
            "spy_vs_50ma": (0, 100),       # Above 50MA
            "vix_level": (0, 18),          # Low VIX
            "trend_strength": (25, 100),   # Strong trend
        },
        "risk_level": "low",
        "param_adjustments": {
            "trailing_stop_pct": -2,       # Tighter stops ok
            "max_positions": +1,           # More positions ok
            "position_size_pct": +2,       # Larger sizes ok
        },
        "scoring_weights": {
            "sharpe_ratio": 0.25,
            "total_return": 0.35,          # Emphasize returns
            "max_drawdown": 0.15,
            "sortino_ratio": 0.15,
            "profit_factor": 0.10,
        }
    },

    "weak_bull": {
        "conditions": {
            "spy_vs_200ma": (0, 10),
            "spy_vs_50ma": (-5, 10),
            "vix_level": (15, 25),
            "trend_strength": (15, 30),
        },
        "risk_level": "medium",
        "param_adjustments": {},           # Use defaults
        "scoring_weights": {
            "sharpe_ratio": 0.30,
            "total_return": 0.25,
            "max_drawdown": 0.20,
            "sortino_ratio": 0.15,
            "profit_factor": 0.10,
        }
    },

    "range_bound": {
        "conditions": {
            "spy_vs_200ma": (-5, 5),
            "spy_vs_50ma": (-5, 5),
            "vix_level": (12, 22),
            "trend_strength": (0, 20),     # Weak trend
        },
        "risk_level": "medium",
        "param_adjustments": {
            "max_positions": -1,
            "near_50d_high_pct": +3,       # Wider entry threshold
        },
        "scoring_weights": {
            "sharpe_ratio": 0.35,          # Emphasize risk-adjusted
            "total_return": 0.15,
            "max_drawdown": 0.20,
            "sortino_ratio": 0.20,
            "profit_factor": 0.10,
        }
    },

    "weak_bear": {
        "conditions": {
            "spy_vs_200ma": (-15, 0),
            "spy_vs_50ma": (-10, 0),
            "vix_level": (20, 30),
            "trend_strength": (15, 35),
        },
        "risk_level": "high",
        "param_adjustments": {
            "trailing_stop_pct": +3,
            "max_positions": -2,
            "position_size_pct": -3,
        },
        "scoring_weights": {
            "sharpe_ratio": 0.20,
            "total_return": 0.15,
            "max_drawdown": 0.35,          # Emphasize drawdown
            "sortino_ratio": 0.20,
            "profit_factor": 0.10,
        }
    },

    "panic_crash": {
        "conditions": {
            "spy_vs_200ma": (-100, -10),
            "vix_level": (30, 100),        # High VIX
            "trend_strength": (30, 100),   # Strong (down) trend
        },
        "risk_level": "extreme",
        "param_adjustments": {
            "trailing_stop_pct": +5,
            "max_positions": -3,
            "position_size_pct": -5,
        },
        "scoring_weights": {
            "sharpe_ratio": 0.15,
            "total_return": 0.10,
            "max_drawdown": 0.40,          # Strongly emphasize protection
            "sortino_ratio": 0.25,
            "profit_factor": 0.10,
        }
    },

    "recovery": {
        "conditions": {
            "spy_vs_200ma": (-5, 5),
            "spy_vs_50ma": (0, 15),        # Breaking above 50MA
            "vix_level": (18, 28),         # Elevated but falling
            "trend_strength": (20, 40),
        },
        "risk_level": "medium",
        "param_adjustments": {
            "trailing_stop_pct": +2,
            "near_50d_high_pct": -2,       # Tighter entry (catch breakout)
        },
        "scoring_weights": {
            "sharpe_ratio": 0.25,
            "total_return": 0.30,
            "max_drawdown": 0.20,
            "sortino_ratio": 0.15,
            "profit_factor": 0.10,
        }
    }
}

# Default scoring weights when no regime detected
DEFAULT_SCORING_WEIGHTS = {
    "sharpe_ratio": 0.30,
    "total_return": 0.25,
    "max_drawdown": 0.25,
    "sortino_ratio": 0.10,
    "profit_factor": 0.10,
}


class MarketRegimeDetector:
    """
    Detects market regime using multiple market factors.

    Uses SPY price data, VIX (if available), and market breadth
    to classify current conditions into one of six regimes.
    """

    def __init__(self):
        self._vix_cache: Dict[str, pd.DataFrame] = {}

    def detect_regime(self, as_of_date: Optional[datetime] = None) -> MarketRegime:
        """
        Detect market regime as of a specific date.

        Args:
            as_of_date: Date to detect regime for (None = current)

        Returns:
            MarketRegime with name, risk level, and recommendations
        """
        conditions = self._calculate_conditions(as_of_date)

        best_regime = None
        best_score = -1
        best_confidence = 0.0

        for regime_name, regime_def in MARKET_REGIMES.items():
            score, confidence = self._score_regime_match(conditions, regime_def["conditions"])
            if score > best_score:
                best_score = score
                best_regime = regime_name
                best_confidence = confidence

        if best_regime is None:
            best_regime = "weak_bull"  # Default fallback
            best_confidence = 0.5

        regime_def = MARKET_REGIMES[best_regime]

        return MarketRegime(
            name=best_regime,
            risk_level=regime_def["risk_level"],
            confidence=round(best_confidence, 2),
            conditions=conditions,
            param_adjustments=regime_def["param_adjustments"],
            scoring_weights=regime_def["scoring_weights"]
        )

    def _calculate_conditions(self, as_of_date: Optional[datetime] = None) -> MarketConditions:
        """Calculate market condition indicators from available data."""
        # Default conditions if no data available
        default = MarketConditions(
            spy_vs_200ma=0,
            spy_vs_50ma=0,
            vix_level=20,
            vix_percentile=50,
            trend_strength=25,
            breadth_pct=50
        )

        if 'SPY' not in scanner_service.data_cache:
            return default

        spy_df = scanner_service.data_cache['SPY']

        # Filter to as_of_date for walk-forward validity
        if as_of_date:
            spy_df = spy_df[spy_df.index <= pd.Timestamp(as_of_date)]

        if len(spy_df) < 200:
            return default

        current_price = spy_df.iloc[-1]['close']
        ma_200 = spy_df['close'].tail(200).mean()
        ma_50 = spy_df['close'].tail(50).mean()

        spy_vs_200ma = (current_price / ma_200 - 1) * 100
        spy_vs_50ma = (current_price / ma_50 - 1) * 100

        # Calculate trend strength (momentum as proxy for ADX)
        returns_20d = (current_price / spy_df.iloc[-20]['close'] - 1) * 100
        trend_strength = min(abs(returns_20d) * 2.5, 100)  # Scale to ~ADX range

        # VIX level (estimate from SPY volatility if VIX not available)
        vix_level = self._get_vix_level(as_of_date)
        vix_percentile = self._get_vix_percentile(vix_level, as_of_date)

        # Market breadth (% of stocks above 50MA)
        breadth_pct = self._calculate_breadth(as_of_date)

        return MarketConditions(
            spy_vs_200ma=round(spy_vs_200ma, 2),
            spy_vs_50ma=round(spy_vs_50ma, 2),
            vix_level=round(vix_level, 1),
            vix_percentile=round(vix_percentile, 1),
            trend_strength=round(trend_strength, 1),
            breadth_pct=round(breadth_pct, 1)
        )

    def _get_vix_level(self, as_of_date: Optional[datetime] = None) -> float:
        """
        Get VIX level. If VIX data not available, estimate from SPY volatility.
        """
        # Check if VIX is in cache
        if '^VIX' in scanner_service.data_cache:
            vix_df = scanner_service.data_cache['^VIX']
            if as_of_date:
                vix_df = vix_df[vix_df.index <= pd.Timestamp(as_of_date)]
            if len(vix_df) > 0:
                return float(vix_df.iloc[-1]['close'])

        # Estimate from SPY volatility
        if 'SPY' not in scanner_service.data_cache:
            return 20.0  # Default

        spy_df = scanner_service.data_cache['SPY']
        if as_of_date:
            spy_df = spy_df[spy_df.index <= pd.Timestamp(as_of_date)]

        if len(spy_df) < 21:
            return 20.0

        # 20-day realized volatility, annualized
        returns = spy_df['close'].pct_change().tail(20)
        volatility = returns.std() * np.sqrt(252) * 100

        return float(volatility)

    def _get_vix_percentile(self, vix_level: float, as_of_date: Optional[datetime] = None) -> float:
        """
        Calculate VIX percentile vs 1-year range.
        """
        if '^VIX' in scanner_service.data_cache:
            vix_df = scanner_service.data_cache['^VIX']
            if as_of_date:
                vix_df = vix_df[vix_df.index <= pd.Timestamp(as_of_date)]
            if len(vix_df) >= 252:
                year_data = vix_df['close'].tail(252)
                percentile = (year_data < vix_level).sum() / len(year_data) * 100
                return float(percentile)

        # Estimate percentile based on typical VIX ranges
        # Historical VIX median ~17, typical range 12-30
        if vix_level < 12:
            return 5.0
        elif vix_level < 15:
            return 20.0
        elif vix_level < 18:
            return 40.0
        elif vix_level < 22:
            return 55.0
        elif vix_level < 28:
            return 75.0
        elif vix_level < 35:
            return 90.0
        else:
            return 98.0

    def _calculate_breadth(self, as_of_date: Optional[datetime] = None) -> float:
        """
        Calculate market breadth: % of stocks above their 50-day MA.
        """
        if not scanner_service.data_cache:
            return 50.0

        above_50ma = 0
        total = 0

        for symbol, df in scanner_service.data_cache.items():
            if symbol.startswith('^'):  # Skip indices
                continue

            if as_of_date:
                df = df[df.index <= pd.Timestamp(as_of_date)]

            if len(df) < 50:
                continue

            current_price = df.iloc[-1]['close']
            ma_50 = df['close'].tail(50).mean()

            total += 1
            if current_price > ma_50:
                above_50ma += 1

        if total == 0:
            return 50.0

        return (above_50ma / total) * 100

    def _score_regime_match(
        self,
        conditions: MarketConditions,
        regime_conditions: Dict[str, Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Score how well current conditions match a regime's expected ranges.

        Returns:
            Tuple of (match_score, confidence)
        """
        total_score = 0
        total_weight = 0
        matches = 0
        total_conditions = 0

        condition_values = {
            "spy_vs_200ma": conditions.spy_vs_200ma,
            "spy_vs_50ma": conditions.spy_vs_50ma,
            "vix_level": conditions.vix_level,
            "trend_strength": conditions.trend_strength,
            "breadth_pct": conditions.breadth_pct,
        }

        for cond_name, (min_val, max_val) in regime_conditions.items():
            if cond_name not in condition_values:
                continue

            value = condition_values[cond_name]
            total_conditions += 1

            # Check if value is within range
            if min_val <= value <= max_val:
                matches += 1
                # Score based on how centered the value is in the range
                range_size = max_val - min_val
                if range_size > 0:
                    center = (min_val + max_val) / 2
                    distance_from_center = abs(value - center) / (range_size / 2)
                    score = 1 - distance_from_center * 0.5  # 0.5-1.0 range
                else:
                    score = 1.0
                total_score += score
                total_weight += 1
            else:
                # Penalize for being outside range
                if value < min_val:
                    distance = (min_val - value) / max(abs(min_val), 1)
                else:
                    distance = (value - max_val) / max(abs(max_val), 1)
                penalty = min(distance * 0.3, 0.5)  # Max 50% penalty
                total_score -= penalty
                total_weight += 1

        if total_weight == 0:
            return 0, 0

        # Calculate confidence based on how many conditions matched
        confidence = matches / total_conditions if total_conditions > 0 else 0

        return total_score / total_weight, confidence

    def get_regime_summary(self, as_of_date: Optional[datetime] = None) -> Dict:
        """
        Get a summary of regime detection for display.
        """
        regime = self.detect_regime(as_of_date)

        return {
            "name": regime.name,
            "display_name": regime.name.replace("_", " ").title(),
            "risk_level": regime.risk_level,
            "confidence": regime.confidence,
            "conditions": {
                "spy_vs_200ma": regime.conditions.spy_vs_200ma,
                "spy_vs_50ma": regime.conditions.spy_vs_50ma,
                "vix_level": regime.conditions.vix_level,
                "vix_percentile": regime.conditions.vix_percentile,
                "trend_strength": regime.conditions.trend_strength,
                "breadth_pct": regime.conditions.breadth_pct,
            },
            "param_adjustments": regime.param_adjustments,
            "scoring_weights": regime.scoring_weights,
        }


# Singleton instance
market_regime_detector = MarketRegimeDetector()

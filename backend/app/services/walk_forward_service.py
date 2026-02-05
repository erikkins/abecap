"""
Walk-Forward Simulation Service

Simulates the auto-switch logic over a historical period to evaluate
how automated strategy switching would have performed.

Enhanced with AI optimization at each reoptimization period to detect
emerging trends and adapt parameters dynamically.
"""

import json
import itertools
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import StrategyDefinition, WalkForwardSimulation
from app.services.backtester import BacktesterService
from app.services.strategy_analyzer import StrategyAnalyzerService, CustomBacktester, StrategyParams, get_top_liquid_symbols
from app.services.scanner import scanner_service
from app.services.market_regime import market_regime_service, MarketRegime, REGIME_DEFINITIONS
from app.services.scanner import scanner_service as regime_scanner_service


@dataclass
class AIOptimizationResult:
    """Result from AI optimization at a reoptimization point"""
    date: str
    best_params: Dict[str, Any]
    expected_sharpe: float
    expected_return_pct: float
    strategy_type: str
    market_regime: str
    was_adopted: bool
    reason: str
    # Enhanced metrics
    expected_sortino: float = 0.0
    expected_calmar: float = 0.0
    expected_profit_factor: float = 0.0
    expected_max_dd: float = 0.0
    combinations_tested: int = 0
    regime_confidence: float = 0.0
    regime_risk_level: str = "medium"
    adaptive_score: float = 0.0


@dataclass
class ParameterSnapshot:
    """Snapshot of active parameters at a point in time"""
    date: str
    strategy_name: str
    strategy_type: str
    params: Dict[str, Any]
    source: str  # "existing" or "ai_generated"


@dataclass
class SimulationPeriod:
    """A single period in the walk-forward simulation"""
    start_date: datetime
    end_date: datetime
    active_strategy_id: Optional[int]
    active_strategy_name: str
    period_return_pct: float
    cumulative_equity: float
    ai_optimization: Optional[AIOptimizationResult] = None


@dataclass
class SwitchEvent:
    """Record of a strategy switch during simulation"""
    date: str
    from_strategy_id: Optional[int]
    from_strategy_name: Optional[str]
    to_strategy_id: Optional[int]  # None if AI-generated
    to_strategy_name: str
    reason: str
    score_before: Optional[float]
    score_after: float
    is_ai_generated: bool = False
    ai_params: Optional[Dict[str, Any]] = None


@dataclass
class PeriodTrade:
    """A trade executed during walk-forward simulation"""
    period_start: str
    period_end: str
    strategy_name: str
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: float
    pnl_pct: float
    pnl_dollars: float
    exit_reason: str


@dataclass
class WalkForwardResult:
    """Complete walk-forward simulation result"""
    start_date: str
    end_date: str
    reoptimization_frequency: str
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_strategy_switches: int
    benchmark_return_pct: float
    switch_history: List[SwitchEvent]
    equity_curve: List[Dict]
    period_details: List[SimulationPeriod]
    ai_optimizations: List[AIOptimizationResult] = field(default_factory=list)
    parameter_evolution: List[ParameterSnapshot] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)  # Simulation errors for debugging
    trades: List[PeriodTrade] = field(default_factory=list)  # All trades across all periods


class WalkForwardService:
    """
    Walk-forward analysis service with AI optimization.

    Simulates what would have happened if we ran the auto-switch logic
    throughout a historical period, making decisions only with data
    available at each point in time.

    Enhanced features:
    - AI parameter optimization at each reoptimization period
    - Multi-factor market regime detection (6 regimes)
    - Expanded parameter grid with smart reduction
    - Adaptive scoring based on market regime
    - Enhanced metrics (Sortino, Calmar, Profit Factor, etc.)
    """

    # Expanded parameter ranges for AI optimization
    EXPANDED_AI_PARAM_RANGES = {
        "momentum": {
            # Risk Management
            "trailing_stop_pct": [10, 12, 15, 18, 20],
            "max_positions": [3, 4, 5, 6, 7],
            "position_size_pct": [12, 15, 18, 20],
            # Entry Timing
            "short_momentum_days": [5, 10, 15],
            "near_50d_high_pct": [3, 5, 7, 10],
        },
        "dwap": {
            "dwap_threshold_pct": [3, 5, 7],
            "stop_loss_pct": [6, 8, 10, 12],
            "profit_target_pct": [15, 20, 25, 30],
            "max_positions": [10, 15, 20],
            "position_size_pct": [5, 6, 7],
        }
    }

    # Coarse grid for Phase 1 (key parameters only)
    COARSE_PARAM_RANGES = {
        "momentum": {
            "trailing_stop_pct": [12, 15, 18],
            "max_positions": [4, 5, 6],
            "position_size_pct": [15, 18],
        },
        "dwap": {
            "stop_loss_pct": [7, 8, 9],
            "profit_target_pct": [18, 20, 22],
        }
    }

    def __init__(self):
        self.analyzer = StrategyAnalyzerService()
        self.initial_capital = 100000

    def _get_period_dates(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: str
    ) -> List[Tuple[datetime, datetime]]:
        """
        Generate list of (period_start, period_end) tuples.

        Args:
            start_date: Simulation start
            end_date: Simulation end
            frequency: "weekly", "biweekly", or "monthly"

        Returns:
            List of (start, end) tuples for each reoptimization period
        """
        periods = []
        current_start = start_date

        if frequency in ("weekly", "fast"):
            delta = timedelta(days=7)
        elif frequency == "biweekly":
            delta = timedelta(days=14)
        else:  # monthly
            delta = timedelta(days=30)

        while current_start < end_date:
            period_end = min(current_start + delta, end_date)
            periods.append((current_start, period_end))
            current_start = period_end

        return periods

    def _evaluate_strategy_at_date(
        self,
        strategy: StrategyDefinition,
        as_of_date: datetime,
        lookback_days: int = 90,
        ticker_list: List[str] = None
    ) -> Optional[Dict]:
        """
        Evaluate a strategy using only data available up to as_of_date.

        This is crucial for walk-forward validity - we can only use
        information that was available at the decision point.
        """
        params = StrategyParams.from_json(strategy.parameters)

        backtester = CustomBacktester()
        backtester.configure(params)

        try:
            result = backtester.run_backtest(
                lookback_days=lookback_days,
                end_date=as_of_date,
                strategy_type=strategy.strategy_type,
                ticker_list=ticker_list
            )

            return {
                "strategy_id": strategy.id,
                "name": strategy.name,
                "sharpe_ratio": result.sharpe_ratio,
                "total_return_pct": result.total_return_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "win_rate": result.win_rate,
            }
        except Exception as e:
            return None

    def _calculate_recommendation_score(self, metrics: Dict) -> float:
        """Calculate composite score (same logic as strategy analyzer)"""
        sharpe = metrics.get('sharpe_ratio', 0)
        total_return = metrics.get('total_return_pct', 0)
        max_drawdown = metrics.get('max_drawdown_pct', 0)

        sharpe_score = min(max(sharpe / 2, 0), 1) * 40
        return_score = min(max(total_return / 50, 0), 1) * 30
        dd_score = max(1 - max_drawdown / 20, 0) * 30

        return sharpe_score + return_score + dd_score

    def _calculate_adaptive_score(
        self,
        metrics: Dict[str, float],
        regime: MarketRegime
    ) -> float:
        """
        Calculate score using regime-specific weights.

        All metrics normalized to 0-1 scale, then weighted based on
        current market regime priorities.
        """
        weights = regime.scoring_weights

        # Normalize each metric to 0-1 scale
        normalized = {
            "sharpe_ratio": min(max(metrics.get("sharpe_ratio", 0) / 2, 0), 1),
            "total_return": min(max(metrics.get("total_return_pct", 0) / 50, 0), 1),
            "max_drawdown": max(1 - metrics.get("max_drawdown_pct", 0) / 25, 0),
            "sortino_ratio": min(max(metrics.get("sortino_ratio", 0) / 2.5, 0), 1),
            "profit_factor": min(max((metrics.get("profit_factor", 1) - 1) / 2, 0), 1),
            "calmar_ratio": min(max(metrics.get("calmar_ratio", 0) / 3, 0), 1),
        }

        # Weighted sum
        score = sum(
            normalized.get(metric, 0) * weight
            for metric, weight in weights.items()
        ) * 100  # Scale to 0-100

        return round(score, 2)

    def _detect_market_regime_at_date(self, as_of_date: datetime) -> MarketRegime:
        """
        Detect market regime using multi-factor analysis.

        Uses SPY trend, VIX, trend strength, and breadth to classify
        into one of seven regimes: strong_bull, weak_bull, rotating_bull,
        range_bound, weak_bear, panic_crash, recovery.

        Returns: MarketRegime object with name, risk level, and recommendations
        """
        spy_df = scanner_service.data_cache.get('SPY')
        vix_df = scanner_service.data_cache.get('^VIX')

        if spy_df is None or len(spy_df) < 200:
            # Return a default regime if no data
            from app.services.market_regime import RegimeType, MarketConditions
            return MarketRegime(
                date=as_of_date.strftime('%Y-%m-%d'),
                regime_type=RegimeType.RANGE_BOUND,
                regime_name="Range-Bound",
                risk_level="medium",
                confidence=50.0,
                conditions=None,
                param_adjustments={},
                scoring_weights=REGIME_DEFINITIONS[RegimeType.RANGE_BOUND]["scoring_weights"],
                description="Insufficient data for regime detection"
            )

        return market_regime_service.detect_regime(
            spy_df=spy_df,
            universe_dfs=scanner_service.data_cache,
            vix_df=vix_df,
            as_of_date=as_of_date
        )

    def _run_ai_optimization_at_date(
        self,
        as_of_date: datetime,
        strategy_type: str,
        lookback_days: int,
        ticker_list: List[str]
    ) -> Optional[AIOptimizationResult]:
        """
        Run AI parameter optimization using only data available at as_of_date.

        Uses two-phase smart grid optimization:
        Phase 1: Coarse grid on key parameters (~18 combinations)
        Phase 2: Fine-tune around best values (~30 additional combinations)

        Applies regime-specific scoring weights for adaptive optimization.
        """
        regime = self._detect_market_regime_at_date(as_of_date)

        # Build base params
        if strategy_type == "momentum":
            base_params = {
                "short_momentum_days": 10,
                "long_momentum_days": 60,
                "trailing_stop_pct": 15.0,
                "max_positions": 5,
                "position_size_pct": 18.0,
                "min_volume": 500_000,
                "min_price": 20.0,
                "market_filter_enabled": True,
                "near_50d_high_pct": 5.0,
            }
        else:
            base_params = {
                "dwap_threshold_pct": 5.0,
                "stop_loss_pct": 8.0,
                "profit_target_pct": 20.0,
                "max_positions": 15,
                "position_size_pct": 6.0,
                "min_volume": 500_000,
                "min_price": 20.0,
            }

        # Apply regime-specific parameter adjustments to base
        for param, adjustment in regime.param_adjustments.items():
            if param in base_params:
                base_params[param] = base_params[param] + adjustment

        # Phase 1: Coarse grid optimization
        coarse_ranges = self._get_regime_adjusted_ranges(
            self.COARSE_PARAM_RANGES.get(strategy_type, {}),
            regime
        )

        keys = list(coarse_ranges.keys())
        values = [coarse_ranges[k] for k in keys]
        phase1_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        best_result = None
        best_score = -float('inf')
        best_combo = None
        combinations_tested = 0

        # Test Phase 1 combinations
        for combo in phase1_combinations:
            test_params = {**base_params, **combo}
            result_data = self._test_param_combination(
                test_params, strategy_type, as_of_date, lookback_days, ticker_list, regime
            )
            combinations_tested += 1

            if result_data and result_data["adaptive_score"] > best_score:
                best_score = result_data["adaptive_score"]
                best_result = result_data
                best_combo = combo

        # Phase 2: Fine-tune around best values
        if best_combo and strategy_type == "momentum":
            fine_tune_ranges = self._get_fine_tune_ranges(best_combo, strategy_type)

            for param_key, values_list in fine_tune_ranges.items():
                for value in values_list:
                    test_params = {**base_params, **best_combo, param_key: value}
                    result_data = self._test_param_combination(
                        test_params, strategy_type, as_of_date, lookback_days, ticker_list, regime
                    )
                    combinations_tested += 1

                    if result_data and result_data["adaptive_score"] > best_score:
                        best_score = result_data["adaptive_score"]
                        best_result = result_data

        if best_result:
            return AIOptimizationResult(
                date=as_of_date.strftime('%Y-%m-%d'),
                best_params=best_result["full_params"],
                expected_sharpe=best_result["sharpe_ratio"],
                expected_return_pct=best_result["total_return_pct"],
                strategy_type=strategy_type,
                market_regime=regime.regime_type.value,
                was_adopted=False,
                reason="",
                expected_sortino=best_result.get("sortino_ratio", 0),
                expected_calmar=best_result.get("calmar_ratio", 0),
                expected_profit_factor=best_result.get("profit_factor", 0),
                expected_max_dd=best_result.get("max_drawdown_pct", 0),
                combinations_tested=combinations_tested,
                regime_confidence=regime.confidence,
                regime_risk_level=regime.risk_level,
                adaptive_score=best_score
            )

        return None

    def _get_regime_adjusted_ranges(
        self,
        base_ranges: Dict[str, List],
        regime: MarketRegime
    ) -> Dict[str, List]:
        """
        Adjust parameter ranges based on market regime.

        For high-risk regimes, bias toward more conservative values.
        For low-risk regimes, allow more aggressive values.
        """
        adjusted = {}

        for param, values in base_ranges.items():
            if regime.risk_level == "extreme":
                # In panic/crash, prefer conservative (higher stops, fewer positions)
                if param in ["trailing_stop_pct", "stop_loss_pct"]:
                    adjusted[param] = [v for v in values if v >= 15] or values[-2:]
                elif param == "max_positions":
                    adjusted[param] = [v for v in values if v <= 4] or values[:2]
                else:
                    adjusted[param] = values
            elif regime.risk_level == "high":
                # Weak bear - slightly conservative
                if param in ["trailing_stop_pct", "stop_loss_pct"]:
                    adjusted[param] = [v for v in values if v >= 12] or values[-3:]
                elif param == "max_positions":
                    adjusted[param] = [v for v in values if v <= 5] or values[:3]
                else:
                    adjusted[param] = values
            elif regime.risk_level == "low":
                # Strong bull - can be more aggressive
                if param in ["trailing_stop_pct", "stop_loss_pct"]:
                    adjusted[param] = [v for v in values if v <= 15] or values[:3]
                elif param == "max_positions":
                    adjusted[param] = [v for v in values if v >= 5] or values[-3:]
                else:
                    adjusted[param] = values
            else:
                adjusted[param] = values

        return adjusted

    def _get_fine_tune_ranges(
        self,
        best_combo: Dict[str, Any],
        strategy_type: str
    ) -> Dict[str, List]:
        """
        Generate fine-tuning ranges around the best Phase 1 values.

        Includes additional parameters not in Phase 1.
        """
        fine_tune = {}
        expanded = self.EXPANDED_AI_PARAM_RANGES.get(strategy_type, {})

        for param, full_values in expanded.items():
            if param not in best_combo:
                # New parameter - test a subset
                fine_tune[param] = full_values[:3] if len(full_values) > 3 else full_values
            else:
                # Existing parameter - test adjacent values
                best_val = best_combo[param]
                if best_val in full_values:
                    idx = full_values.index(best_val)
                    # Get values ±1 step around best
                    adjacent = []
                    if idx > 0:
                        adjacent.append(full_values[idx - 1])
                    if idx < len(full_values) - 1:
                        adjacent.append(full_values[idx + 1])
                    if adjacent:
                        fine_tune[param] = adjacent

        return fine_tune

    def _test_param_combination(
        self,
        test_params: Dict[str, Any],
        strategy_type: str,
        as_of_date: datetime,
        lookback_days: int,
        ticker_list: List[str],
        regime: MarketRegime
    ) -> Optional[Dict]:
        """
        Test a single parameter combination and return metrics with adaptive score.
        """
        try:
            params = StrategyParams(**test_params)
            backtester = CustomBacktester()
            backtester.configure(params)

            result = backtester.run_backtest(
                lookback_days=lookback_days,
                end_date=as_of_date,
                strategy_type=strategy_type,
                ticker_list=ticker_list
            )

            metrics = {
                "sharpe_ratio": result.sharpe_ratio,
                "total_return_pct": result.total_return_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "sortino_ratio": result.sortino_ratio,
                "calmar_ratio": result.calmar_ratio,
                "profit_factor": result.profit_factor,
                "win_rate": result.win_rate,
            }

            adaptive_score = self._calculate_adaptive_score(metrics, regime)

            return {
                "full_params": test_params,
                "sharpe_ratio": result.sharpe_ratio,
                "total_return_pct": result.total_return_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "sortino_ratio": result.sortino_ratio,
                "calmar_ratio": result.calmar_ratio,
                "profit_factor": result.profit_factor,
                "adaptive_score": adaptive_score,
            }
        except Exception:
            return None

    def _simulate_period_with_params(
        self,
        params: Dict[str, Any],
        strategy_type: str,
        start_date: datetime,
        end_date: datetime,
        starting_capital: float,
        ticker_list: List[str] = None,
        strategy_name: str = "AI-Optimized"
    ) -> Tuple[float, float, str, List[PeriodTrade]]:
        """
        Simulate trading for a period using custom parameters (for AI-generated strategies).

        Returns:
            Tuple of (ending_capital, period_return_pct, info_string, trades)
        """
        try:
            strategy_params = StrategyParams(**params)
            backtester = CustomBacktester()
            backtester.configure(strategy_params)
            backtester.initial_capital = starting_capital

            result = backtester.run_backtest(
                start_date=start_date,
                end_date=end_date,
                strategy_type=strategy_type,
                ticker_list=ticker_list
            )

            ending_capital = starting_capital * (1 + result.total_return_pct / 100)

            # Use debug_info from BacktestResult if available
            backtest_debug = getattr(result, 'debug_info', '') or ''

            print(f"[WF-SIM] Period {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}: "
                  f"{result.total_return_pct:.2f}% return, {result.total_trades} trades. {backtest_debug}")

            # Convert trades to PeriodTrade objects
            period_trades = [
                PeriodTrade(
                    period_start=start_date.strftime('%Y-%m-%d'),
                    period_end=end_date.strftime('%Y-%m-%d'),
                    strategy_name=strategy_name,
                    symbol=t.symbol,
                    entry_date=t.entry_date,
                    exit_date=t.exit_date,
                    entry_price=round(t.entry_price, 2),
                    exit_price=round(t.exit_price, 2),
                    shares=round(t.shares, 2),
                    pnl_pct=round(t.pnl_pct, 2),
                    pnl_dollars=round(t.pnl, 2),
                    exit_reason=t.exit_reason
                )
                for t in result.trades
            ]

            # Always return info about the period for debugging
            info = (f"Period {start_date.strftime('%Y-%m-%d')}: {backtest_debug}")
            return ending_capital, result.total_return_pct, info, period_trades
        except Exception as e:
            error_msg = f"Period {start_date.strftime('%Y-%m-%d')}: ERROR {str(e)}"
            print(f"[WF-SIM] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            return starting_capital, 0.0, error_msg, []

    def _simulate_period_trading(
        self,
        strategy: StrategyDefinition,
        start_date: datetime,
        end_date: datetime,
        starting_capital: float,
        ticker_list: List[str] = None
    ) -> Tuple[float, List[Dict], float, str, List[PeriodTrade]]:
        """
        Simulate trading for a single period using a specific strategy.

        Returns:
            Tuple of (ending_capital, equity_points, period_return_pct, info, trades)
        """
        params = StrategyParams.from_json(strategy.parameters)

        backtester = CustomBacktester()
        backtester.configure(params)
        backtester.initial_capital = starting_capital

        try:
            result = backtester.run_backtest(
                start_date=start_date,
                end_date=end_date,
                strategy_type=strategy.strategy_type,
                ticker_list=ticker_list
            )

            ending_capital = starting_capital * (1 + result.total_return_pct / 100)
            print(f"[WF-SIM] Period {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}: {result.total_return_pct:.2f}% return, {result.total_trades} trades (strategy: {strategy.name})")
            # Always return info about the period for debugging
            info = f"Period {start_date.strftime('%Y-%m-%d')}: {result.total_trades} trades, {result.total_return_pct:.1f}% ({strategy.name})"

            # Build equity curve points for this period
            # We don't have detailed daily data, so approximate
            equity_points = [
                {
                    "date": result.start_date,
                    "equity": starting_capital,
                    "strategy": strategy.name
                },
                {
                    "date": result.end_date,
                    "equity": ending_capital,
                    "strategy": strategy.name
                }
            ]

            # Convert trades to PeriodTrade objects
            period_trades = [
                PeriodTrade(
                    period_start=start_date.strftime('%Y-%m-%d'),
                    period_end=end_date.strftime('%Y-%m-%d'),
                    strategy_name=strategy.name,
                    symbol=t.symbol,
                    entry_date=t.entry_date,
                    exit_date=t.exit_date,
                    entry_price=round(t.entry_price, 2),
                    exit_price=round(t.exit_price, 2),
                    shares=round(t.shares, 2),
                    pnl_pct=round(t.pnl_pct, 2),
                    pnl_dollars=round(t.pnl, 2),
                    exit_reason=t.exit_reason
                )
                for t in result.trades
            ]

            return ending_capital, equity_points, result.total_return_pct, info, period_trades

        except Exception as e:
            # If backtest fails, assume flat return
            error_msg = f"Period {start_date.strftime('%Y-%m-%d')}: {str(e)}"
            print(f"[WF-SIM] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            return starting_capital, [], 0.0, error_msg, []

    async def run_walk_forward_simulation(
        self,
        db: AsyncSession,
        start_date: datetime,
        end_date: datetime,
        reoptimization_frequency: str = "biweekly",
        min_score_diff: float = 10.0,
        lookback_days: int = 60,
        enable_ai_optimization: bool = True,
        max_symbols: int = 50,
        existing_job_id: int = None
    ) -> WalkForwardResult:
        """
        Run walk-forward simulation with AI optimization over a historical period.

        Enhanced Algorithm:
        1. Start at start_date with the best strategy/params at that point
        2. Every reoptimization period:
           a. Evaluate existing strategies using ONLY data available at that point
           b. Run AI optimization to find optimal params for current conditions
           c. Compare AI-optimized params against existing strategies
           d. Switch to best option if it beats current by min_score_diff
           e. Track parameter evolution and AI recommendations
           f. Simulate trading until next period
        3. Return full equity curve, switch timeline, AI recommendations, and param evolution

        Args:
            db: Database session
            start_date: Simulation start date
            end_date: Simulation end date
            reoptimization_frequency: "weekly", "biweekly", or "monthly"
            min_score_diff: Minimum score difference to trigger switch
            lookback_days: Days of data for strategy evaluation
            enable_ai_optimization: Whether to run AI param optimization each period

        Returns:
            WalkForwardResult with complete simulation data including AI insights
        """
        import logging
        logger = logging.getLogger()
        print(f"[WF-SERVICE] Starting simulation: {start_date} to {end_date}, ai={enable_ai_optimization}")

        # Load all strategies
        result = await db.execute(
            select(StrategyDefinition).order_by(StrategyDefinition.id)
        )
        strategies = result.scalars().all()
        print(f"[WF-SERVICE] Loaded {len(strategies)} strategies")

        if not strategies:
            raise RuntimeError("No strategies found in database")

        # Get top liquid symbols (use max_symbols param for full universe testing)
        top_symbols = get_top_liquid_symbols(max_symbols=max_symbols)
        print(f"[WF-SERVICE] Got {len(top_symbols) if top_symbols else 0} top symbols")
        if not top_symbols:
            raise RuntimeError("No liquid symbols found. Ensure data is loaded.")

        # Get period boundaries
        periods = self._get_period_dates(start_date, end_date, reoptimization_frequency)
        print(f"[WF-SERVICE] Processing {len(periods)} periods")

        # Initialize simulation state
        capital = self.initial_capital
        active_strategy = strategies[0]  # Will be replaced by first analysis
        active_strategy_score = 0.0
        active_strategy_type = strategies[0].strategy_type
        active_params: Optional[Dict] = None  # None means using existing strategy
        using_ai_params = False

        switch_history: List[SwitchEvent] = []
        equity_curve: List[Dict] = []
        period_details: List[SimulationPeriod] = []
        ai_optimizations: List[AIOptimizationResult] = []
        parameter_evolution: List[ParameterSnapshot] = []
        simulation_errors: List[str] = []  # Track errors for debugging
        all_trades: List[PeriodTrade] = []  # All trades across all periods

        # Get SPY starting price for benchmark line
        spy_start_price = None
        spy_df = None
        if 'SPY' in scanner_service.data_cache:
            spy_df = scanner_service.data_cache['SPY']
            start_ts = pd.Timestamp(start_date)
            # Handle timezone-aware index
            if spy_df.index.tz is not None:
                start_ts = start_ts.tz_localize(spy_df.index.tz) if start_ts.tz is None else start_ts.tz_convert(spy_df.index.tz)
            spy_at_start = spy_df[spy_df.index >= start_ts]
            if len(spy_at_start) > 0:
                spy_start_price = spy_at_start.iloc[0]['close']
                print(f"[WF-SERVICE] SPY start price: ${spy_start_price:.2f}")

        def get_spy_equity(date_str: str) -> float:
            """Get SPY equity normalized to initial capital"""
            if spy_start_price is None or spy_df is None:
                return None
            try:
                date_ts = pd.Timestamp(date_str)
                # Handle timezone-aware index
                if spy_df.index.tz is not None:
                    date_ts = date_ts.tz_localize(spy_df.index.tz) if date_ts.tz is None else date_ts.tz_convert(spy_df.index.tz)
                spy_at_date = spy_df[spy_df.index <= date_ts]
                if len(spy_at_date) > 0:
                    spy_price = spy_at_date.iloc[-1]['close']
                    return self.initial_capital * (spy_price / spy_start_price)
            except:
                pass
            return None

        # Add initial point
        equity_curve.append({
            "date": start_date.strftime('%Y-%m-%d'),
            "equity": capital,
            "spy_equity": self.initial_capital,
            "strategy": "Initial",
            "is_switch": False
        })

        # Process each period
        print(f"[WF-SERVICE] Starting simulation loop: {len(periods)} periods, initial capital=${capital:,.2f}")
        for i, (period_start, period_end) in enumerate(periods):
            print(f"[WF-SERVICE] Period {i+1}/{len(periods)}: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
            period_ai_opt = None

            # Step 1: Evaluate all existing strategies
            evaluations = []
            for strategy in strategies:
                metrics = self._evaluate_strategy_at_date(
                    strategy, period_start, lookback_days, ticker_list=top_symbols
                )
                if metrics:
                    metrics["score"] = self._calculate_recommendation_score(metrics)
                    metrics["is_ai"] = False
                    evaluations.append(metrics)

            # Step 2: Run AI optimization if enabled
            if enable_ai_optimization:
                try:
                    print(f"[WF-SERVICE] Running AI optimization for period {i+1}/{len(periods)}")
                    # Run for primary strategy type (momentum)
                    ai_result = self._run_ai_optimization_at_date(
                        period_start, "momentum", lookback_days, top_symbols
                    )
                    if ai_result:
                        period_ai_opt = ai_result
                        # Add AI result to evaluations for comparison
                        # Use adaptive score for AI, calculated with regime-specific weights
                        evaluations.append({
                            "strategy_id": None,
                            "name": f"AI-{ai_result.market_regime.replace('_', '-').title()}",
                            "sharpe_ratio": ai_result.expected_sharpe,
                            "total_return_pct": ai_result.expected_return_pct,
                            "max_drawdown_pct": ai_result.expected_max_dd,
                            "sortino_ratio": ai_result.expected_sortino,
                            "calmar_ratio": ai_result.expected_calmar,
                            "profit_factor": ai_result.expected_profit_factor,
                            "score": ai_result.adaptive_score,  # Use adaptive score
                            "is_ai": True,
                            "ai_params": ai_result.best_params,
                            "market_regime": ai_result.market_regime,
                            "regime_risk_level": ai_result.regime_risk_level,
                            "regime_confidence": ai_result.regime_confidence,
                            "combinations_tested": ai_result.combinations_tested
                        })
                        print(f"[WF-SERVICE] AI optimization complete: regime={ai_result.market_regime} "
                              f"(risk={ai_result.regime_risk_level}, conf={ai_result.regime_confidence:.0%}), "
                              f"tested={ai_result.combinations_tested}, adaptive_score={ai_result.adaptive_score:.1f}")
                except Exception as ai_err:
                    print(f"[WF-SERVICE] AI optimization failed for period {i+1}: {ai_err}")
                    import traceback
                    print(f"[WF-SERVICE] AI traceback: {traceback.format_exc()}")
                    # Continue without AI for this period

            if evaluations:
                # Find best option (existing strategy or AI-optimized)
                best = max(evaluations, key=lambda x: x["score"])
                score_diff = best["score"] - active_strategy_score

                # Determine if we should switch
                should_switch = False
                switch_reason = ""

                if i == 0:
                    # First period - always pick best
                    should_switch = True
                    switch_reason = "initial_selection"
                elif score_diff >= min_score_diff:
                    # Score improvement meets threshold
                    if best.get("is_ai"):
                        if not using_ai_params:
                            should_switch = True
                            switch_reason = f"ai_optimization_+{score_diff:.1f}pts"
                    elif best["strategy_id"] != (active_strategy.id if active_strategy else None):
                        should_switch = True
                        switch_reason = f"strategy_switch_+{score_diff:.1f}pts"

                if should_switch:
                    if best.get("is_ai"):
                        # Switching to AI-optimized params
                        switch_history.append(SwitchEvent(
                            date=period_start.strftime('%Y-%m-%d'),
                            from_strategy_id=active_strategy.id if active_strategy else None,
                            from_strategy_name=active_strategy.name if active_strategy else "AI-Params",
                            to_strategy_id=None,
                            to_strategy_name=best["name"],
                            reason=switch_reason,
                            score_before=active_strategy_score,
                            score_after=best["score"],
                            is_ai_generated=True,
                            ai_params=best.get("ai_params")
                        ))
                        active_params = best.get("ai_params")
                        active_strategy_type = "momentum"
                        using_ai_params = True
                        active_strategy = None
                        active_strategy_score = best["score"]

                        if period_ai_opt:
                            period_ai_opt.was_adopted = True
                            period_ai_opt.reason = switch_reason
                    else:
                        # Switching to existing strategy
                        best_strategy = next(s for s in strategies if s.id == best["strategy_id"])
                        switch_history.append(SwitchEvent(
                            date=period_start.strftime('%Y-%m-%d'),
                            from_strategy_id=active_strategy.id if active_strategy else None,
                            from_strategy_name=active_strategy.name if active_strategy else "AI-Params",
                            to_strategy_id=best_strategy.id,
                            to_strategy_name=best_strategy.name,
                            reason=switch_reason,
                            score_before=active_strategy_score,
                            score_after=best["score"],
                            is_ai_generated=False
                        ))
                        active_strategy = best_strategy
                        active_strategy_score = best["score"]
                        active_params = None
                        using_ai_params = False

                        if period_ai_opt:
                            period_ai_opt.was_adopted = False
                            period_ai_opt.reason = "existing_strategy_better"

            # Record AI optimization result
            if period_ai_opt:
                ai_optimizations.append(period_ai_opt)

            # Record parameter snapshot
            if using_ai_params and active_params:
                parameter_evolution.append(ParameterSnapshot(
                    date=period_start.strftime('%Y-%m-%d'),
                    strategy_name="AI-Optimized",
                    strategy_type=active_strategy_type,
                    params=active_params,
                    source="ai_generated"
                ))
            elif active_strategy:
                parameter_evolution.append(ParameterSnapshot(
                    date=period_start.strftime('%Y-%m-%d'),
                    strategy_name=active_strategy.name,
                    strategy_type=active_strategy.strategy_type,
                    params=json.loads(active_strategy.parameters),
                    source="existing"
                ))

            # Simulate trading for this period
            if using_ai_params and active_params:
                new_capital, period_return, error, period_trades = self._simulate_period_with_params(
                    active_params, active_strategy_type, period_start, period_end,
                    capital, ticker_list=top_symbols, strategy_name="AI-Optimized"
                )
                strategy_name = "AI-Optimized"
                if error:
                    simulation_errors.append(error)
                all_trades.extend(period_trades)
            else:
                new_capital, period_equity, period_return, error, period_trades = self._simulate_period_trading(
                    active_strategy, period_start, period_end, capital, ticker_list=top_symbols
                )
                strategy_name = active_strategy.name
                if error:
                    simulation_errors.append(error)
                all_trades.extend(period_trades)

            period_details.append(SimulationPeriod(
                start_date=period_start,
                end_date=period_end,
                active_strategy_id=active_strategy.id if active_strategy else None,
                active_strategy_name=strategy_name,
                period_return_pct=period_return,
                cumulative_equity=new_capital,
                ai_optimization=period_ai_opt
            ))

            # Determine if this is a switch point for the chart
            is_switch = len(switch_history) > 0 and switch_history[-1].date == period_start.strftime('%Y-%m-%d')

            # Add to equity curve with strategy info and SPY benchmark
            date_str = period_end.strftime('%Y-%m-%d')
            equity_curve.append({
                "date": date_str,
                "equity": new_capital,
                "spy_equity": get_spy_equity(date_str),
                "strategy": strategy_name,
                "is_switch": is_switch,
                "is_ai": using_ai_params
            })

            capital = new_capital

        # Calculate final metrics
        print(f"[WF-SERVICE] Simulation complete: final capital=${capital:,.2f}, initial=${self.initial_capital:,.2f}")
        print(f"[WF-SERVICE] Equity curve has {len(equity_curve)} points")
        if len(equity_curve) > 0:
            equities = [p['equity'] for p in equity_curve]
            print(f"[WF-SERVICE] Equity range: ${min(equities):,.2f} to ${max(equities):,.2f}")
        total_return_pct = (capital - self.initial_capital) / self.initial_capital * 100

        # Calculate Sharpe ratio from equity curve
        if len(equity_curve) > 1:
            returns = []
            for i in range(1, len(equity_curve)):
                ret = (equity_curve[i]["equity"] - equity_curve[i-1]["equity"]) / equity_curve[i-1]["equity"]
                returns.append(ret)
            if returns:
                # Annualize based on frequency
                periods_per_year = {"weekly": 52, "fast": 52, "biweekly": 26, "monthly": 12}.get(reoptimization_frequency, 26)
                avg_return = np.mean(returns) * periods_per_year
                std_return = np.std(returns) * np.sqrt(periods_per_year)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        # Calculate max drawdown
        peak = equity_curve[0]["equity"]
        max_dd = 0
        for point in equity_curve:
            if point["equity"] > peak:
                peak = point["equity"]
            dd = (peak - point["equity"]) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Calculate SPY benchmark return
        benchmark_return_pct = 0.0
        if 'SPY' in scanner_service.data_cache:
            spy_df = scanner_service.data_cache['SPY']
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            # Handle timezone-aware index
            if spy_df.index.tz is not None:
                start_ts = start_ts.tz_localize(spy_df.index.tz) if start_ts.tz is None else start_ts.tz_convert(spy_df.index.tz)
                end_ts = end_ts.tz_localize(spy_df.index.tz) if end_ts.tz is None else end_ts.tz_convert(spy_df.index.tz)
            spy_data = spy_df[(spy_df.index >= start_ts) & (spy_df.index <= end_ts)]
            if len(spy_data) >= 2:
                spy_start = spy_data.iloc[0]['close']
                spy_end = spy_data.iloc[-1]['close']
                benchmark_return_pct = (spy_end - spy_start) / spy_start * 100
                print(f"[BENCHMARK] SPY {spy_start:.2f} -> {spy_end:.2f} = {benchmark_return_pct:.2f}%")
            else:
                print(f"[BENCHMARK] Warning: Only {len(spy_data)} SPY data points for {start_date} to {end_date}")
        else:
            print("[BENCHMARK] Warning: SPY not in data cache")

        # Count AI-generated switches
        ai_switches = sum(1 for s in switch_history if s.is_ai_generated)

        # Build JSON data
        switch_history_data = json.dumps([
            {
                "date": s.date,
                "from_id": s.from_strategy_id,
                "from_name": s.from_strategy_name,
                "to_id": s.to_strategy_id,
                "to_name": s.to_strategy_name,
                "reason": s.reason,
                "score_before": s.score_before,
                "score_after": s.score_after,
                "is_ai_generated": s.is_ai_generated,
                "ai_params": s.ai_params
            }
            for s in switch_history
        ])
        equity_curve_data = json.dumps(equity_curve)
        errors_data = json.dumps(simulation_errors[:20])  # Store up to 20 period info strings
        trades_data = json.dumps([
            {
                "period_start": t.period_start,
                "period_end": t.period_end,
                "strategy_name": t.strategy_name,
                "symbol": t.symbol,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "pnl_pct": t.pnl_pct,
                "pnl_dollars": t.pnl_dollars,
                "exit_reason": t.exit_reason
            }
            for t in all_trades
        ])

        # Save or update simulation in database
        if existing_job_id:
            # Update existing record (async job flow)
            result = await db.execute(
                select(WalkForwardSimulation).where(WalkForwardSimulation.id == existing_job_id)
            )
            sim_record = result.scalar_one_or_none()
            if sim_record:
                sim_record.total_return_pct = total_return_pct
                sim_record.sharpe_ratio = sharpe_ratio
                sim_record.max_drawdown_pct = max_dd
                sim_record.num_strategy_switches = len(switch_history) - 1
                sim_record.benchmark_return_pct = benchmark_return_pct
                sim_record.switch_history_json = switch_history_data
                sim_record.equity_curve_json = equity_curve_data
                sim_record.errors_json = errors_data
                sim_record.trades_json = trades_data
                sim_record.status = "completed"
        else:
            # Create new record (sync flow)
            sim_record = WalkForwardSimulation(
                simulation_date=datetime.utcnow(),
                start_date=start_date,
                end_date=end_date,
                reoptimization_frequency=reoptimization_frequency,
                total_return_pct=total_return_pct,
                sharpe_ratio=sharpe_ratio,
                max_drawdown_pct=max_dd,
                num_strategy_switches=len(switch_history) - 1,
                benchmark_return_pct=benchmark_return_pct,
                switch_history_json=switch_history_data,
                equity_curve_json=equity_curve_data,
                errors_json=errors_data,
                trades_json=trades_data,
                status="completed"
            )
            db.add(sim_record)

        await db.commit()

        # Log error summary
        if simulation_errors:
            print(f"[WF-SERVICE] {len(simulation_errors)} errors during simulation")
            for err in simulation_errors[:5]:  # Log first 5 errors
                print(f"  - {err}")

        return WalkForwardResult(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            reoptimization_frequency=reoptimization_frequency,
            total_return_pct=round(total_return_pct, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            max_drawdown_pct=round(max_dd, 2),
            num_strategy_switches=len(switch_history) - 1,
            benchmark_return_pct=round(benchmark_return_pct, 2),
            switch_history=switch_history,
            equity_curve=equity_curve,
            period_details=period_details,
            ai_optimizations=ai_optimizations,
            parameter_evolution=parameter_evolution,
            errors=simulation_errors[:10],  # Include up to 10 errors
            trades=all_trades  # All trades across all periods
        )

    async def get_simulation_history(
        self,
        db: AsyncSession,
        limit: int = 10
    ) -> List[Dict]:
        """Get recent walk-forward simulations"""
        from sqlalchemy import desc

        result = await db.execute(
            select(WalkForwardSimulation)
            .order_by(desc(WalkForwardSimulation.simulation_date))
            .limit(limit)
        )
        sims = result.scalars().all()

        return [
            {
                "id": s.id,
                "simulation_date": s.simulation_date.isoformat() if s.simulation_date else None,
                "start_date": s.start_date.isoformat() if s.start_date else None,
                "end_date": s.end_date.isoformat() if s.end_date else None,
                "reoptimization_frequency": s.reoptimization_frequency,
                "total_return_pct": s.total_return_pct,
                "sharpe_ratio": s.sharpe_ratio,
                "max_drawdown_pct": s.max_drawdown_pct,
                "num_strategy_switches": s.num_strategy_switches,
                "benchmark_return_pct": s.benchmark_return_pct,
                "status": s.status
            }
            for s in sims
        ]

    async def get_simulation_details(
        self,
        db: AsyncSession,
        simulation_id: int
    ) -> Optional[Dict]:
        """Get detailed results for a specific simulation"""
        result = await db.execute(
            select(WalkForwardSimulation).where(WalkForwardSimulation.id == simulation_id)
        )
        sim = result.scalar_one_or_none()

        if not sim:
            return None

        return {
            "id": sim.id,
            "simulation_date": sim.simulation_date.isoformat() if sim.simulation_date else None,
            "start_date": sim.start_date.isoformat() if sim.start_date else None,
            "end_date": sim.end_date.isoformat() if sim.end_date else None,
            "reoptimization_frequency": sim.reoptimization_frequency,
            "total_return_pct": sim.total_return_pct,
            "sharpe_ratio": sim.sharpe_ratio,
            "max_drawdown_pct": sim.max_drawdown_pct,
            "num_strategy_switches": sim.num_strategy_switches,
            "benchmark_return_pct": sim.benchmark_return_pct,
            "switch_history": json.loads(sim.switch_history_json) if sim.switch_history_json else [],
            "equity_curve": json.loads(sim.equity_curve_json) if sim.equity_curve_json else [],
            "errors": json.loads(sim.errors_json) if sim.errors_json else [],
            "trades": json.loads(sim.trades_json) if sim.trades_json else [],
            "status": sim.status
        }


# Singleton instance
walk_forward_service = WalkForwardService()

"""
Strategy Generator Service - AI-powered strategy parameter optimization

Uses grid search over parameter ranges, adapting based on current market conditions.
"""

import json
import itertools
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import StrategyDefinition, StrategyGenerationRun
from app.services.backtester import BacktesterService
from app.services.scanner import scanner_service


@dataclass
class MarketConditions:
    """Current market conditions for adapting parameters"""
    regime: str  # "bull", "bear", "neutral"
    volatility: str  # "low", "normal", "high"
    spy_above_ma200: bool
    vix_level: float
    trend_strength: float


@dataclass
class OptimizationResult:
    """Result from parameter optimization"""
    best_params: Dict
    expected_sharpe: float
    expected_return_pct: float
    expected_drawdown_pct: float
    combinations_tested: int
    market_regime: str
    top_5_results: List[Dict]


class StrategyGeneratorService:
    """AI-powered strategy parameter optimization via grid search"""

    # Parameter ranges for grid search (minimal for speed - 6 combinations max)
    PARAMETER_RANGES = {
        "momentum": {
            "short_momentum_days": [10],            # 1 value (fixed)
            "long_momentum_days": [60],             # 1 value (fixed)
            "trailing_stop_pct": [12, 15, 18],      # 3 values
            "max_positions": [5],                   # 1 value (fixed)
            "position_size_pct": [18],              # 1 value (fixed)
            "near_50d_high_pct": [5],               # 1 value (fixed)
        },
        "dwap": {
            "dwap_threshold_pct": [5],              # 1 value (fixed)
            "stop_loss_pct": [7, 8, 9],             # 3 values
            "profit_target_pct": [20],              # 1 value (fixed)
            "max_positions": [15],                  # 1 value (fixed)
            "position_size_pct": [6],               # 1 value (fixed)
        }
    }

    # Narrowed ranges for different market conditions
    REGIME_ADJUSTMENTS = {
        "bear": {
            "momentum": {
                "trailing_stop_pct": [15, 18, 20],  # Wider stops
                "max_positions": [2, 3, 4],  # Fewer positions
                "position_size_pct": [10, 12, 15],  # Smaller sizes
            },
            "dwap": {
                "stop_loss_pct": [10, 12, 15],  # Wider stops
                "max_positions": [5, 8, 10],  # Fewer positions
            }
        },
        "high_volatility": {
            "momentum": {
                "trailing_stop_pct": [15, 18, 20],
                "position_size_pct": [10, 12, 15],
            },
            "dwap": {
                "stop_loss_pct": [10, 12, 15],
                "position_size_pct": [4, 5, 6],
            }
        }
    }

    def __init__(self):
        pass

    async def detect_market_conditions(self) -> MarketConditions:
        """
        Analyze SPY and VIX to determine current market conditions.

        Returns:
            MarketConditions with regime, volatility, and other metrics
        """
        spy_above_ma200 = True
        vix_level = 15.0
        trend_strength = 0.5
        regime = "neutral"
        volatility = "normal"

        # Check SPY
        if 'SPY' in scanner_service.data_cache:
            spy_df = scanner_service.data_cache['SPY']
            if len(spy_df) >= 200:
                current_price = spy_df.iloc[-1]['close']
                ma_200 = spy_df['close'].rolling(200).mean().iloc[-1]
                ma_50 = spy_df['close'].rolling(50).mean().iloc[-1]

                spy_above_ma200 = current_price > ma_200
                trend_strength = (current_price / ma_200 - 1) * 100

                # Determine regime
                if current_price > ma_200 and current_price > ma_50:
                    regime = "bull" if trend_strength > 5 else "neutral"
                elif current_price < ma_200 and current_price < ma_50:
                    regime = "bear" if trend_strength < -5 else "neutral"

        # Check VIX (volatility)
        if 'VIX' in scanner_service.data_cache:
            vix_df = scanner_service.data_cache['VIX']
            if len(vix_df) > 0:
                vix_level = vix_df.iloc[-1]['close']
                if vix_level > 25:
                    volatility = "high"
                elif vix_level < 15:
                    volatility = "low"

        # Override to bear if VIX is very high
        if vix_level > 30:
            regime = "bear"

        return MarketConditions(
            regime=regime,
            volatility=volatility,
            spy_above_ma200=spy_above_ma200,
            vix_level=vix_level,
            trend_strength=trend_strength
        )

    def _get_adjusted_ranges(
        self,
        strategy_type: str,
        conditions: MarketConditions
    ) -> Dict[str, List]:
        """
        Get parameter ranges adjusted for current market conditions.

        Args:
            strategy_type: "momentum" or "dwap"
            conditions: Current market conditions

        Returns:
            Dict of parameter name -> list of values to test
        """
        base_ranges = self.PARAMETER_RANGES.get(strategy_type, {}).copy()

        # Apply bear market adjustments
        if conditions.regime == "bear" and "bear" in self.REGIME_ADJUSTMENTS:
            bear_adj = self.REGIME_ADJUSTMENTS["bear"].get(strategy_type, {})
            base_ranges.update(bear_adj)

        # Apply high volatility adjustments
        if conditions.volatility == "high" and "high_volatility" in self.REGIME_ADJUSTMENTS:
            vol_adj = self.REGIME_ADJUSTMENTS["high_volatility"].get(strategy_type, {})
            base_ranges.update(vol_adj)

        return base_ranges

    def _generate_combinations(self, param_ranges: Dict[str, List]) -> List[Dict]:
        """Generate all combinations of parameters"""
        keys = list(param_ranges.keys())
        values = [param_ranges[k] for k in keys]

        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)

        return combinations

    def _evaluate_combination(
        self,
        params: Dict,
        strategy_type: str,
        lookback_days: int,
        top_symbols: List[str]
    ) -> Optional[Dict]:
        """
        Run backtest for a single parameter combination.

        Returns:
            Dict with metrics or None if backtest failed
        """
        from app.services.strategy_analyzer import CustomBacktester, StrategyParams

        try:
            # Build full params with defaults
            full_params = {
                "max_positions": 5,
                "position_size_pct": 18.0,
                "min_volume": 500_000,
                "min_price": 20.0,
            }

            if strategy_type == "momentum":
                full_params.update({
                    "short_momentum_days": 10,
                    "long_momentum_days": 60,
                    "trailing_stop_pct": 12.0,
                    "market_filter_enabled": True,
                    "rebalance_frequency": "weekly",
                    "short_mom_weight": 0.5,
                    "long_mom_weight": 0.3,
                    "volatility_penalty": 0.2,
                    "near_50d_high_pct": 5.0,
                })
            else:
                full_params.update({
                    "dwap_threshold_pct": 5.0,
                    "stop_loss_pct": 8.0,
                    "profit_target_pct": 20.0,
                    "volume_spike_mult": 1.5,
                })

            # Apply test parameters
            full_params.update(params)

            # Create backtester with custom params
            backtester = CustomBacktester()
            strategy_params = StrategyParams(**full_params)
            backtester.configure(strategy_params)

            # Run backtest with limited symbols for speed
            result = backtester.run_backtest(
                lookback_days=lookback_days,
                strategy_type=strategy_type,
                ticker_list=top_symbols if top_symbols else None
            )

            return {
                "params": params,
                "sharpe_ratio": result.sharpe_ratio,
                "total_return_pct": result.total_return_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
            }

        except Exception as e:
            import traceback
            print(f"Evaluation failed for {params}: {e}\n{traceback.format_exc()}")
            return None

    async def generate_optimized_strategy(
        self,
        db: AsyncSession,
        lookback_weeks: int = 12,
        strategy_type: str = "momentum",
        optimization_metric: str = "sharpe",
        auto_create: bool = False
    ) -> OptimizationResult:
        """
        Run grid search to find optimal strategy parameters.

        Args:
            db: Database session
            lookback_weeks: Weeks of data to use for optimization
            strategy_type: "momentum" or "dwap"
            optimization_metric: "sharpe", "return", or "calmar"
            auto_create: Whether to automatically create a strategy from results

        Returns:
            OptimizationResult with best parameters and metrics
        """
        from app.services.strategy_analyzer import get_top_liquid_symbols

        lookback_days = lookback_weeks * 5  # ~5 trading days per week

        # Get top liquid symbols once (for speed - use fewer for generator)
        top_symbols = get_top_liquid_symbols(max_symbols=50)
        if not top_symbols:
            raise RuntimeError("No liquid symbols found. Ensure data is loaded.")

        # Detect market conditions
        conditions = await self.detect_market_conditions()

        # Get adjusted parameter ranges
        param_ranges = self._get_adjusted_ranges(strategy_type, conditions)

        # Generate all combinations
        combinations = self._generate_combinations(param_ranges)
        total_combinations = len(combinations)

        # Evaluate each combination
        results = []
        for combo in combinations:
            result = self._evaluate_combination(combo, strategy_type, lookback_days, top_symbols)
            if result:
                results.append(result)

        if not results:
            raise RuntimeError("No valid backtest results. Ensure data is loaded.")

        # Sort by optimization metric
        if optimization_metric == "sharpe":
            results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
        elif optimization_metric == "return":
            results.sort(key=lambda x: x["total_return_pct"], reverse=True)
        elif optimization_metric == "calmar":
            # Calmar ratio = return / max_drawdown
            for r in results:
                r["calmar"] = r["total_return_pct"] / max(r["max_drawdown_pct"], 0.1)
            results.sort(key=lambda x: x.get("calmar", 0), reverse=True)

        best = results[0]
        top_5 = results[:5]

        # Create generation run record
        gen_run = StrategyGenerationRun(
            run_date=datetime.utcnow(),
            lookback_weeks=lookback_weeks,
            strategy_type=strategy_type,
            optimization_metric=optimization_metric,
            market_regime_detected=conditions.regime,
            best_params_json=json.dumps(best["params"]),
            expected_sharpe=best["sharpe_ratio"],
            expected_return_pct=best["total_return_pct"],
            expected_drawdown_pct=best["max_drawdown_pct"],
            combinations_tested=total_combinations,
            status="completed"
        )

        # Optionally create a strategy from the results
        if auto_create:
            # Build full params
            full_params = {
                "max_positions": best["params"].get("max_positions", 5),
                "position_size_pct": best["params"].get("position_size_pct", 18.0),
                "min_volume": 500_000,
                "min_price": 20.0,
            }

            if strategy_type == "momentum":
                full_params.update({
                    "short_momentum_days": best["params"].get("short_momentum_days", 10),
                    "long_momentum_days": best["params"].get("long_momentum_days", 60),
                    "trailing_stop_pct": best["params"].get("trailing_stop_pct", 12.0),
                    "market_filter_enabled": True,
                    "rebalance_frequency": "weekly",
                    "short_mom_weight": 0.5,
                    "long_mom_weight": 0.3,
                    "volatility_penalty": 0.2,
                    "near_50d_high_pct": best["params"].get("near_50d_high_pct", 5.0),
                })
            else:
                full_params.update({
                    "dwap_threshold_pct": best["params"].get("dwap_threshold_pct", 5.0),
                    "stop_loss_pct": best["params"].get("stop_loss_pct", 8.0),
                    "profit_target_pct": best["params"].get("profit_target_pct", 20.0),
                    "volume_spike_mult": 1.5,
                })

            # Create strategy
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
            strategy = StrategyDefinition(
                name=f"AI-{strategy_type.title()}-{timestamp}",
                description=f"AI-generated {strategy_type} strategy optimized for {optimization_metric}. "
                           f"Market regime: {conditions.regime}. "
                           f"Expected Sharpe: {best['sharpe_ratio']:.2f}",
                strategy_type=strategy_type,
                parameters=json.dumps(full_params),
                is_active=False,
                source="ai_generated",
                is_custom=False
            )
            db.add(strategy)
            await db.flush()  # Get the ID
            gen_run.created_strategy_id = strategy.id

        db.add(gen_run)
        await db.commit()

        return OptimizationResult(
            best_params=best["params"],
            expected_sharpe=best["sharpe_ratio"],
            expected_return_pct=best["total_return_pct"],
            expected_drawdown_pct=best["max_drawdown_pct"],
            combinations_tested=total_combinations,
            market_regime=conditions.regime,
            top_5_results=top_5
        )

    async def get_generation_history(
        self,
        db: AsyncSession,
        limit: int = 10
    ) -> List[Dict]:
        """Get recent strategy generation runs"""
        from sqlalchemy import select, desc

        result = await db.execute(
            select(StrategyGenerationRun)
            .order_by(desc(StrategyGenerationRun.run_date))
            .limit(limit)
        )
        runs = result.scalars().all()

        return [
            {
                "id": r.id,
                "run_date": r.run_date.isoformat() if r.run_date else None,
                "lookback_weeks": r.lookback_weeks,
                "strategy_type": r.strategy_type,
                "optimization_metric": r.optimization_metric,
                "market_regime_detected": r.market_regime_detected,
                "best_params": json.loads(r.best_params_json) if r.best_params_json else {},
                "expected_sharpe": r.expected_sharpe,
                "expected_return_pct": r.expected_return_pct,
                "expected_drawdown_pct": r.expected_drawdown_pct,
                "combinations_tested": r.combinations_tested,
                "status": r.status,
                "created_strategy_id": r.created_strategy_id
            }
            for r in runs
        ]


# Singleton instance
strategy_generator_service = StrategyGeneratorService()

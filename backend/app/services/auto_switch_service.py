"""
Auto-Switch Service - Automated strategy analysis and switching with AI optimization

Manages scheduled strategy analysis and automatic switching based on
performance with safeguards (minimum score difference, cooldown period).

Enhanced with AI parameter optimization to detect emerging market trends
and adapt strategy parameters dynamically.
"""

import json
import logging
import itertools
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.core.database import (
    AutoSwitchConfig, StrategyDefinition, StrategyEvaluation,
    StrategySwitchHistory, async_session
)
from app.services.strategy_analyzer import (
    strategy_analyzer_service, CustomBacktester, StrategyParams, get_top_liquid_symbols
)
from app.services.email_service import admin_email_service
from app.services.scanner import scanner_service
from app.services.market_regime import market_regime_detector, MarketRegime, MARKET_REGIMES
import pandas as pd

logger = logging.getLogger(__name__)

# Default admin email
DEFAULT_ADMIN_EMAIL = "erik@rigacap.com"


class AutoSwitchService:
    """
    Manages automated strategy analysis and switching with AI optimization.

    Features:
    - Scheduled analysis (biweekly by default)
    - AI parameter optimization to detect emerging trends
    - Safeguards: minimum score difference, cooldown period
    - Email notifications on analysis and switch
    - Audit trail of all switches
    """

    # Expanded parameter ranges for AI optimization
    EXPANDED_AI_PARAM_RANGES = {
        "momentum": {
            "trailing_stop_pct": [10, 12, 15, 18, 20],
            "max_positions": [3, 4, 5, 6, 7],
            "position_size_pct": [12, 15, 18, 20],
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

    # Coarse grid for fast optimization
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
        pass

    def _detect_market_regime(self) -> MarketRegime:
        """Detect current market regime using multi-factor analysis"""
        return market_regime_detector.detect_regime()

    def _calculate_score(self, metrics: Dict) -> float:
        """Calculate recommendation score from metrics (legacy fixed weights)"""
        sharpe = metrics.get('sharpe_ratio', 0)
        total_return = metrics.get('total_return_pct', 0)
        max_drawdown = metrics.get('max_drawdown_pct', 0)

        sharpe_score = min(max(sharpe / 2, 0), 1) * 40
        return_score = min(max(total_return / 50, 0), 1) * 30
        dd_score = max(1 - max_drawdown / 20, 0) * 30

        return sharpe_score + return_score + dd_score

    def _calculate_adaptive_score(self, metrics: Dict, regime: MarketRegime) -> float:
        """Calculate score using regime-specific weights"""
        weights = regime.scoring_weights

        normalized = {
            "sharpe_ratio": min(max(metrics.get("sharpe_ratio", 0) / 2, 0), 1),
            "total_return": min(max(metrics.get("total_return_pct", 0) / 50, 0), 1),
            "max_drawdown": max(1 - metrics.get("max_drawdown_pct", 0) / 25, 0),
            "sortino_ratio": min(max(metrics.get("sortino_ratio", 0) / 2.5, 0), 1),
            "profit_factor": min(max((metrics.get("profit_factor", 1) - 1) / 2, 0), 1),
            "calmar_ratio": min(max(metrics.get("calmar_ratio", 0) / 3, 0), 1),
        }

        score = sum(
            normalized.get(metric, 0) * weight
            for metric, weight in weights.items()
        ) * 100

        return round(score, 2)

    def _run_ai_optimization(
        self,
        strategy_type: str = "momentum",
        lookback_days: int = 60
    ) -> Optional[Dict]:
        """
        Run AI parameter optimization for current market conditions.

        Uses two-phase optimization with regime-specific adaptive scoring.
        Returns best parameters with expected metrics.
        """
        regime = self._detect_market_regime()
        top_symbols = get_top_liquid_symbols(max_symbols=50)

        if not top_symbols:
            return None

        # Base params
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

        # Get regime-adjusted parameter ranges
        param_ranges = self._get_regime_adjusted_ranges(
            self.COARSE_PARAM_RANGES.get(strategy_type, {}),
            regime
        )

        if not param_ranges:
            return None

        # Generate combinations
        keys = list(param_ranges.keys())
        values = [param_ranges[k] for k in keys]
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        best_result = None
        best_score = -float('inf')
        combinations_tested = 0

        for combo in combinations:
            test_params = {**base_params, **combo}

            try:
                params = StrategyParams(**test_params)
                backtester = CustomBacktester()
                backtester.configure(params)

                result = backtester.run_backtest(
                    lookback_days=lookback_days,
                    strategy_type=strategy_type,
                    ticker_list=top_symbols
                )
                combinations_tested += 1

                metrics = {
                    "sharpe_ratio": result.sharpe_ratio,
                    "total_return_pct": result.total_return_pct,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "sortino_ratio": result.sortino_ratio,
                    "calmar_ratio": result.calmar_ratio,
                    "profit_factor": result.profit_factor,
                }

                # Use adaptive scoring based on regime
                adaptive_score = self._calculate_adaptive_score(metrics, regime)

                if adaptive_score > best_score:
                    best_score = adaptive_score
                    best_result = {
                        "params": test_params,
                        "sharpe_ratio": result.sharpe_ratio,
                        "total_return_pct": result.total_return_pct,
                        "max_drawdown_pct": result.max_drawdown_pct,
                        "sortino_ratio": result.sortino_ratio,
                        "calmar_ratio": result.calmar_ratio,
                        "profit_factor": result.profit_factor,
                        "score": adaptive_score,
                        "market_regime": regime.name,
                        "regime_risk_level": regime.risk_level,
                        "regime_confidence": regime.confidence,
                        "strategy_type": strategy_type,
                        "combinations_tested": combinations_tested,
                    }
            except Exception:
                continue

        return best_result

    def _get_regime_adjusted_ranges(
        self,
        base_ranges: Dict[str, List],
        regime: MarketRegime
    ) -> Dict[str, List]:
        """Adjust parameter ranges based on market regime risk level"""
        adjusted = {}

        for param, values in base_ranges.items():
            if regime.risk_level == "extreme":
                if param in ["trailing_stop_pct", "stop_loss_pct"]:
                    adjusted[param] = [v for v in values if v >= 15] or values[-2:]
                elif param == "max_positions":
                    adjusted[param] = [v for v in values if v <= 4] or values[:2]
                else:
                    adjusted[param] = values
            elif regime.risk_level == "high":
                if param in ["trailing_stop_pct", "stop_loss_pct"]:
                    adjusted[param] = [v for v in values if v >= 12] or values[-3:]
                elif param == "max_positions":
                    adjusted[param] = [v for v in values if v <= 5] or values[:3]
                else:
                    adjusted[param] = values
            elif regime.risk_level == "low":
                if param in ["trailing_stop_pct", "stop_loss_pct"]:
                    adjusted[param] = [v for v in values if v <= 15] or values[:3]
                elif param == "max_positions":
                    adjusted[param] = [v for v in values if v >= 5] or values[-3:]
                else:
                    adjusted[param] = values
            else:
                adjusted[param] = values

        return adjusted

    async def get_config(self, db: AsyncSession) -> AutoSwitchConfig:
        """Get or create auto-switch configuration"""
        result = await db.execute(
            select(AutoSwitchConfig).limit(1)
        )
        config = result.scalar_one_or_none()

        if not config:
            # Create default config
            config = AutoSwitchConfig(
                is_enabled=False,
                analysis_frequency="biweekly",
                min_score_diff_to_switch=10.0,
                min_days_since_last_switch=14,
                notify_on_analysis=True,
                notify_on_switch=True,
                admin_email=DEFAULT_ADMIN_EMAIL
            )
            db.add(config)
            await db.commit()
            await db.refresh(config)

        return config

    async def update_config(
        self,
        db: AsyncSession,
        is_enabled: Optional[bool] = None,
        analysis_frequency: Optional[str] = None,
        min_score_diff: Optional[float] = None,
        min_days_cooldown: Optional[int] = None,
        notify_on_analysis: Optional[bool] = None,
        notify_on_switch: Optional[bool] = None,
        admin_email: Optional[str] = None
    ) -> AutoSwitchConfig:
        """Update auto-switch configuration"""
        config = await self.get_config(db)

        if is_enabled is not None:
            config.is_enabled = is_enabled
        if analysis_frequency is not None:
            config.analysis_frequency = analysis_frequency
        if min_score_diff is not None:
            config.min_score_diff_to_switch = min_score_diff
        if min_days_cooldown is not None:
            config.min_days_since_last_switch = min_days_cooldown
        if notify_on_analysis is not None:
            config.notify_on_analysis = notify_on_analysis
        if notify_on_switch is not None:
            config.notify_on_switch = notify_on_switch
        if admin_email is not None:
            config.admin_email = admin_email

        await db.commit()
        await db.refresh(config)
        return config

    async def get_last_switch_date(self, db: AsyncSession) -> Optional[datetime]:
        """Get the date of the most recent strategy switch"""
        result = await db.execute(
            select(StrategySwitchHistory)
            .order_by(desc(StrategySwitchHistory.switch_date))
            .limit(1)
        )
        last_switch = result.scalar_one_or_none()
        return last_switch.switch_date if last_switch else None

    async def check_safeguards(
        self,
        db: AsyncSession,
        score_diff: float,
        config: AutoSwitchConfig
    ) -> Tuple[bool, str]:
        """
        Check if a switch passes all safeguards.

        Args:
            db: Database session
            score_diff: Score difference between recommended and current
            config: Auto-switch configuration

        Returns:
            Tuple of (passes: bool, reason: str)
        """
        # Check minimum score difference
        if score_diff < config.min_score_diff_to_switch:
            return False, f"Score difference ({score_diff:.1f}) below minimum ({config.min_score_diff_to_switch})"

        # Check cooldown period
        last_switch = await self.get_last_switch_date(db)
        if last_switch:
            days_since_switch = (datetime.utcnow() - last_switch).days
            if days_since_switch < config.min_days_since_last_switch:
                return False, f"Cooldown active ({days_since_switch} days since last switch, minimum {config.min_days_since_last_switch})"

        return True, "All safeguards passed"

    async def execute_switch(
        self,
        db: AsyncSession,
        to_strategy_id: int,
        trigger: str,
        reason: str,
        score_before: Optional[float] = None,
        score_after: Optional[float] = None
    ) -> Dict:
        """
        Execute a strategy switch with full audit trail.

        Args:
            db: Database session
            to_strategy_id: ID of strategy to switch to
            trigger: "manual" or "auto_scheduled"
            reason: Human-readable reason for switch
            score_before: Score of previous strategy
            score_after: Score of new strategy

        Returns:
            Dict with switch details
        """
        # Get current active strategy
        result = await db.execute(
            select(StrategyDefinition).where(StrategyDefinition.is_active == True)
        )
        current_active = result.scalar_one_or_none()
        from_id = current_active.id if current_active else None
        from_name = current_active.name if current_active else None

        # Get new strategy
        result = await db.execute(
            select(StrategyDefinition).where(StrategyDefinition.id == to_strategy_id)
        )
        new_strategy = result.scalar_one_or_none()

        if not new_strategy:
            raise ValueError(f"Strategy {to_strategy_id} not found")

        # Deactivate all strategies
        all_result = await db.execute(select(StrategyDefinition))
        for strat in all_result.scalars():
            strat.is_active = False

        # Activate new strategy
        new_strategy.is_active = True
        new_strategy.activated_at = datetime.utcnow()

        # Record switch in history
        switch_record = StrategySwitchHistory(
            switch_date=datetime.utcnow(),
            from_strategy_id=from_id,
            to_strategy_id=to_strategy_id,
            trigger=trigger,
            reason=reason,
            score_before=score_before,
            score_after=score_after
        )
        db.add(switch_record)

        await db.commit()

        logger.info(f"Strategy switch executed: {from_name} -> {new_strategy.name} ({trigger})")

        return {
            "switch_id": switch_record.id,
            "from_strategy": from_name,
            "to_strategy": new_strategy.name,
            "trigger": trigger,
            "reason": reason,
            "switch_date": switch_record.switch_date.isoformat()
        }

    async def scheduled_analysis_job(self):
        """
        Scheduled job that runs biweekly to analyze strategies with AI optimization.

        This is called by the scheduler service. It:
        1. Loads config, checks if enabled
        2. Runs strategy analysis on existing strategies
        3. Runs AI parameter optimization to detect emerging trends
        4. Compares AI-optimized params against existing strategies
        5. Checks safeguards
        6. Executes switch if recommended and passes safeguards
        7. Sends email notifications with AI insights
        """
        logger.info("🔄 Running scheduled strategy analysis with AI optimization...")

        async with async_session() as db:
            try:
                # Load config
                config = await self.get_config(db)

                if not config.is_enabled:
                    logger.info("Auto-switch is disabled, skipping analysis")
                    return

                # Step 1: Run analysis on existing strategies
                analysis = await strategy_analyzer_service.evaluate_all_strategies(
                    db=db,
                    lookback_days=60
                )

                recommended_id = analysis.get("recommended_strategy_id")
                current_active_id = analysis.get("current_active_strategy_id")
                recommendation_notes = analysis.get("recommendation_notes", "")

                # Get scores
                evaluations = analysis.get("evaluations", [])
                recommended_eval = next(
                    (e for e in evaluations if e["strategy_id"] == recommended_id), None
                )
                current_eval = next(
                    (e for e in evaluations if e["strategy_id"] == current_active_id), None
                )

                best_existing_score = recommended_eval.get("recommendation_score", 0) if recommended_eval else 0
                current_score = current_eval.get("recommendation_score", 0) if current_eval else 0

                # Step 2: Run AI optimization to detect emerging trends
                logger.info("🤖 Running AI parameter optimization...")
                ai_result = self._run_ai_optimization(strategy_type="momentum", lookback_days=60)

                ai_recommendation = None
                ai_outperforms = False

                if ai_result:
                    ai_score = ai_result.get("score", 0)
                    market_regime = ai_result.get("market_regime", "weak_bull")
                    regime_risk = ai_result.get("regime_risk_level", "medium")
                    regime_conf = ai_result.get("regime_confidence", 0)

                    logger.info(f"🤖 AI optimization complete: score={ai_score:.1f}, "
                               f"regime={market_regime} (risk={regime_risk}, conf={regime_conf:.0%})")

                    # Compare AI against best existing strategy
                    if ai_score > best_existing_score:
                        ai_outperforms = True
                        score_diff = ai_score - current_score
                        ai_recommendation = {
                            "params": ai_result["params"],
                            "score": ai_score,
                            "score_diff": score_diff,
                            "market_regime": market_regime,
                            "regime_risk_level": regime_risk,
                            "regime_confidence": regime_conf,
                            "expected_sharpe": ai_result.get("sharpe_ratio", 0),
                            "expected_return": ai_result.get("total_return_pct", 0),
                            "expected_sortino": ai_result.get("sortino_ratio", 0),
                            "expected_calmar": ai_result.get("calmar_ratio", 0),
                            "expected_profit_factor": ai_result.get("profit_factor", 0),
                            "combinations_tested": ai_result.get("combinations_tested", 0),
                        }
                        logger.info(f"🤖 AI optimization outperforms existing strategies by {ai_score - best_existing_score:.1f} points")

                # Determine best option
                if ai_outperforms and ai_recommendation:
                    best_score = ai_recommendation["score"]
                    score_diff = ai_recommendation["score_diff"]
                    is_ai_recommendation = True
                else:
                    best_score = best_existing_score
                    score_diff = best_existing_score - current_score
                    is_ai_recommendation = False

                # Check if switch is recommended
                switch_executed = False
                switch_reason = ""

                should_switch = (is_ai_recommendation and ai_recommendation) or (recommended_id != current_active_id)

                if should_switch and score_diff > 0:
                    # Check safeguards
                    passes, safeguard_reason = await self.check_safeguards(db, score_diff, config)

                    if passes:
                        if is_ai_recommendation:
                            # AI-optimized params are best - update the active strategy with new params
                            # For now, we'll create a new AI-generated strategy
                            from app.core.database import StrategyDefinition

                            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
                            regime_display = ai_recommendation['market_regime'].replace('_', ' ').title()
                            new_strategy = StrategyDefinition(
                                name=f"AI-{regime_display.replace(' ', '')}-{timestamp}",
                                description=f"AI-generated strategy for {regime_display} market. "
                                           f"Expected Sharpe: {ai_recommendation['expected_sharpe']:.2f}, "
                                           f"Sortino: {ai_recommendation.get('expected_sortino', 0):.2f}",
                                strategy_type="momentum",
                                parameters=json.dumps(ai_recommendation["params"]),
                                is_active=False,
                                source="ai_generated",
                                is_custom=False
                            )
                            db.add(new_strategy)
                            await db.flush()

                            # Execute switch to the new AI strategy
                            switch_result = await self.execute_switch(
                                db=db,
                                to_strategy_id=new_strategy.id,
                                trigger="auto_scheduled_ai",
                                reason=f"AI optimization detected better params for {ai_recommendation['market_regime']} market (+{score_diff:.1f} score)",
                                score_before=current_score,
                                score_after=best_score
                            )
                            switch_executed = True
                            switch_reason = f"AI created and activated: {new_strategy.name}"
                            logger.info(f"🤖 AI-generated strategy activated: {new_strategy.name}")

                        else:
                            # Existing strategy is best
                            switch_result = await self.execute_switch(
                                db=db,
                                to_strategy_id=recommended_id,
                                trigger="auto_scheduled",
                                reason=f"Scheduled analysis recommended switch (+{score_diff:.1f} score improvement)",
                                score_before=current_score,
                                score_after=best_score
                            )
                            switch_executed = True
                            switch_reason = f"Switched to {switch_result['to_strategy']}"
                            logger.info(f"✅ Auto-switch executed: {switch_result['from_strategy']} -> {switch_result['to_strategy']}")

                        # Send switch notification email
                        if config.notify_on_switch and config.admin_email:
                            await admin_email_service.send_switch_notification_email(
                                to_email=config.admin_email,
                                from_strategy=switch_result.get('from_strategy', 'None'),
                                to_strategy=switch_result['to_strategy'],
                                reason=switch_result['reason'],
                                metrics={
                                    "score_before": current_score,
                                    "score_after": best_score,
                                    "score_diff": score_diff,
                                    "is_ai_generated": is_ai_recommendation,
                                    "market_regime": ai_recommendation.get("market_regime") if ai_recommendation else None
                                }
                            )
                    else:
                        switch_reason = f"Switch blocked: {safeguard_reason}"
                        logger.info(f"⚠️ Switch blocked by safeguard: {safeguard_reason}")
                else:
                    switch_reason = "Current strategy is optimal"
                    logger.info("✅ Current strategy remains optimal")

                # Enrich analysis with AI insights
                analysis["ai_optimization"] = {
                    "performed": ai_result is not None,
                    "outperforms_existing": ai_outperforms,
                    "recommendation": ai_recommendation,
                    "market_regime": ai_result.get("market_regime") if ai_result else None
                }

                # Send analysis notification email
                if config.notify_on_analysis and config.admin_email:
                    await admin_email_service.send_strategy_analysis_email(
                        to_email=config.admin_email,
                        analysis_results=analysis,
                        recommendation=recommendation_notes,
                        switch_executed=switch_executed,
                        switch_reason=switch_reason
                    )

                logger.info("✅ Scheduled analysis with AI optimization complete")

            except Exception as e:
                logger.error(f"❌ Scheduled analysis failed: {e}")
                import traceback
                traceback.print_exc()

    async def get_switch_history(
        self,
        db: AsyncSession,
        limit: int = 20
    ) -> list:
        """Get recent switch history"""
        result = await db.execute(
            select(StrategySwitchHistory)
            .order_by(desc(StrategySwitchHistory.switch_date))
            .limit(limit)
        )
        switches = result.scalars().all()

        # Get strategy names
        history = []
        for s in switches:
            from_name = None
            to_name = None

            if s.from_strategy_id:
                from_result = await db.execute(
                    select(StrategyDefinition).where(StrategyDefinition.id == s.from_strategy_id)
                )
                from_strat = from_result.scalar_one_or_none()
                from_name = from_strat.name if from_strat else f"Strategy {s.from_strategy_id}"

            if s.to_strategy_id:
                to_result = await db.execute(
                    select(StrategyDefinition).where(StrategyDefinition.id == s.to_strategy_id)
                )
                to_strat = to_result.scalar_one_or_none()
                to_name = to_strat.name if to_strat else f"Strategy {s.to_strategy_id}"

            history.append({
                "id": s.id,
                "switch_date": s.switch_date.isoformat() if s.switch_date else None,
                "from_strategy_id": s.from_strategy_id,
                "from_strategy_name": from_name,
                "to_strategy_id": s.to_strategy_id,
                "to_strategy_name": to_name,
                "trigger": s.trigger,
                "reason": s.reason,
                "score_before": s.score_before,
                "score_after": s.score_after
            })

        return history

    async def manual_trigger_analysis(self, db: AsyncSession) -> Dict:
        """
        Manually trigger strategy analysis with AI optimization.

        Unlike scheduled_analysis_job, this doesn't auto-execute switches,
        but returns the recommendation including AI insights for admin review.
        """
        # Run analysis on existing strategies
        analysis = await strategy_analyzer_service.evaluate_all_strategies(
            db=db,
            lookback_days=60
        )

        recommended_id = analysis.get("recommended_strategy_id")
        current_active_id = analysis.get("current_active_strategy_id")

        # Get scores
        evaluations = analysis.get("evaluations", [])
        recommended_eval = next(
            (e for e in evaluations if e["strategy_id"] == recommended_id), None
        )
        current_eval = next(
            (e for e in evaluations if e["strategy_id"] == current_active_id), None
        )

        best_existing_score = recommended_eval.get("recommendation_score", 0) if recommended_eval else 0
        current_score = current_eval.get("recommendation_score", 0) if current_eval else 0

        # Run AI optimization
        ai_result = self._run_ai_optimization(strategy_type="momentum", lookback_days=60)

        ai_optimization = {
            "performed": False,
            "outperforms_existing": False,
            "recommendation": None,
            "market_regime": None
        }

        if ai_result:
            ai_score = ai_result.get("score", 0)
            ai_optimization = {
                "performed": True,
                "outperforms_existing": ai_score > best_existing_score,
                "recommendation": {
                    "params": ai_result["params"],
                    "score": ai_score,
                    "expected_sharpe": ai_result.get("sharpe_ratio", 0),
                    "expected_return": ai_result.get("total_return_pct", 0),
                },
                "market_regime": ai_result.get("market_regime", "neutral"),
                "score_vs_existing": ai_score - best_existing_score
            }

        # Determine best option
        if ai_optimization["outperforms_existing"]:
            best_score = ai_result["score"]
            score_diff = best_score - current_score
            switch_to_ai = True
        else:
            best_score = best_existing_score
            score_diff = best_existing_score - current_score
            switch_to_ai = False

        # Check safeguards
        config = await self.get_config(db)
        passes, safeguard_reason = await self.check_safeguards(db, score_diff, config)

        return {
            **analysis,
            "switch_recommended": (recommended_id != current_active_id) or switch_to_ai,
            "switch_to_ai_recommended": switch_to_ai,
            "score_diff": score_diff,
            "safeguards_pass": passes,
            "safeguard_reason": safeguard_reason,
            "auto_switch_enabled": config.is_enabled,
            "ai_optimization": ai_optimization
        }


# Singleton instance
auto_switch_service = AutoSwitchService()

"""
Tests for Enhanced AI Optimization: Parameter Expansion & Market Regime Clustering

Tests:
1. Enhanced metrics calculation in backtester
2. Market regime detection
3. Adaptive scoring
4. Parameter grid generation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from dataclasses import asdict

# Test 1: Market Regime Service
def test_market_regime_definitions():
    """Test that all 6 market regimes are defined correctly"""
    from app.services.market_regime import MARKET_REGIMES, DEFAULT_SCORING_WEIGHTS

    expected_regimes = ["strong_bull", "weak_bull", "range_bound", "weak_bear", "panic_crash", "recovery"]

    assert len(MARKET_REGIMES) == 6, f"Expected 6 regimes, got {len(MARKET_REGIMES)}"

    for regime_name in expected_regimes:
        assert regime_name in MARKET_REGIMES, f"Missing regime: {regime_name}"
        regime = MARKET_REGIMES[regime_name]

        # Each regime should have required fields
        assert "conditions" in regime, f"Missing conditions in {regime_name}"
        assert "risk_level" in regime, f"Missing risk_level in {regime_name}"
        assert "param_adjustments" in regime, f"Missing param_adjustments in {regime_name}"
        assert "scoring_weights" in regime, f"Missing scoring_weights in {regime_name}"

        # Risk level should be valid
        assert regime["risk_level"] in ["low", "medium", "high", "extreme"], \
            f"Invalid risk level in {regime_name}: {regime['risk_level']}"

        # Scoring weights should sum to approximately 1.0
        weights_sum = sum(regime["scoring_weights"].values())
        assert 0.99 <= weights_sum <= 1.01, \
            f"Scoring weights in {regime_name} sum to {weights_sum}, expected ~1.0"

    print("✓ All 6 market regimes defined correctly")


def test_market_conditions_dataclass():
    """Test MarketConditions dataclass"""
    from app.services.market_regime import MarketConditions

    conditions = MarketConditions(
        spy_vs_200ma=5.5,
        spy_vs_50ma=2.3,
        vix_level=15.0,
        vix_percentile=30.0,
        trend_strength=25.0,
        breadth_pct=65.0
    )

    assert conditions.spy_vs_200ma == 5.5
    assert conditions.vix_level == 15.0
    assert conditions.breadth_pct == 65.0

    print("✓ MarketConditions dataclass works correctly")


def test_market_regime_dataclass():
    """Test MarketRegime dataclass"""
    from app.services.market_regime import MarketRegime, MarketConditions

    conditions = MarketConditions(
        spy_vs_200ma=8.0,
        spy_vs_50ma=3.0,
        vix_level=14.0,
        vix_percentile=25.0,
        trend_strength=30.0,
        breadth_pct=70.0
    )

    regime = MarketRegime(
        name="strong_bull",
        risk_level="low",
        confidence=0.85,
        conditions=conditions,
        param_adjustments={"trailing_stop_pct": -2, "max_positions": +1},
        scoring_weights={"sharpe_ratio": 0.25, "total_return": 0.35}
    )

    assert regime.name == "strong_bull"
    assert regime.risk_level == "low"
    assert regime.confidence == 0.85
    assert regime.param_adjustments["trailing_stop_pct"] == -2

    print("✓ MarketRegime dataclass works correctly")


def test_regime_scoring_weights_structure():
    """Test that scoring weights have correct structure for adaptive scoring"""
    from app.services.market_regime import MARKET_REGIMES

    required_metrics = ["sharpe_ratio", "total_return", "max_drawdown"]
    optional_metrics = ["sortino_ratio", "profit_factor", "calmar_ratio"]

    for regime_name, regime in MARKET_REGIMES.items():
        weights = regime["scoring_weights"]

        # Check required metrics
        for metric in required_metrics:
            assert metric in weights, f"Missing {metric} in {regime_name} scoring weights"
            assert 0 <= weights[metric] <= 1, f"Invalid weight for {metric} in {regime_name}"

        # Risk-based weight validation
        if regime["risk_level"] == "extreme":
            assert weights["max_drawdown"] >= 0.35, \
                f"Extreme risk {regime_name} should emphasize max_drawdown"
        elif regime["risk_level"] == "low":
            assert weights["total_return"] >= 0.30, \
                f"Low risk {regime_name} should allow higher return emphasis"

    print("✓ Scoring weights structure is correct for all regimes")


# Test 2: Enhanced Metrics in Backtester
def test_backtest_result_has_enhanced_metrics():
    """Test that BacktestResult includes all new enhanced metrics"""
    from app.services.backtester import BacktestResult
    from dataclasses import fields

    field_names = [f.name for f in fields(BacktestResult)]

    # Original metrics
    original_metrics = [
        "total_return_pct", "sharpe_ratio", "max_drawdown_pct",
        "win_rate", "total_trades"
    ]

    # New enhanced metrics
    enhanced_metrics = [
        "calmar_ratio", "sortino_ratio", "profit_factor",
        "avg_win_pct", "avg_loss_pct", "win_loss_ratio",
        "recovery_factor", "max_consecutive_losses", "ulcer_index"
    ]

    for metric in original_metrics + enhanced_metrics:
        assert metric in field_names, f"Missing metric in BacktestResult: {metric}"

    print(f"✓ BacktestResult has all {len(enhanced_metrics)} enhanced metrics")


def test_enhanced_metrics_calculation():
    """Test that _calculate_enhanced_metrics returns expected structure"""
    from app.services.backtester import BacktesterService, SimulatedTrade

    backtester = BacktesterService()

    # Create mock trade data
    trades = [
        SimulatedTrade(id=1, symbol="AAPL", entry_date="2024-01-01", exit_date="2024-01-15",
                      entry_price=100, exit_price=110, shares=10, pnl=100, pnl_pct=10.0,
                      exit_reason="profit", days_held=14, dwap_at_entry=95),
        SimulatedTrade(id=2, symbol="MSFT", entry_date="2024-01-10", exit_date="2024-01-20",
                      entry_price=200, exit_price=190, shares=5, pnl=-50, pnl_pct=-5.0,
                      exit_reason="stop", days_held=10, dwap_at_entry=195),
        SimulatedTrade(id=3, symbol="GOOG", entry_date="2024-01-15", exit_date="2024-01-25",
                      entry_price=150, exit_price=165, shares=8, pnl=120, pnl_pct=10.0,
                      exit_reason="profit", days_held=10, dwap_at_entry=145),
    ]

    equity_curve = [
        {"date": "2024-01-01", "equity": 100000},
        {"date": "2024-01-10", "equity": 100500},
        {"date": "2024-01-15", "equity": 99800},
        {"date": "2024-01-20", "equity": 100200},
        {"date": "2024-01-25", "equity": 101500},
    ]

    returns = [0.005, -0.007, 0.004, 0.013]  # Daily returns

    metrics = backtester._calculate_enhanced_metrics(
        trades=trades,
        equity_curve=equity_curve,
        returns=returns,
        total_return_pct=1.5,
        max_dd=2.0
    )

    # Verify all metrics are returned
    expected_keys = [
        "calmar_ratio", "sortino_ratio", "profit_factor",
        "avg_win_pct", "avg_loss_pct", "win_loss_ratio",
        "recovery_factor", "max_consecutive_losses", "ulcer_index"
    ]

    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"
        assert isinstance(metrics[key], (int, float)), f"{key} should be numeric"

    # Verify some metric values make sense
    assert metrics["profit_factor"] > 0, "Profit factor should be positive"
    assert metrics["avg_win_pct"] > 0, "Avg win should be positive"
    assert metrics["max_consecutive_losses"] >= 0, "Max consecutive losses should be >= 0"

    print(f"✓ Enhanced metrics calculated correctly:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")


# Test 3: Walk Forward Service Updates
def test_expanded_param_ranges():
    """Test that expanded parameter ranges are defined"""
    from app.services.walk_forward_service import WalkForwardService

    service = WalkForwardService()

    # Check expanded ranges exist
    assert hasattr(service, 'EXPANDED_AI_PARAM_RANGES'), "Missing EXPANDED_AI_PARAM_RANGES"
    assert hasattr(service, 'COARSE_PARAM_RANGES'), "Missing COARSE_PARAM_RANGES"

    # Check momentum params
    momentum_params = service.EXPANDED_AI_PARAM_RANGES.get("momentum", {})
    assert "trailing_stop_pct" in momentum_params
    assert "max_positions" in momentum_params
    assert "position_size_pct" in momentum_params
    assert "short_momentum_days" in momentum_params
    assert "near_50d_high_pct" in momentum_params

    # Verify expanded ranges have more options than coarse
    for param in ["trailing_stop_pct", "max_positions"]:
        expanded_count = len(momentum_params.get(param, []))
        coarse_count = len(service.COARSE_PARAM_RANGES.get("momentum", {}).get(param, []))
        assert expanded_count >= coarse_count, \
            f"Expanded {param} should have >= options than coarse"

    print("✓ Expanded parameter ranges defined correctly")
    print(f"  - Momentum params: {list(momentum_params.keys())}")
    print(f"  - Trailing stop options: {momentum_params['trailing_stop_pct']}")


def test_ai_optimization_result_enhanced_fields():
    """Test that AIOptimizationResult includes enhanced fields"""
    from app.services.walk_forward_service import AIOptimizationResult
    from dataclasses import fields

    field_names = [f.name for f in fields(AIOptimizationResult)]

    # Original fields
    original_fields = [
        "date", "best_params", "expected_sharpe", "expected_return_pct",
        "strategy_type", "market_regime", "was_adopted", "reason"
    ]

    # New enhanced fields
    enhanced_fields = [
        "expected_sortino", "expected_calmar", "expected_profit_factor",
        "expected_max_dd", "combinations_tested", "regime_confidence",
        "regime_risk_level", "adaptive_score"
    ]

    for field in original_fields + enhanced_fields:
        assert field in field_names, f"Missing field in AIOptimizationResult: {field}"

    print(f"✓ AIOptimizationResult has all {len(enhanced_fields)} enhanced fields")


def test_adaptive_score_calculation():
    """Test adaptive scoring with different regime weights"""
    from app.services.walk_forward_service import WalkForwardService
    from app.services.market_regime import MarketRegime, MarketConditions, MARKET_REGIMES

    service = WalkForwardService()

    # Create metrics
    metrics = {
        "sharpe_ratio": 1.5,
        "total_return_pct": 25.0,
        "max_drawdown_pct": 10.0,
        "sortino_ratio": 2.0,
        "profit_factor": 1.8,
        "calmar_ratio": 2.5,
    }

    conditions = MarketConditions(
        spy_vs_200ma=8.0, spy_vs_50ma=3.0, vix_level=14.0,
        vix_percentile=25.0, trend_strength=30.0, breadth_pct=70.0
    )

    # Test with different regimes
    scores = {}
    for regime_name, regime_def in MARKET_REGIMES.items():
        regime = MarketRegime(
            name=regime_name,
            risk_level=regime_def["risk_level"],
            confidence=0.8,
            conditions=conditions,
            param_adjustments=regime_def["param_adjustments"],
            scoring_weights=regime_def["scoring_weights"]
        )

        score = service._calculate_adaptive_score(metrics, regime)
        scores[regime_name] = score
        assert 0 <= score <= 100, f"Score should be 0-100, got {score}"

    print("✓ Adaptive scoring works for all regimes:")
    for regime, score in scores.items():
        print(f"  - {regime}: {score:.1f}")

    # Scores should vary between regimes
    unique_scores = len(set(round(s, 1) for s in scores.values()))
    assert unique_scores >= 3, "Scores should vary between regimes"


# Test 4: Auto Switch Service Updates
def test_auto_switch_has_regime_methods():
    """Test that AutoSwitchService has new regime-aware methods"""
    from app.services.auto_switch_service import AutoSwitchService

    service = AutoSwitchService()

    # Check methods exist
    assert hasattr(service, '_detect_market_regime'), "Missing _detect_market_regime"
    assert hasattr(service, '_calculate_adaptive_score'), "Missing _calculate_adaptive_score"
    assert hasattr(service, '_get_regime_adjusted_ranges'), "Missing _get_regime_adjusted_ranges"

    # Check expanded param ranges
    assert hasattr(service, 'EXPANDED_AI_PARAM_RANGES'), "Missing EXPANDED_AI_PARAM_RANGES"
    assert hasattr(service, 'COARSE_PARAM_RANGES'), "Missing COARSE_PARAM_RANGES"

    print("✓ AutoSwitchService has all new regime-aware methods")


def test_regime_adjusted_ranges():
    """Test that parameter ranges adjust based on regime risk"""
    from app.services.auto_switch_service import AutoSwitchService
    from app.services.market_regime import MarketRegime, MarketConditions

    service = AutoSwitchService()

    conditions = MarketConditions(
        spy_vs_200ma=0, spy_vs_50ma=0, vix_level=20,
        vix_percentile=50, trend_strength=20, breadth_pct=50
    )

    base_ranges = {
        "trailing_stop_pct": [10, 12, 15, 18, 20],
        "max_positions": [3, 4, 5, 6, 7],
    }

    # Test extreme risk regime (should be conservative)
    extreme_regime = MarketRegime(
        name="panic_crash", risk_level="extreme", confidence=0.9,
        conditions=conditions, param_adjustments={}, scoring_weights={}
    )
    extreme_ranges = service._get_regime_adjusted_ranges(base_ranges, extreme_regime)

    # In extreme risk, should prefer higher stops (more protective)
    assert all(v >= 15 for v in extreme_ranges.get("trailing_stop_pct", [])), \
        "Extreme risk should prefer higher trailing stops"
    # Should prefer fewer positions
    assert all(v <= 4 for v in extreme_ranges.get("max_positions", [])), \
        "Extreme risk should prefer fewer positions"

    # Test low risk regime (can be aggressive)
    low_regime = MarketRegime(
        name="strong_bull", risk_level="low", confidence=0.9,
        conditions=conditions, param_adjustments={}, scoring_weights={}
    )
    low_ranges = service._get_regime_adjusted_ranges(base_ranges, low_regime)

    # In low risk, can use tighter stops
    assert any(v <= 15 for v in low_ranges.get("trailing_stop_pct", [])), \
        "Low risk should allow tighter trailing stops"
    # Can have more positions
    assert any(v >= 5 for v in low_ranges.get("max_positions", [])), \
        "Low risk should allow more positions"

    print("✓ Parameter ranges adjust correctly based on regime risk")
    print(f"  - Extreme risk trailing_stop_pct: {extreme_ranges['trailing_stop_pct']}")
    print(f"  - Low risk trailing_stop_pct: {low_ranges['trailing_stop_pct']}")


# Test 5: Strategy Analyzer Enhanced Metrics
def test_strategy_analyzer_returns_enhanced_metrics():
    """Test that strategy evaluations include enhanced metrics"""
    from app.services.strategy_analyzer import StrategyAnalyzerService

    # Check that the service exists and has evaluate_strategy method
    service = StrategyAnalyzerService()
    assert hasattr(service, 'evaluate_strategy')
    assert hasattr(service, 'calculate_recommendation_score')

    print("✓ StrategyAnalyzerService has required methods")


# Run all tests
if __name__ == "__main__":
    print("=" * 60)
    print("Running Enhanced AI Optimization Tests")
    print("=" * 60)
    print()

    tests = [
        ("Market Regime Definitions", test_market_regime_definitions),
        ("MarketConditions Dataclass", test_market_conditions_dataclass),
        ("MarketRegime Dataclass", test_market_regime_dataclass),
        ("Scoring Weights Structure", test_regime_scoring_weights_structure),
        ("BacktestResult Enhanced Metrics", test_backtest_result_has_enhanced_metrics),
        ("Enhanced Metrics Calculation", test_enhanced_metrics_calculation),
        ("Expanded Parameter Ranges", test_expanded_param_ranges),
        ("AIOptimizationResult Fields", test_ai_optimization_result_enhanced_fields),
        ("Adaptive Score Calculation", test_adaptive_score_calculation),
        ("AutoSwitch Regime Methods", test_auto_switch_has_regime_methods),
        ("Regime-Adjusted Ranges", test_regime_adjusted_ranges),
        ("Strategy Analyzer Enhanced", test_strategy_analyzer_returns_enhanced_metrics),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)

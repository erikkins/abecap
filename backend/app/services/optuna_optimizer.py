"""
Optuna-based Bayesian Optimizer for Strategy Parameters

Replaces grid search with TPE (Tree-structured Parzen Estimator) Bayesian
optimization. Same compute budget (~30 trials) but explores a continuous
parameter space instead of a handful of fixed grid points.

Features:
- Continuous parameter spaces with regime constraints
- Warm-starting from previous period's best params
- Deterministic seeding for reproducibility
- Multivariate TPE to model parameter correlations
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


# Parameter space definitions: (low, high, step) for each strategy type
# step=None means continuous float, step=int means discrete integer
PARAM_SPACES = {
    "ensemble": {
        "dwap_threshold_pct": {"low": 3.0, "high": 8.0, "step": 0.5},
        "trailing_stop_pct": {"low": 8.0, "high": 22.0, "step": 1.0},
        "max_positions": {"low": 3, "high": 8, "step": 1},
        "position_size_pct": {"low": 10.0, "high": 20.0, "step": 1.0},
        "near_50d_high_pct": {"low": 2.0, "high": 10.0, "step": 1.0},
        "short_mom_weight": {"low": 0.3, "high": 0.7, "step": 0.05},
        "long_mom_weight": {"low": 0.1, "high": 0.5, "step": 0.05},
        "volatility_penalty": {"low": 0.05, "high": 0.35, "step": 0.05},
    },
    "momentum": {
        "trailing_stop_pct": {"low": 8.0, "high": 22.0, "step": 1.0},
        "max_positions": {"low": 3, "high": 8, "step": 1},
        "position_size_pct": {"low": 10.0, "high": 20.0, "step": 1.0},
        "short_momentum_days": {"type": "categorical", "choices": [5, 10, 15, 20]},
        "near_50d_high_pct": {"low": 2.0, "high": 10.0, "step": 1.0},
        "short_mom_weight": {"low": 0.3, "high": 0.7, "step": 0.05},
        "long_mom_weight": {"low": 0.1, "high": 0.5, "step": 0.05},
        "volatility_penalty": {"low": 0.05, "high": 0.35, "step": 0.05},
    },
    "dwap": {
        "dwap_threshold_pct": {"low": 3.0, "high": 8.0, "step": 0.5},
        "stop_loss_pct": {"low": 5.0, "high": 14.0, "step": 1.0},
        "profit_target_pct": {"low": 12.0, "high": 35.0, "step": 1.0},
        "max_positions": {"low": 8, "high": 20, "step": 1},
        "position_size_pct": {"low": 4.0, "high": 8.0, "step": 0.5},
    },
}

# Regime-specific constraint overrides: {risk_level: {param: {bound: value}}}
REGIME_CONSTRAINTS = {
    "extreme": {
        "trailing_stop_pct": {"low": 15.0},
        "max_positions": {"high": 4},
        "position_size_pct": {"high": 15.0},
    },
    "high": {
        "trailing_stop_pct": {"low": 12.0},
        "max_positions": {"high": 6},
    },
    "low": {
        "trailing_stop_pct": {"high": 18.0},
        "max_positions": {"low": 4},
    },
    # "medium": no constraints (full space)
}


class StrategyOptimizer:
    """
    Bayesian optimizer for trading strategy parameters using Optuna TPE.
    """

    def optimize(
        self,
        strategy_type: str,
        objective_fn: Callable[[Dict[str, Any]], Optional[float]],
        regime_risk_level: str = "medium",
        warm_start_params: Optional[Dict[str, Any]] = None,
        n_trials: int = 30,
        seed_date: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run Bayesian optimization over strategy parameter space.

        Args:
            strategy_type: "ensemble", "momentum", or "dwap"
            objective_fn: Function that takes a params dict and returns a score
                         (higher is better), or None if the trial failed.
            regime_risk_level: Current market regime risk ("low", "medium", "high", "extreme")
            warm_start_params: Previous period's best params to enqueue as first trial
            n_trials: Number of optimization trials (default 30)
            seed_date: Date for deterministic seeding (default: uses fixed seed)

        Returns:
            Dict with "best_params" and "best_score", or None if all trials failed.
        """
        import optuna
        from optuna.samplers import TPESampler

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Deterministic seed from date
        if seed_date:
            seed = int(seed_date.timestamp()) % (2**31)
        else:
            seed = 42

        space = PARAM_SPACES.get(strategy_type)
        if not space:
            return None

        # Apply regime constraints
        constrained_space = self._apply_regime_constraints(space, regime_risk_level)

        # Create study with TPE sampler
        sampler = TPESampler(
            seed=seed,
            n_startup_trials=max(n_trials // 3, 3),
            multivariate=True,
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Warm-start: enqueue previous best as first trial
        if warm_start_params:
            enqueue_params = {}
            for param_name, param_def in constrained_space.items():
                if param_name in warm_start_params:
                    val = warm_start_params[param_name]
                    # Clamp to current constrained bounds
                    if param_def.get("type") == "categorical":
                        if val in param_def["choices"]:
                            enqueue_params[param_name] = val
                    else:
                        low = param_def["low"]
                        high = param_def["high"]
                        val = max(low, min(high, val))
                        enqueue_params[param_name] = val
            if enqueue_params:
                study.enqueue_trial(enqueue_params)

        # Define objective that maps Optuna trial to our param space
        best_result = {"score": -float("inf"), "params": None}

        def optuna_objective(trial):
            params = {}
            for param_name, param_def in constrained_space.items():
                if param_def.get("type") == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_def["choices"]
                    )
                elif isinstance(param_def.get("step"), int) or (
                    isinstance(param_def.get("low"), int)
                    and isinstance(param_def.get("high"), int)
                ):
                    params[param_name] = trial.suggest_int(
                        param_name,
                        int(param_def["low"]),
                        int(param_def["high"]),
                        step=int(param_def.get("step", 1)),
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_def["low"],
                        param_def["high"],
                        step=param_def.get("step"),
                    )

            score = objective_fn(params)
            if score is None:
                raise optuna.TrialPruned()

            if score > best_result["score"]:
                best_result["score"] = score
                best_result["params"] = params.copy()

            return score

        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)

        if best_result["params"] is None:
            return None

        return {
            "best_params": best_result["params"],
            "best_score": best_result["score"],
        }

    def _apply_regime_constraints(
        self,
        space: Dict[str, Dict],
        risk_level: str,
    ) -> Dict[str, Dict]:
        """
        Narrow parameter bounds based on market regime risk level.
        Returns a new dict (does not mutate input).
        """
        constraints = REGIME_CONSTRAINTS.get(risk_level, {})
        if not constraints:
            return dict(space)

        constrained = {}
        for param_name, param_def in space.items():
            new_def = dict(param_def)
            if param_name in constraints:
                overrides = constraints[param_name]
                if "low" in overrides and "low" in new_def:
                    new_def["low"] = max(new_def["low"], overrides["low"])
                if "high" in overrides and "high" in new_def:
                    new_def["high"] = min(new_def["high"], overrides["high"])
                # Ensure low <= high
                if "low" in new_def and "high" in new_def:
                    if new_def["low"] > new_def["high"]:
                        new_def["low"] = new_def["high"]
            constrained[param_name] = new_def

        return constrained

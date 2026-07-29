"""Rolling trailing-365-day walk-forward per tier (Preserver / Maximizer).

Computes a REAL walk-forward of each tier's strategy over the last ~365 days, refreshed daily
so the Simulated Portfolio card moves every day as the window rolls. Runs in the WORKER (has
the price data_cache + 900s budget); result cached to S3 (tier_walkforward.json) and served by
tier_serving.tier_backtest().

Mirrors scripts/maximizer_blend_certify.py's Option-B blend, but on PRODUCTION components
(scripts/ don't ship in the Lambda image):
  - t30v core curve      -> backtester_service.run_backtest (momentum + 30% trailing)
  - regime series (daily)-> market_regime_service.get_regime_history
  - breakout sleeve      -> maximizer_portfolio.replay_sleeve (gated to rotating_bull) +
                            vol_scaled_returns
  - Preserver            -> t30v x capitulation exposure overlay (0.25 in capitulation)
  - Maximizer (Option-B) -> where(rotating_bull, breakout+vol-target, Preserver)

NOTE: because it uses the production regime classifier + live universe (not the offline
research pipeline/PITFWU), this rolling number is DISTINCT from the marketed full-cycle
certified figure — it is the honest "what the strategy did on our data over the trailing year."
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

CAPITULATION = {"panic_crash", "recovery", "weak_bear"}
ROTATING = "rotating_bull"
COST = 0.0015


def _stats(equity: pd.Series) -> dict:
    """Total return %, annualized Sharpe, max drawdown % from a daily equity curve."""
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return {"total_return_pct": None, "sharpe_ratio": None, "max_drawdown_pct": None}
    total = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    r = equity.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * (252 ** 0.5)) if r.std() > 0 else 0.0
    mdd = float((equity / equity.cummax() - 1).min() * 100)
    return {"total_return_pct": round(total, 1), "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(mdd, 1)}


def compute_tier_walkforward(data_cache: Dict[str, pd.DataFrame], days: int = 365) -> Optional[dict]:
    """Compute both tiers' trailing-`days` walk-forward. Returns a dict keyed by tier, each
    {total_return_pct, sharpe_ratio, max_drawdown_pct, benchmark_return_pct, start_date,
    end_date, label, window, rolling:True} — the shape tier_serving.tier_backtest expects.
    Returns None on failure (caller keeps the certified fallback)."""
    from app.services.backtester import backtester_service, ExitStrategyConfig, ExitStrategyType
    from app.services.market_regime import market_regime_service
    from app.services.maximizer_portfolio import replay_sleeve, vol_scaled_returns

    spy = data_cache.get("SPY")
    if spy is None or len(spy) < 260:
        return None
    end = pd.Timestamp(spy.index[-1]).normalize()
    start = end - pd.Timedelta(days=days)

    # 1) t30v core curve over the window (momentum engine + live 30% trailing stop).
    core = backtester_service.run_backtest(
        start_date=start.to_pydatetime(), end_date=end.to_pydatetime(),
        use_momentum_strategy=True,
        exit_strategy=ExitStrategyConfig(strategy_type=ExitStrategyType.TRAILING_STOP,
                                         trailing_stop_pct=30.0),
    )
    ec = getattr(core, "equity_curve", None) or []
    if len(ec) < 30:
        return None
    core_eq = pd.Series([p["equity"] for p in ec],
                        index=pd.to_datetime([p["date"] for p in ec])).sort_index()
    grid = core_eq.index
    rt = core_eq.pct_change().fillna(0.0)

    # 2) daily regime series over the window, aligned to the core grid.
    vix = data_cache.get("^VIX")
    hist = market_regime_service.get_regime_history(
        spy_df=spy, universe_dfs=data_cache, vix_df=vix,
        start_date=start.to_pydatetime(), end_date=end.to_pydatetime(), sample_frequency="daily")
    reg_map = {pd.Timestamp(r.date).normalize(): r.regime_type.value for r in hist}
    reg = pd.Series([reg_map.get(pd.Timestamp(d).normalize()) for d in grid], index=grid).ffill().fillna("range_bound")
    cap = reg.isin(CAPITULATION)
    rot = (reg == ROTATING)

    # 3) Preserver = t30v x capitulation exposure overlay (0.25 in capitulation, else 1.0),
    #    one-time trim cost on exposure change.
    exp_p = pd.Series(np.where(cap.to_numpy(), 0.25, 1.0), index=grid)
    r_pres = exp_p * rt - exp_p.diff().abs().fillna(0.0) * COST
    pres_eq = 100_000.0 * (1 + r_pres).cumprod()

    # 4) Maximizer Option-B: breakout sleeve gated to rotating_bull + vol-target, else Preserver.
    reg_by_date = {pd.Timestamp(d).normalize(): reg.loc[d] for d in grid}
    bk_eq = replay_sleeve(data_cache, "breakout", start, end, n_positions=15,
                          entry_regimes={ROTATING}, regime_by_date=reg_by_date)
    r_bk = vol_scaled_returns(bk_eq).reindex(grid).fillna(0.0)
    r_blend = pd.Series(np.where(rot.to_numpy(), r_bk.to_numpy(), r_pres.to_numpy()), index=grid)
    r_blend = r_blend - (rot != rot.shift(1)).fillna(False).astype(float) * COST  # swap turnover cost
    max_eq = 100_000.0 * (1 + r_blend).cumprod()

    # 5) SPY benchmark over the same grid.
    spy_g = spy["close"].reindex(grid, method="ffill")
    bench = round((spy_g.iloc[-1] / spy_g.iloc[0] - 1) * 100, 1) if len(spy_g.dropna()) >= 2 else None

    base = {"benchmark_return_pct": bench, "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"), "window": "trailing 365 days", "rolling": True}
    return {
        "preserver": {**_stats(pres_eq), **base, "label": "t30v + capitulation overlay"},
        "maximizer": {**_stats(max_eq), **base, "label": "Option-B breakout blend (N=15)"},
        "computed_at": None,  # stamped by the caller (worker) — Date.now() unavailable here is fine
    }

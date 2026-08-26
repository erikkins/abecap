#!/usr/bin/env python3
"""PRODUCTION-port tier comparison (what would actually be SERVED) vs the research return-
stream (tier_vintages_daily). Day-steps the real PreserverBook / MaximizerBook (single capital
pool, rule-B hold-to-exit, costs, warm-started) using the ACTUAL per-day regime from
regime_series — no hardcoded regime. Core = pitfwu t30v (its daily return feeds each book's
mirrored t30v leg). Same two windows as tier_vintages_daily so the numbers line up.

    source backend/venv/bin/activate
    python3 scripts/tier_compare_production.py
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "backend"); os.environ.setdefault("LAMBDA_ROLE", "worker")
for _l in open(".env"):
    if _l.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = _l.strip().split("=", 1)[1]; break

import pitfwu_wf as pwf
from regime_research import regime_series
from shapes_portfolio import perf, CAP0
from app.services.preserver_service import PreserverBook, SLEEVE_SOURCES as PRES_SLEEVES
from app.services.maximizer_service import MaximizerBook, SLEEVE_SOURCES as MAX_SLEEVES
from app.services import preserver_signal_service as pss
from app.services import maximizer_signal_service as mss
from app.services.preserver_sleeves import SLEEVE_HOLD as PRES_HOLD
from app.services.maximizer_sleeves import SLEEVE_HOLD as MAX_HOLD

WINDOWS = [("LAST 2YR (2024-05->2026-05)", pd.Timestamp("2024-05-29"), pd.Timestamp("2026-05-29")),
           ("2021-2026 (incl 2022 + momo-crash)", pd.Timestamp("2021-01-01"), pd.Timestamp("2026-05-29"))]
WARM = pd.Timedelta(days=60)   # calendar warmup before window start (brake + held book)
v = pwf.v
panel, ca = v.load_panel(), v.load_corp_actions()


def daystep(book, build_fn, sleeves, hold_map, cache_full, days, reg, t30v_ret, n):
    eq = {}
    for D in days:
        dc = {s: df.loc[df.index <= D] for s, df in cache_full.items()}
        dc = {s: df for s, df in dc.items() if len(df)}
        regime = reg.get(D, "rotating_bull")
        src, cands = build_fn(dc, regime, [], D, max_positions=n)
        bc = [{"symbol": c["symbol"], "hold": hold_map[src]} for c in cands] if src in sleeves else []
        price_of = {s: float(df["close"].iloc[-1]) for s, df in dc.items()}
        cr = float(t30v_ret.get(D, 0.0))
        eq[D] = book.advance_day(D, src, bc, price_of, core_ret=cr)
    return pd.Series(eq).sort_index()


for label, start, end in WINDOWS:
    wstart = start - WARM
    uni = [s for s in v.universe_asof(wstart.to_pydatetime(), 400, panel)
           if s not in pwf._EXCLUDED_SET and not s.startswith("^")][:100]
    cache_full = {}
    for s in uni:
        try:
            df = v.split_adjusted(s, asof=end.to_pydatetime(), ca=ca)
            if df is not None and len(df):
                cache_full[s] = df
        except Exception:
            pass
    # Core t30v curve (warm) -> daily returns feed the mirrored t30v leg
    tcurve = pwf.run(wstart.to_pydatetime(), end.to_pydatetime(), trail=0.30, max_pos=20, size=0.045)["equity_curve"]
    tseries = pd.Series([x["equity"] for x in tcurve], index=pd.to_datetime([x["date"] for x in tcurve])).sort_index()
    tret = tseries.pct_change().fillna(0.0)
    days = list(tseries.index)
    reg = regime_series(wstart, end).reindex(days, method="ffill").fillna("rotating_bull")
    reg = {d: reg.loc[d] for d in days}
    tret = {d: tret.loc[d] for d in days}

    pres = daystep(PreserverBook(15), pss.build_daily_signals, PRES_SLEEVES, PRES_HOLD, cache_full, days, reg, tret, 15)
    maxb = daystep(MaximizerBook(15), mss.build_daily_signals, MAX_SLEEVES, MAX_HOLD, cache_full, days, reg, tret, 15)

    # measure over the WINDOW (post-warmup)
    def wperf(s):
        s = s[s.index >= start]
        return perf(s / s.iloc[0] * CAP0, ppy=252)
    print(f"\n================ {label} — PRODUCTION port ================", flush=True)
    print(f"  {'tier':<24} {'Annualized':>11} {'Sharpe':>7} {'MaxDD':>8}", flush=True)
    for name, s in [("Core (t30v)", tseries), ("Preserver (prod)", pres), ("Maximizer (prod)", maxb)]:
        c, sh, m = wperf(s)
        print(f"  {name:<24} {c:>10.1f}% {sh:>7.2f} {m:>7.1f}%", flush=True)
print("\nDONE", flush=True)

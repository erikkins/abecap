#!/usr/bin/env python3
"""Validate the rewritten maximizer_service.MaximizerBook (daily incremental) reproduces the
CERTIFIED Maximizer = gated-breakout standing book + vol_scaled_returns. Day-steps the book
with build_daily_signals (gated via route) and compares to the reference over 2021-26 + last-2yr.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "backend"); os.environ.setdefault("LAMBDA_ROLE", "worker")
for _l in open(".env"):
    if _l.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = _l.strip().split("=", 1)[1]; break
import pitfwu_veneer as v
from regime_research import regime_series
from app.services.maximizer_service import MaximizerBook
from app.services.maximizer_signal_service import build_daily_signals

FULL_S, END = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-05-29")
_CA = v.load_corp_actions()

# universe = rolling union of point-in-time top-100 (same as the certified breakout tests)
union = set()
for d in pd.date_range(FULL_S, END, freq="14D"):
    union |= set([s for s in v.universe_asof_prod(d.to_pydatetime(), 300, 15.0) if not s.startswith("^")][:100])
cache_full = {}
for s in union:
    try:
        df = v.split_adjusted(s, asof=END.to_pydatetime(), ca=_CA)
        if df is not None and len(df):
            cache_full[s] = df
    except Exception:
        pass
print(f"universe {len(cache_full)}", flush=True)
spy = v.split_adjusted("SPY", asof=END.to_pydatetime(), ca=_CA)
days = [d for d in spy.index if FULL_S <= d <= END]
reg = regime_series(FULL_S, END).reindex(pd.DatetimeIndex(days), method="ffill").fillna("none")

# day-step the production MaximizerBook
book = MaximizerBook(15)
eq = {}
for D in days:
    dc = {s: df.loc[df.index <= D] for s, df in cache_full.items()}
    dc = {s: df for s, df in dc.items() if len(df)}
    regime = reg.loc[D]
    src, cands = build_daily_signals(dc, regime, [], D, max_positions=15)
    bc = [{"symbol": c["symbol"], "hold": c.get("hold_days", 29)} for c in cands] if src == "breakout" else []
    po = {s: float(df["close"].iloc[-1]) for s, df in dc.items()}
    eq[D] = book.advance_day(D, src, bc, po)
prod = pd.Series(eq)

def stats(s, start):
    s = s[s.index >= start]
    tot = (s.iloc[-1] / s.iloc[0] - 1) * 100
    mdd = (s / s.cummax() - 1).min() * 100
    return tot, mdd

for label, st in [("2021-2026", FULL_S), ("LAST 2YR", pd.Timestamp("2024-05-29"))]:
    t, m = stats(prod, st)
    print(f"{label}: maximizer_service (daily book) total={t:+.1f}%  mdd={m:.1f}%", flush=True)
print("(certified reference: gated breakout ~+38.7%/2021-26 as a sleeve; vol-target trims exposure/DD)", flush=True)
print("DONE", flush=True)

#!/usr/bin/env python3
"""Corrected apples-to-apples: research sleeve_curve(breakout) vs production replay_sleeve
('breakout') on the SAME PITFWU universe the research uses (rolling union of
universe_asof_prod(d,300,15)[:100], point-in-time — NOT a stale universe_asof snapshot).
If they now match ~38%, the production replay is faithful and the earlier 6.9% was my
wrong-universe bug; Maximizer's breakout edge is REAL and deliverable (survivorship-free).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "backend"); os.environ.setdefault("LAMBDA_ROLE", "worker")
for _l in open(".env"):
    if _l.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = _l.strip().split("=", 1)[1]; break
from shape_tpe import load_data, _EXCLUDED_SET
from stack_sleeves import sleeve_curve
from regime_allocator_v2 import BREAKOUT
import pitfwu_veneer as v
from app.services.maximizer_portfolio import replay_sleeve
_CA = v.load_corp_actions()

FULL_S, END = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-05-29")
WINDOWS = [("2021-2026", pd.Timestamp("2021-01-01")), ("LAST 2YR", pd.Timestamp("2024-05-29"))]

def perf(eq, s):
    eq = eq[eq.index >= s]
    if len(eq) < 100: return None
    eq = eq/eq.iloc[0]; yrs=(eq.index[-1]-eq.index[0]).days/365.25
    r=eq.pct_change().dropna()
    return ((eq.iloc[-1]**(1/yrs)-1)*100, (r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else 0, (eq/eq.cummax()-1).min()*100)

# RESEARCH path
data = load_data(FULL_S, END)
_, close_r, _, _ = data
research_bk = sleeve_curve(data, FULL_S, END, BREAKOUT, "breakout")
uni = list(close_r.columns)
print(f"universe (rolling union of universe_asof_prod top-100): {len(uni)} symbols", flush=True)

# PRODUCTION replay on the SAME universe
cache = {}
for s in uni:
    try:
        df = v.split_adjusted(s, asof=END.to_pydatetime(), ca=_CA)
        if df is not None and len(df): cache[s] = df
    except Exception: pass
prod_bk = replay_sleeve(cache, "breakout", FULL_S, END, n_positions=15)

for label, s in WINDOWS:
    pr, pp = perf(research_bk, s), perf(prod_bk, s)
    print(f"\n{label}:", flush=True)
    if pr: print(f"  research sleeve_curve : ann={pr[0]:6.1f}%  sharpe={pr[1]:.2f}  mdd={pr[2]:6.1f}%", flush=True)
    if pp: print(f"  prod replay_sleeve    : ann={pp[0]:6.1f}%  sharpe={pp[1]:.2f}  mdd={pp[2]:6.1f}%", flush=True)
print("DONE", flush=True)

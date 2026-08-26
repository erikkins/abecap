#!/usr/bin/env python3
"""Test the two plumbing fixes: (1) relax replay_sleeve's _MIN_BARS drop (keep all symbols like
research load_data), (2) match corp-actions (shapes_entry_edge.CA). If prod replay now converges
to research sleeve_curve (~38.7% 2021-26), the gap was harness plumbing, breakout edge is real.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "backend"); os.environ.setdefault("LAMBDA_ROLE", "worker")
for _l in open(".env"):
    if _l.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = _l.strip().split("=", 1)[1]; break
from shape_tpe import load_data
from stack_sleeves import sleeve_curve
from regime_allocator_v2 import BREAKOUT
import shapes_entry_edge as S
import pitfwu_veneer as v
import app.services.maximizer_portfolio as mp
mp._MIN_BARS = 1  # FIX 1: relax the short-history drop to match research (keep all union symbols)

FULL_S, END = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-05-29")
WINDOWS = [("2021-2026", pd.Timestamp("2021-01-01")), ("LAST 2YR", pd.Timestamp("2024-05-29"))]

def perf(eq, s):
    eq = eq[eq.index >= s]
    if len(eq) < 100: return None
    eq = eq/eq.iloc[0]; yrs=(eq.index[-1]-eq.index[0]).days/365.25
    r=eq.pct_change().dropna()
    return ((eq.iloc[-1]**(1/yrs)-1)*100, (r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else 0, (eq/eq.cummax()-1).min()*100)

data = load_data(FULL_S, END)
_, close_r, _, _ = data
research_bk = sleeve_curve(data, FULL_S, END, BREAKOUT, "breakout")
uni = list(close_r.columns)
cache = {}
for s in uni:  # FIX 2: same corp-actions (S.CA) as research load_data
    try:
        df = v.split_adjusted(s, asof=END.to_pydatetime(), ca=S.CA)
        if df is not None and len(df): cache[s] = df
    except Exception: pass
print(f"universe {len(uni)}, cache {len(cache)} (was dropping <200-bar before)", flush=True)
prod_bk = mp.replay_sleeve(cache, "breakout", FULL_S, END, n_positions=15)

for label, s in WINDOWS:
    pr, pp = perf(research_bk, s), perf(prod_bk, s)
    print(f"\n{label}:", flush=True)
    if pr: print(f"  research sleeve_curve : ann={pr[0]:6.1f}%  sharpe={pr[1]:.2f}  mdd={pr[2]:6.1f}%", flush=True)
    if pp: print(f"  prod replay (fixed)   : ann={pp[0]:6.1f}%  sharpe={pp[1]:.2f}  mdd={pp[2]:6.1f}%", flush=True)
print("DONE", flush=True)

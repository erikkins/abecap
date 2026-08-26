#!/usr/bin/env python3
"""Clean apples-to-apples: research breakout SLEEVE (stack_sleeves.sleeve_curve on shape_tpe
data) vs production breakout REPLAY (maximizer_portfolio.replay_sleeve on PITFWU), SAME windows,
SAME construction (breakout sleeve alone). Prints universe sizes. Isolates the TRUE gap (data /
universe / replay), no regime-allocation confound, no survivorship confound (both surv-free).
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
import pitfwu_wf as pwf
from app.services.maximizer_portfolio import replay_sleeve

FULL_S, END = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-05-29")
WINDOWS = [("2021-2026", pd.Timestamp("2021-01-01")), ("LAST 2YR", pd.Timestamp("2024-05-29"))]

def perf(eq, s):
    eq = eq[eq.index >= s]
    if len(eq) < 100: return None
    eq = eq/eq.iloc[0]; yrs=(eq.index[-1]-eq.index[0]).days/365.25
    r=eq.pct_change().dropna()
    return ((eq.iloc[-1]**(1/yrs)-1)*100, (r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else 0, (eq/eq.cummax()-1).min()*100)

# RESEARCH: sleeve_curve(breakout) on shape_tpe.load_data
data = load_data(FULL_S, END)
_, close_r, _, feats_r = data
research_bk = sleeve_curve(data, FULL_S, END, BREAKOUT, "breakout")
print(f"RESEARCH universe (shape_tpe): {close_r.shape[1]} symbols", flush=True)

# PRODUCTION: replay_sleeve('breakout') on PITFWU (broad universe: top-600 as-of, dynamic-ish)
panel, ca = pwf.v.load_panel(), pwf.v.load_corp_actions()
uni = [s for s in pwf.v.universe_asof(FULL_S.to_pydatetime(), 600, panel)
       if s not in pwf._EXCLUDED_SET and not s.startswith("^")]
cache = {}
for s in uni:
    try:
        df = pwf.v.split_adjusted(s, asof=END.to_pydatetime(), ca=ca)
        if df is not None and len(df): cache[s] = df
    except Exception: pass
print(f"PRODUCTION universe (PITFWU top-600 as-of): {len(cache)} symbols", flush=True)
prod_bk = replay_sleeve(cache, "breakout", FULL_S, END, n_positions=15)

for label, s in WINDOWS:
    pr, pp = perf(research_bk, s), perf(prod_bk, s)
    print(f"\n{label}:", flush=True)
    if pr: print(f"  research breakout : ann={pr[0]:6.1f}%  sharpe={pr[1]:.2f}  mdd={pr[2]:6.1f}%", flush=True)
    if pp: print(f"  prod breakout     : ann={pp[0]:6.1f}%  sharpe={pp[1]:.2f}  mdd={pp[2]:6.1f}%", flush=True)
print("DONE", flush=True)

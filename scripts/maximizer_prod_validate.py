#!/usr/bin/env python3
"""Validate the PRODUCTION Maximizer construction ports to the research return-stream.
Production = maximizer_portfolio.replay_sleeve('breakout')  (STANDING breakout book, warm)
            + vol_scaled_returns  (Barroso vol-target overlay).
Target (research return-stream, from the feasible study): Max bk+voltgt ~16.8%/1.02/-20.4%
(2021-26 median across windows). Here: single warm window, compare vs Core.

    source backend/venv/bin/activate && python3 scripts/maximizer_prod_validate.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "backend"); os.environ.setdefault("LAMBDA_ROLE", "worker")
for _l in open(".env"):
    if _l.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = _l.strip().split("=", 1)[1]; break
import pitfwu_wf as pwf
from app.services.maximizer_portfolio import replay_sleeve, vol_scaled_returns

WARM, END = pd.Timestamp("2020-06-01"), pd.Timestamp("2026-05-29")
WINDOWS = [("2021-2026", pd.Timestamp("2021-01-01")), ("LAST 2YR", pd.Timestamp("2024-05-29"))]
v = pwf.v
panel, ca = v.load_panel(), v.load_corp_actions()

uni = [s for s in v.universe_asof(WARM.to_pydatetime(), 400, panel)
       if s not in pwf._EXCLUDED_SET and not s.startswith("^")][:100]
cache = {}
for s in uni:
    try:
        df = v.split_adjusted(s, asof=END.to_pydatetime(), ca=ca)
        if df is not None and len(df):
            cache[s] = df
    except Exception:
        pass
print(f"universe {len(cache)}", flush=True)

# production standing breakout book (warm from 2020-06) + vol-target overlay
bk_eq = replay_sleeve(cache, "breakout", WARM, END, n_positions=15)
r_max = vol_scaled_returns(bk_eq)
max_eq = (1 + r_max).cumprod()
# raw breakout (no overlay) for reference
r_bk = bk_eq.pct_change().fillna(0.0)
bk_only = (1 + r_bk).cumprod()
# Core (pitfwu t30v), warm
tm = pwf.run(WARM.to_pydatetime(), END.to_pydatetime(), trail=0.30, max_pos=20, size=0.045)["equity_curve"]
core = pd.Series([x["equity"] for x in tm], index=pd.to_datetime([x["date"] for x in tm])).sort_index()

def perf(eq, s):
    eq = eq[eq.index >= s]
    if len(eq) < 100:
        return None
    eq = eq / eq.iloc[0]
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    ann = (eq.iloc[-1] ** (1 / yrs) - 1) * 100
    r = eq.pct_change().dropna()
    sh = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min() * 100
    return ann, sh, mdd

for label, s in WINDOWS:
    print(f"\n==== {label} (warm) ====", flush=True)
    for name, eq in [("Core (t30v)", core), ("Max breakout-only", bk_only), ("Max bk+voltgt (PROD)", max_eq)]:
        p = perf(eq, s)
        if p:
            print(f"  {name:22} ann={p[0]:6.1f}%  sharpe={p[1]:.2f}  mdd={p[2]:6.1f}%", flush=True)
print("\n(research target: Max bk+voltgt ~16.8%/1.02/-20.4% multi-window median)", flush=True)
print("DONE", flush=True)

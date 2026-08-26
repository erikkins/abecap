#!/usr/bin/env python3
"""Confirm the REAL cause of prod-breakout (14%) vs research-breakout (49%) gap.
NOT survivorship (PITFWU is survivorship-free). Prime suspect: UNIVERSE size/staleness.
Run research sleeve_curve(breakout) AND production replay_sleeve('breakout') on PITFWU with
small (top-100 stale) vs broad (top-400 as-of) universes; print universe sizes + trade counts.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "backend"); os.environ.setdefault("LAMBDA_ROLE", "worker")
for _l in open(".env"):
    if _l.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = _l.strip().split("=", 1)[1]; break
import pitfwu_wf as pwf
from app.services.maximizer_portfolio import replay_sleeve, vol_scaled_returns

WARM, S, END = pd.Timestamp("2020-06-01"), pd.Timestamp("2024-05-29"), pd.Timestamp("2026-05-29")  # LAST-2YR focus
v = pwf.v
panel, ca = v.load_panel(), v.load_corp_actions()

def perf(eq, s=S):
    eq = eq[eq.index >= s]
    if len(eq) < 100: return None
    eq = eq / eq.iloc[0]; yrs = (eq.index[-1]-eq.index[0]).days/365.25
    r = eq.pct_change().dropna()
    return ((eq.iloc[-1]**(1/yrs)-1)*100, (r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else 0,
            (eq/eq.cummax()-1).min()*100)

def build_cache(n_uni, dynamic=False):
    # dynamic=False: fixed top-n as-of WARM (what my validation did). We vary n to test breadth.
    uni = [s for s in v.universe_asof(WARM.to_pydatetime(), n_uni, panel)
           if s not in pwf._EXCLUDED_SET and not s.startswith("^")]
    cache = {}
    for s in uni:
        try:
            df = v.split_adjusted(s, asof=END.to_pydatetime(), ca=ca)
            if df is not None and len(df): cache[s] = df
        except Exception: pass
    return cache

for n_uni in (100, 250, 400):
    cache = build_cache(n_uni)
    bk = replay_sleeve(cache, "breakout", WARM, END, n_positions=15)
    r_max = vol_scaled_returns(bk); mx = (1+r_max).cumprod()
    p_bk, p_mx = perf(bk.pct_change().fillna(0).add(1).cumprod()), perf(mx)
    print(f"universe top-{n_uni} (cache={len(cache)}): "
          f"breakout-only ann={p_bk[0]:5.1f}%/mdd={p_bk[2]:5.1f}%  "
          f"bk+voltgt ann={p_mx[0]:5.1f}%/sh={p_mx[1]:.2f}/mdd={p_mx[2]:5.1f}%", flush=True)
print("(research Maximizer LAST-2YR was ~48.9%; if broad universe recovers -> was a UNIVERSE artifact, not strategy)", flush=True)
print("DONE", flush=True)

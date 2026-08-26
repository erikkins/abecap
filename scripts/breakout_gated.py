#!/usr/bin/env python3
"""Confirm: gating the production breakout to rotating_bull recovers the research return.
research sleeve_curve(breakout) [gated via detect REGIME_GATES] vs a production standing-book
replay driven by breakout_signal AND-ed with (regime==rotating_bull). Same universe/data/mechanics.
If gated-prod ~ research (~38.7% 2021-26), the gap was purely the regime gate my earlier test omitted.
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
from regime_research import regime_series
from app.services.maximizer_sleeves import breakout_signal
import pitfwu_veneer as v

CAP0, COST, HOLD, N = 100_000.0, 0.0015, int(BREAKOUT["hold"]), 15
FULL_S, END = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-05-29")
WINDOWS = [("2021-2026", pd.Timestamp("2021-01-01")), ("LAST 2YR", pd.Timestamp("2024-05-29"))]

def perf(eq, s):
    eq = eq[eq.index >= s]
    if len(eq) < 100: return None
    eq = eq/eq.iloc[0]; yrs=(eq.index[-1]-eq.index[0]).days/365.25
    r=eq.pct_change().dropna()
    return ((eq.iloc[-1]**(1/yrs)-1)*100, (r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else 0, (eq/eq.cummax()-1).min()*100)

arrs, close, dvols, feats = load_data(FULL_S, END)
research_bk = sleeve_curve(data=(arrs, close, dvols, feats), start=FULL_S, end=END, p=BREAKOUT, shape="breakout")

# GATED production signal: breakout_signal AND (regime == rotating_bull)
reg = regime_series(FULL_S, END).reindex(close.index, method="ffill").fillna("none")
rot = (reg == "rotating_bull")
sigs = {}
for s in close.columns:
    o, h, l, c, vol, idx = arrs[s]
    sigs[s] = pd.Series(breakout_signal(o, h, l, c, vol), index=idx)
sig = pd.DataFrame(sigs).reindex(close.index).fillna(False)
sig = sig.mul(rot.astype(bool), axis=0)  # <-- regime gate

# standing-book replay (mirrors replay_sleeve / simulate)
valid = {s: close[s].dropna().index for s in close.columns}
win = close.index[(close.index >= FULL_S) & (close.index <= END)]
pos, cash, last_px = {}, CAP0, {}
eq_d, eq_v = [], []
for today in win:
    row = close.loc[today]
    for s in close.columns:
        px = row[s]
        if px == px: last_px[s] = px
    for s in [s for s, p in pos.items() if p["exit_date"] <= today]:
        cash += pos[s]["shares"] * last_px.get(s, pos[s]["last"]) * (1 - COST); del pos[s]
    free = N - len(pos)
    if free > 0:
        cands = [s for s in close.columns if bool(sig.loc[today, s]) and s not in pos and (row[s] == row[s])]
        cands.sort(key=lambda s: -(dvols.loc[today, s] if dvols.loc[today, s] == dvols.loc[today, s] else 0))
        for s in cands[:free]:
            price = row[s]
            alloc = min((cash + sum(pos[x]["shares"] * last_px.get(x, pos[x]["last"]) for x in pos)) / N, cash)
            if alloc <= 0: break
            shares = alloc / price; cash -= alloc + alloc * COST
            vd = valid[s]; j = vd.get_indexer([today])[0]
            pos[s] = {"shares": shares, "exit_date": vd[min(j + HOLD, len(vd) - 1)] if j >= 0 else today, "last": price}
    eq_d.append(today); eq_v.append(cash + sum(pos[s]["shares"] * last_px.get(s, pos[s]["last"]) for s in pos))
gated = pd.Series(eq_v, index=pd.DatetimeIndex(eq_d))

for label, s in WINDOWS:
    pr, pg = perf(research_bk, s), perf(gated, s)
    print(f"\n{label}:", flush=True)
    if pr: print(f"  research (gated)      : ann={pr[0]:6.1f}%  sharpe={pr[1]:.2f}  mdd={pr[2]:6.1f}%", flush=True)
    if pg: print(f"  prod replay (GATED)   : ann={pg[0]:6.1f}%  sharpe={pg[1]:.2f}  mdd={pg[2]:6.1f}%", flush=True)
print("DONE", flush=True)

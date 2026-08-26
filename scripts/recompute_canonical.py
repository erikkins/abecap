"""Authoritative canonical recompute — modern window 2021-01-01 .. TODAY, on PITFWU.
Produces daily equity curves for Preserver, Maximizer, S&P 500, Raw momentum (12-mo factor).
Tier build == scripts/tier_vintages_21y.py (sleeve routing + breakout vol-brake). RESEARCH, local.
Stats (rolling / path / Calmar) are computed in a separate light post-process from the dumped curves.
"""
import os, sys, json
R = "/Users/erikkins/CODE/stocker-app"
sys.path.insert(0, os.path.join(R, "backend")); sys.path.insert(0, os.path.join(R, "scripts"))
for _l in open(os.path.join(R, ".env")):
    if _l.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = _l.strip().split("=", 1)[1]; break
os.environ.setdefault("LAMBDA_ROLE", "worker")

import numpy as np, pandas as pd
import pitfwu_veneer as v; v.EXT = True
from shape_tpe import load_data
from stack_sleeves import sleeve_curve, PULLBACK, OVERSOLD
from regime_allocator_v2 import BREAKOUT
from regime_research import regime_series
from shapes_portfolio import CAP0
import pitfwu_wf as pwf
from pitfwu_wf_periods import naive_curve

START, END = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-07-30")
CALM_BULL = {"strong_bull", "weak_bull"}
CAPITULATION = {"panic_crash", "recovery", "weak_bear"}
ROTATING = {"rotating_bull"}


def t30v_daily(s, e):
    m = pwf.run(s.to_pydatetime(), e.to_pydatetime(), trail=0.30, max_pos=20, size=0.045)
    ec = m.get("equity_curve") or []
    return pd.Series([x["equity"] for x in ec], index=pd.to_datetime([x["date"] for x in ec])).sort_index()


def vol_scale(ret, target=0.20):
    rv = (ret.rolling(20).std() * np.sqrt(252)).shift(1)
    return (target / rv).clip(upper=1.0).fillna(1.0)


print(f"Loading {START.date()}..{END.date()} DAILY...", flush=True)
data = load_data(START, END)
print("  sleeve curves...", flush=True)
pb = sleeve_curve(data, START, END, PULLBACK, "pullback_ma")
ob = sleeve_curve(data, START, END, OVERSOLD, "oversold_bounce")
bk = sleeve_curve(data, START, END, BREAKOUT, "breakout")
print("  core walk-forward (t30v)...", flush=True)
t = t30v_daily(START, END); grid = t.index
rt = t.pct_change()
rp = pb.reindex(grid, method="ffill").pct_change()
ro = ob.reindex(grid, method="ffill").pct_change()
rb = bk.reindex(grid, method="ffill").pct_change()
print("  regime series (derived, PITFWU)...", flush=True)
reg = regime_series(START, END).reindex(grid, method="ffill").fillna("none")
calm = reg.isin(CALM_BULL).to_numpy(); cap = reg.isin(CAPITULATION).to_numpy(); rot = reg.isin(ROTATING).to_numpy()
df = pd.DataFrame({"t": rt, "p": rp, "o": ro, "b": rb}).fillna(0.0)
b_scaled = df["b"] * vol_scale(df["b"])
preserver = pd.Series(np.where(calm, df["p"], np.where(cap, df["o"], df["t"])), index=grid)
maxpp = pd.Series(np.where(calm, df["p"], np.where(cap, df["o"], np.where(rot, b_scaled, df["t"]))), index=grid)

print("  SPY + raw momentum (12-mo factor)...", flush=True)
spy = v.split_adjusted("SPY")["close"].reindex(grid, method="ffill")
rm_dates, rm_eq = naive_curve(START, END, lookback=252, hold=21, topk=20)

curves = {
    "Preserver": (CAP0 * (1 + preserver).cumprod()),
    "Maximizer": (CAP0 * (1 + maxpp).cumprod()),
    "S&P 500": (spy / spy.iloc[0] * CAP0),
    "Raw momentum": (pd.Series(rm_eq, index=pd.to_datetime(rm_dates)) * CAP0 if rm_eq else None),
}
res = {}
for name, eq in curves.items():
    if eq is None:
        print(f"  ! {name}: no curve", flush=True); continue
    eq = eq.dropna()
    res[name] = {"dates": [d.strftime("%Y-%m-%d") for d in eq.index],
                 "equity": [round(float(x), 4) for x in eq.values]}
json.dump(res, open(os.path.join(R, "scripts", "canonical_recompute.json"), "w"))
print("DONE ->", {k: len(vv["dates"]) for k, vv in res.items()}, flush=True)

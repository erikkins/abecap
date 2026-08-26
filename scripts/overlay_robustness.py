#!/usr/bin/env python3
"""Preserver overlay robustness (multi-start) + oversold-bounce tilt.

Capitulation overlay flavors (keep t30v book, only act in capitulation):
  CASH  : exposure -> E, freed (1-E) to cash        (return = E*rt)
  TILT  : exposure -> E, freed (1-E) to OVERSOLD     (return = E*rt + (1-E)*ro)
Compute the t30v + oversold sleeve return streams ONCE over 2021-2026 (causal -> sliceable),
then evaluate across many 3y start dates. Report distribution (median + worst-case) per flavor/E.

    source backend/venv/bin/activate && python3 scripts/overlay_robustness.py
"""
import os, sys, statistics
import numpy as np, pandas as pd
R = "/Users/erikkins/CODE/stocker-app"
sys.path.insert(0, os.path.join(R, "backend")); sys.path.insert(0, os.path.join(R, "scripts"))
for _l in open(os.path.join(R, ".env")):
    if _l.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = _l.strip().split("=", 1)[1]; break
os.environ.setdefault("LAMBDA_ROLE", "worker")
from dateutil.relativedelta import relativedelta
from shape_tpe import load_data
from stack_sleeves import sleeve_curve, OVERSOLD
from regime_research import regime_series
from shapes_portfolio import perf, CAP0
import pitfwu_wf as pwf

CAP = {"panic_crash", "recovery", "weak_bear"}
FULL_S, FULL_E = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-05-29")

from stack_sleeves import PULLBACK
from regime_allocator_v2 import BREAKOUT
CALM = {"strong_bull", "weak_bull"}; ROT = {"rotating_bull"}
print("computing t30v + oversold + pullback + breakout curves over full range (once)...", flush=True)
data = load_data(FULL_S, FULL_E)
ob = sleeve_curve(data, FULL_S, FULL_E, OVERSOLD, "oversold_bounce")
pb = sleeve_curve(data, FULL_S, FULL_E, PULLBACK, "pullback_ma")
bk = sleeve_curve(data, FULL_S, FULL_E, BREAKOUT, "breakout")
tm = pwf.run(FULL_S.to_pydatetime(), FULL_E.to_pydatetime(), trail=0.30, max_pos=20, size=0.045)["equity_curve"]
t = pd.Series([x["equity"] for x in tm], index=pd.to_datetime([x["date"] for x in tm])).sort_index()
grid = t.index
rt = t.pct_change().fillna(0.0)
ro = ob.reindex(grid, method="ffill").pct_change().fillna(0.0)
rp = pb.reindex(grid, method="ffill").pct_change().fillna(0.0)
rb = bk.reindex(grid, method="ffill").pct_change().fillna(0.0)
reg = regime_series(FULL_S, FULL_E).reindex(grid, method="ffill").fillna("none")
capm = reg.isin(CAP); calmm = reg.isin(CALM); rotm = reg.isin(ROT)
def _volscale(r, target=0.20):
    rv = (r.rolling(20).std() * np.sqrt(252)).shift(1)
    return (target / rv).clip(upper=1.0).fillna(1.0)
rb_sc = rb * _volscale(rb)

def overlay_ret(offense, E, tilt):
    # offense = base daily return when NOT in capitulation; in capitulation apply the E overlay.
    cap_ret = E * rt + ((1 - E) * ro if tilt else 0.0)
    r = pd.Series(np.where(capm.to_numpy(), cap_ret, offense), index=grid)
    exp = pd.Series(np.where(capm.to_numpy(), E, 1.0), index=grid)
    return r - exp.diff().abs().fillna(0.0) * 0.0015

# FEASIBLE Maximizer study: hold the breakout sleeve continuously (orthogonal to Core,
# gradual hold=29 turnover) + EXPOSURE defense (cap cash-raise / vol-target). NO infeasible
# regime name-rotation. Compare vs research Maximizer (which switches books = infeasible ~8/yr).
pres_off = rt
def cap_cash(base, E):   # raise cash in capitulation (exposure-scale the base return); feasible
    r = pd.Series(np.where(capm.to_numpy(), E * base, base), index=grid)
    exp = pd.Series(np.where(capm.to_numpy(), E, 1.0), index=grid)
    return r - exp.diff().abs().fillna(0.0) * 0.0015
max_research = pd.Series(np.where(calmm.to_numpy(), rp, np.where(capm.to_numpy(), ro,
                         np.where(rotm.to_numpy(), rb_sc, rt))), index=grid)  # INFEASIBLE (switches books)
VARIANTS = [("Core", rt),
            ("P cash E.25", overlay_ret(pres_off, 0.25, False)),
            ("Max research*", max_research),         # * = infeasible (full-book regime switching)
            ("Max bkout-only", rb),                  # feasible: hold breakout always
            ("Max bk+voltgt", rb_sc),                # + Barroso vol-target (exposure)
            ("Max bk+capcash", cap_cash(rb, 0.25)),  # + capitulation cash-raise
            ("Max bk+vt+cap", cap_cash(rb_sc, 0.25))]  # + both
series = {name: s for name, s in VARIANTS}

# multi-start 3y windows, monthly
starts = []
d = pd.Timestamp("2021-01-04")
while d <= pd.Timestamp("2023-05-01"):
    starts.append(d); d += relativedelta(months=1)
res = {name: [] for name, _ in VARIANTS}
for s in starts:
    e = s + relativedelta(years=3)
    for name, _ in VARIANTS:
        r = series[name][(series[name].index >= s) & (series[name].index <= e)]
        if len(r) < 200:
            continue
        c, sh, m = perf(CAP0 * (1 + r).cumprod(), ppy=252)
        res[name].append((c, sh, m))

print(f"\n===== OVERLAY ROBUSTNESS: {len(starts)} monthly 3y-window starts (2021-2026) =====", flush=True)
print(f"{'variant':<14} {'ann med':>8} {'ann min':>8} {'sharpe med':>10} {'mdd med':>8} {'mdd WORST':>10}", flush=True)
for name, _ in VARIANTS:
    rows = res[name]
    if not rows:
        continue
    ann = [x[0] for x in rows]; sh = [x[1] for x in rows]; mdd = [x[2] for x in rows]
    print(f"{name:<14} {statistics.median(ann):>7.1f}% {min(ann):>7.1f}% {statistics.median(sh):>10.2f} "
          f"{statistics.median(mdd):>7.1f}% {min(mdd):>9.1f}%", flush=True)  # min(mdd)=true worst (most negative)
print("DONE", flush=True)

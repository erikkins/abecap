#!/usr/bin/env python3
"""Recent-timeline tier levels on the CERTIFIED constructions (survivorship-free PITFWU,
warm-started, breakout regime-gated, exposure-scaled overlays):
  Core      = t30v (pitfwu: trail .30 / 20 pos / 4.5%)
  Preserver = Core return x exposure (0.25 in capitulation regimes, else 1.0) + trim cost
  Maximizer = regime-GATED breakout sleeve (rotating_bull) x vol-target (Barroso)
Reports July-MTD, trailing-3mo, trailing-12mo returns for each + SPY, and the July regime mix.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "backend"); os.environ.setdefault("LAMBDA_ROLE", "worker")
for _l in open(".env"):
    if _l.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = _l.strip().split("=", 1)[1]; break
import pitfwu_wf as pwf
from shape_tpe import load_data
from stack_sleeves import sleeve_curve
from regime_allocator_v2 import BREAKOUT
from regime_research import regime_series
from app.services.maximizer_portfolio import vol_scaled_returns

CAP0, COST = 100_000.0, 0.0015
CAPITULATION = {"panic_crash", "recovery", "weak_bear"}
WARM = pd.Timestamp("2020-06-01")

# --- end date = latest SPY bar in the PITFWU panel ---
ca = pwf.v.load_corp_actions()
spy = pwf.v.split_adjusted("SPY", asof=pd.Timestamp("2026-12-31").to_pydatetime(), ca=ca)
END = spy.index.max()
print(f"latest PITFWU data: {END:%Y-%m-%d}", flush=True)

# --- Core (t30v) ---
tm = pwf.run(WARM.to_pydatetime(), END.to_pydatetime(), trail=0.30, max_pos=20, size=0.045)["equity_curve"]
core = pd.Series([x["equity"] for x in tm], index=pd.to_datetime([x["date"] for x in tm])).sort_index()
grid = core.index
rt = core.pct_change().fillna(0.0)
reg = regime_series(WARM, END).reindex(grid, method="ffill").fillna("none")

# --- Preserver = exposure-scaled Core (cash overlay in capitulation) ---
cap = reg.isin(CAPITULATION)
exp = pd.Series(np.where(cap.to_numpy(), 0.25, 1.0), index=grid)
r_pres = exp * rt - exp.diff().abs().fillna(0.0) * COST
pres = CAP0 * (1 + r_pres).cumprod()

# --- Maximizer = GATED breakout (rotating_bull, via detect REGIME_GATES) x vol-target ---
data = load_data(WARM, END)
bk = sleeve_curve(data, WARM, END, BREAKOUT, "breakout")   # gated to rotating_bull
r_max = vol_scaled_returns(bk)
mx = CAP0 * (1 + r_max).cumprod()

spy_eq = spy["close"].reindex(grid, method="ffill")

def stats(eq, s, e):
    eq = eq[(eq.index >= s) & (eq.index <= e)]
    if len(eq) < 2: return None
    tot = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    mdd = (eq / eq.cummax() - 1).min() * 100
    return tot, mdd

JUL = pd.Timestamp("2026-07-01")
WINS = [("July MTD", JUL), ("Trailing 3mo", END - pd.Timedelta(days=91)),
        ("Trailing 12mo", END - pd.Timedelta(days=365)), ("Full cycle 21-26", pd.Timestamp("2021-01-01"))]
print(f"\n{'window':16} {'Core (ret/mdd)':>18} {'Preserver':>18} {'Maximizer':>18} {'SPY':>16}", flush=True)
for label, s in WINS:
    def f(eq):
        r = stats(eq, s, END); return f"{r[0]:+.1f}% / {r[1]:.1f}%" if r else "   -   "
    print(f"{label:16} {f(core):>18} {f(pres):>18} {f(mx):>18} {f(spy_eq):>16}", flush=True)

jul_reg = reg[reg.index >= JUL]
print(f"\nJuly regime mix: {dict(jul_reg.value_counts())}", flush=True)
print(f"July levels ($100k start-of-July): Core ${core[core.index>=JUL].iloc[0]:.0f}->{core.iloc[-1]:.0f} "
      f"(shown as % above)", flush=True)
print("DONE", flush=True)

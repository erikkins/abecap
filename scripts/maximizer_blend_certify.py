#!/usr/bin/env python3
"""Certify Option-B Maximizer = Preserver EXCEPT rotating_bull -> breakout+vol-target.
Blend daily return: rotating_bull -> gated breakout sleeve (at N positions) x vol-target;
every other regime -> Preserver (t30v x capitulation-exposure overlay). Swap cost charged at
each rotating<->non-rotating boundary (full turnover). Report return / maxDD / #swaps at
N=15/20/25 vs Preserver, Core, SPY. Warm from 2020-06; survivorship-free PITFWU.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "backend"); os.environ.setdefault("LAMBDA_ROLE", "worker")
for _l in open(".env"):
    if _l.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = _l.strip().split("=", 1)[1]; break
import pitfwu_wf as pwf
from regime_research import regime_series
from app.services.maximizer_sleeves import breakout_signal, BREAKOUT
from app.services.maximizer_portfolio import vol_scaled_returns

CAP0, COST, HOLD = 100_000.0, 0.0015, int(BREAKOUT["hold"])
CAPIT = {"panic_crash", "recovery", "weak_bear"}
WARM, END = pd.Timestamp("2020-06-01"), pd.Timestamp("2026-05-29")
v = pwf.v; ca = v.load_corp_actions()

# Core t30v (warm) -> returns; Preserver = exposure-overlay on Core
tm = pwf.run(WARM.to_pydatetime(), END.to_pydatetime(), trail=0.30, max_pos=20, size=0.045)["equity_curve"]
core = pd.Series([x["equity"] for x in tm], index=pd.to_datetime([x["date"] for x in tm])).sort_index()
grid = core.index
rt = core.pct_change().fillna(0.0)
reg = regime_series(WARM, END).reindex(grid, method="ffill").fillna("none")
cap = reg.isin(CAPIT); rot = (reg == "rotating_bull")
exp_p = pd.Series(np.where(cap.to_numpy(), 0.25, 1.0), index=grid)
r_pres = exp_p * rt - exp_p.diff().abs().fillna(0.0) * COST     # Preserver daily return

# breakout universe (rolling union top-100) + per-symbol arrays
union = set()
for d in pd.date_range(WARM, END, freq="14D"):
    union |= set([s for s in v.universe_asof_prod(d.to_pydatetime(), 300, 15.0) if not s.startswith("^")][:100])
closes, sigs, dvols, valids = {}, {}, {}, {}
for s in union:
    try:
        df = v.split_adjusted(s, asof=END.to_pydatetime(), ca=ca)
        if df is None or len(df) < 200: continue
        o, h, l, c, vol = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
        closes[s] = df["close"]; sigs[s] = pd.Series(breakout_signal(o, h, l, c, vol), index=df.index)
        dvols[s] = (df["close"] * df["volume"]).rolling(20, min_periods=5).mean(); valids[s] = df["close"].dropna().index
    except Exception: pass
close = pd.DataFrame(closes).sort_index()
sig = pd.DataFrame(sigs).reindex(close.index).fillna(False)
sig = sig.mul((regime_series(WARM, END).reindex(close.index, method="ffill") == "rotating_bull").astype(bool), axis=0)
dvol = pd.DataFrame(dvols).reindex(close.index)

def breakout_sleeve(N):
    win = close.index[(close.index >= WARM) & (close.index <= END)]
    pos, cash, last = {}, CAP0, {}; ed, ev = [], []
    for today in win:
        row = close.loc[today]
        for s in close.columns:
            px = row[s]
            if px == px: last[s] = px
        for s in [s for s, p in pos.items() if p["exit"] <= today]:
            cash += pos[s]["sh"] * last.get(s, pos[s]["last"]) * (1 - COST); del pos[s]
        free = N - len(pos)
        if free > 0:
            cds = [s for s in close.columns if bool(sig.loc[today, s]) and s not in pos and row[s] == row[s]]
            cds.sort(key=lambda s: -(dvol.loc[today, s] if dvol.loc[today, s] == dvol.loc[today, s] else 0))
            for s in cds[:free]:
                price = row[s]; alloc = min((cash + sum(pos[x]["sh"] * last.get(x, pos[x]["last"]) for x in pos)) / N, cash)
                if alloc <= 0: break
                cash -= alloc + alloc * COST; vd = valids[s]; j = vd.get_indexer([today])[0]
                pos[s] = {"sh": alloc / price, "exit": vd[min(j + HOLD, len(vd) - 1)] if j >= 0 else today, "last": price}
        ed.append(today); ev.append(cash + sum(pos[s]["sh"] * last.get(s, pos[s]["last"]) for s in pos))
    return pd.Series(ev, index=pd.DatetimeIndex(ed))

def stats(eq, s):
    eq = eq[eq.index >= s]
    ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    mdd = (eq / eq.cummax() - 1).min() * 100
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0
    dr = eq.pct_change().dropna()
    sharpe = (dr.mean() / dr.std() * (252 ** 0.5)) if dr.std() > 0 else 0.0
    calmar = cagr / abs(mdd) if mdd else 0.0
    return ret, mdd, cagr, sharpe, calmar

spy = v.split_adjusted("SPY", asof=END.to_pydatetime(), ca=ca)["close"].reindex(grid, method="ffill")
swaps = int((rot != rot.shift(1)).fillna(False).sum())   # rotating<->non-rotating boundaries
W = [("2021-2026", pd.Timestamp("2021-01-01")), ("Trailing 12mo", END - pd.Timedelta(days=365))]
print(f"rotating<->non-rotating boundaries (swaps) over warm range: {swaps} (~{swaps/((END-WARM).days/365.25):.1f}/yr)", flush=True)
def row(name, eq):
    cells = []
    for _, s in W:
        r, m, cg, sh, ca = stats(eq, s)
        cells.append(f"{r:+7.1f}% /{m:6.1f}% /C{ca:4.2f}/S{sh:4.2f}")
    print(f"{name:16} " + "   ".join(cells), flush=True)

pres = CAP0 * (1 + r_pres).cumprod()
print("\n(ret / maxDD / Calmar=CAGR÷|MDD| / Sharpe)", flush=True)
print(f"{'variant':16} " + "   ".join(f"{w[0]:>27}" for w in W), flush=True)
for name, eq in [("Core (t30v)", core), ("Preserver", pres), ("SPY", CAP0 * spy / spy.iloc[0])]:
    row(name, eq)
for N in (15, 20, 25):  # LOCKED SPEC: N=15 (Sharpe wins both windows; dominates trailing-12mo)
    bk = breakout_sleeve(N); r_bk = vol_scaled_returns(bk).reindex(grid).fillna(0.0)
    r_blend = pd.Series(np.where(rot.to_numpy(), r_bk, r_pres), index=grid)
    r_blend = r_blend - (rot != rot.shift(1)).fillna(False).astype(float) * COST  # swap turnover cost
    mxb = CAP0 * (1 + r_blend).cumprod()
    row("Maximizer-B N=" + str(N), mxb)
print("DONE", flush=True)

#!/usr/bin/env python3
"""Pinpoint the ~21pp gap (daily book +259% vs certified +280%). Run BOTH on the IDENTICAL
universe; compare the RAW breakout sleeve equity first (isolates universe/signal), then the
vol-scaled result (isolates vol_scale windowing). Gate already ruled out (both = {rotating_bull}).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "backend"); os.environ.setdefault("LAMBDA_ROLE", "worker")
for _l in open(".env"):
    if _l.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = _l.strip().split("=", 1)[1]; break
import pitfwu_veneer as v
from regime_research import regime_series
from app.services.maximizer_service import MaximizerBook
from app.services.maximizer_signal_service import build_daily_signals
from app.services.maximizer_portfolio import replay_sleeve, vol_scaled_returns
from app.services.maximizer_sleeves import breakout_signal, BREAKOUT

FULL_S, END = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-05-29")
_CA = v.load_corp_actions()
# ONE shared universe (rolling union of point-in-time top-100)
union = set()
for d in pd.date_range(FULL_S, END, freq="14D"):
    union |= set([s for s in v.universe_asof_prod(d.to_pydatetime(), 300, 15.0) if not s.startswith("^")][:100])
cache = {}
for s in union:
    try:
        df = v.split_adjusted(s, asof=END.to_pydatetime(), ca=_CA)
        if df is not None and len(df):
            cache[s] = df
    except Exception:
        pass
print(f"shared universe {len(cache)}", flush=True)
spy = v.split_adjusted("SPY", asof=END.to_pydatetime(), ca=_CA)
days = [d for d in spy.index if FULL_S <= d <= END]
reg = regime_series(FULL_S, END).reindex(pd.DatetimeIndex(days), method="ffill").fillna("none")

# --- REFERENCE: gated breakout standing book (replay_sleeve is UNGATED, so gate the signal
#     externally like breakout_gated) then vol_scaled_returns ---
closes, sigs, dvols, valids = {}, {}, {}, {}
for s, df in cache.items():
    if len(df) < 200: continue
    o, h, l, c, vol = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    closes[s] = df["close"]; sigs[s] = pd.Series(breakout_signal(o, h, l, c, vol), index=df.index)
    dvols[s] = (df["close"] * df["volume"]).rolling(20, min_periods=5).mean(); valids[s] = df["close"].dropna().index
close = pd.DataFrame(closes).sort_index()
sig = pd.DataFrame(sigs).reindex(close.index).fillna(False)
rot = (regime_series(FULL_S, END).reindex(close.index, method="ffill") == "rotating_bull")
sig = sig.mul(rot.astype(bool), axis=0)   # GATE
dvol = pd.DataFrame(dvols).reindex(close.index)
HOLD, N, COST, CAP0 = int(BREAKOUT["hold"]), 15, 0.0015, 100000.0
win = close.index[(close.index >= FULL_S) & (close.index <= END)]
pos, cash, last = {}, CAP0, {}; ref_d, ref_v = [], []
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
    ref_d.append(today); ref_v.append(cash + sum(pos[s]["sh"] * last.get(s, pos[s]["last"]) for s in pos))
ref_bk = pd.Series(ref_v, index=pd.DatetimeIndex(ref_d))
ref_max = CAP0 * (1 + vol_scaled_returns(ref_bk)).cumprod()

# --- DAILY BOOK (maximizer_service) on the SAME universe ---
book = MaximizerBook(15); bk_eq = {}
for D in days:
    dc = {s: df.loc[df.index <= D] for s, df in cache.items()}; dc = {s: df for s, df in dc.items() if len(df)}
    regime = reg.loc[D]
    src, cands = build_daily_signals(dc, regime, [], D, max_positions=15)
    bc = [{"symbol": c["symbol"], "hold": c.get("hold_days", 29)} for c in cands] if src == "breakout" else []
    po = {s: float(df["close"].iloc[-1]) for s, df in dc.items()}
    book.advance_day(D, src, bc, po); bk_eq[D] = book._bk_equity()
book_bk = pd.Series(bk_eq)

def tot(s): s = s.dropna(); return (s.iloc[-1] / s.iloc[0] - 1) * 100
print(f"\nRAW breakout SLEEVE (pre-vol-target):  reference {tot(ref_bk):+.1f}%   daily-book {tot(book_bk):+.1f}%", flush=True)
print(f"VOL-SCALED (Maximizer):                reference {tot(ref_max):+.1f}%   daily-book(max_value) {tot(pd.Series({d: 0 for d in days})) if False else book.max_value/CAP0*100-100:+.1f}%", flush=True)
print("=> if RAW sleeves match, the gap is vol_scale windowing; if not, it's universe/signal.", flush=True)
print("DONE", flush=True)

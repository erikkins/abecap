#!/usr/bin/env python3
"""Maximizer replay with a WARMUP so the vol-brake + held book are realistic at the Jun-15
measurement anchor (a continuously-running strategy hits Jun 15 with the brake already warm
and positions already on — NOT a cold all-in into the top). Warms from WARMUP_START, then
reports the Jun 15 -> Jul 20 segment return (apples-to-apples with SPY/Core over that window).

Whole period rotating_bull -> pure breakout sleeve, n_positions=15 (matches live shadow).
Survivorship-free PITFWU panel. Prints segment return + brake state at the anchor.

    source backend/venv/bin/activate
    python3 scripts/maximizer_backfill.py
"""
import os, sys, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "backend"))
os.environ.setdefault("LAMBDA_ROLE", "worker")

import pitfwu_wf as pwf
from app.services.maximizer_service import MaximizerBook
from app.services.maximizer_signal_service import build_daily_signals
from app.services.maximizer_sleeves import SLEEVE_HOLD

WARMUP_START = pd.Timestamp("2026-05-01")   # ~30 trading days before the anchor -> brake warm by Jun 15
ANCHOR = pd.Timestamp("2026-06-15")         # the "launch" measurement point (Core inception)
END = pd.Timestamp("2026-07-20")
N_POS = 15
v = pwf.v

panel, ca = v.load_panel(), v.load_corp_actions()
uni = [s for s in v.universe_asof(WARMUP_START.to_pydatetime(), 400, panel)
       if s not in pwf._EXCLUDED_SET and not s.startswith("^")][:100]

cache_full = {}
for s in uni + ["SPY"]:
    try:
        df = v.split_adjusted(s, asof=END.to_pydatetime(), ca=ca)
        if df is not None and len(df):
            cache_full[s] = df
    except Exception:
        pass
spy = cache_full.pop("SPY", None)
days = [d for d in spy.index if WARMUP_START <= d <= END]
print(f"warmup {days[0]:%Y-%m-%d} -> anchor {ANCHOR:%Y-%m-%d} -> end {END:%Y-%m-%d}  ({len(days)} days)")

book = MaximizerBook(N_POS)
eq_at, brake_at, held_at = {}, None, None
for D in days:
    dc = {s: df.loc[df.index <= D] for s, df in cache_full.items()}
    dc = {s: df for s, df in dc.items() if len(df)}
    if D == ANCHOR or (brake_at is None and D >= ANCHOR):
        brake_at = book._vol_scale_factor(); held_at = len(book.pos)
    src, cands = build_daily_signals(dc, "rotating_bull", None, D, max_positions=N_POS)
    bc = [{"symbol": c["symbol"], "hold": SLEEVE_HOLD[src]} for c in cands] if src != "t30v" else []
    price_of = {s: float(df["close"].iloc[-1]) for s, df in dc.items()}
    eq = book.advance_day(D, src, bc, price_of, core_ret=0.0)
    eq_at[D.strftime("%Y-%m-%d")] = eq

anchor_s = ANCHOR.strftime("%Y-%m-%d")
# nearest trading day >= anchor
akey = next(k for k in eq_at if k >= anchor_s)
ekey = max(eq_at)
seg = (eq_at[ekey] / eq_at[akey] - 1) * 100
spy_a = float(spy.loc[spy.index <= pd.Timestamp(akey), "close"].iloc[-1])
spy_e = float(spy.loc[spy.index <= pd.Timestamp(ekey), "close"].iloc[-1])
spy_seg = (spy_e / spy_a - 1) * 100
print(f"\nAt anchor {akey}: brake factor={brake_at:.2f} (1.0=no brake), held positions={held_at}")
print(f"WARMED Maximizer {akey} -> {ekey}: {seg:+.2f}%")
print(f"  vs SPY: {spy_seg:+.2f}%   Core/Preserver: -7.5%   COLD-start replay was: -22.2%")

#!/usr/bin/env python3
"""Does Maximizer produce positive alpha in BULL runs? Warm-started (never cold), segment
return vs Core(t30v) and SPY over two clean recent bull windows. Survivorship-free panel."""
import os, sys
import pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "backend"); os.environ.setdefault("LAMBDA_ROLE", "worker")
import pitfwu_wf as pwf
from app.services.maximizer_service import MaximizerBook
from app.services.maximizer_signal_service import build_daily_signals
from app.services.maximizer_sleeves import SLEEVE_HOLD

v = pwf.v
panel, ca = v.load_panel(), v.load_corp_actions()

def _cache(uni, end):
    c = {}
    for s in list(uni) + ["SPY"]:
        try:
            df = v.split_adjusted(s, asof=end.to_pydatetime(), ca=ca)
            if df is not None and len(df):
                c[s] = df
        except Exception:
            pass
    return c

def max_seg(warm, anchor, end):
    uni = [s for s in v.universe_asof(warm.to_pydatetime(), 400, panel)
           if s not in pwf._EXCLUDED_SET and not s.startswith("^")][:100]
    cache = _cache(uni, end); spy = cache.pop("SPY")
    days = [d for d in spy.index if warm <= d <= end]
    book = MaximizerBook(15); eq = {}
    for D in days:
        dc = {s: df.loc[df.index <= D] for s, df in cache.items()}; dc = {s: df for s, df in dc.items() if len(df)}
        src, cands = build_daily_signals(dc, "rotating_bull", None, D, max_positions=15)
        bc = [{"symbol": c["symbol"], "hold": SLEEVE_HOLD[src]} for c in cands] if src != "t30v" else []
        po = {s: float(df["close"].iloc[-1]) for s, df in dc.items()}
        eq[D] = book.advance_day(D, src, bc, po, core_ret=0.0)
    ks = sorted(eq); ak = next(k for k in ks if k >= anchor)
    return (eq[ks[-1]] / eq[ak] - 1) * 100, spy

def t30v_seg(warm, anchor, end):
    r = pwf.run(warm.to_pydatetime(), end.to_pydatetime(), vb_on=True, trail=0.30, max_pos=20, size=0.045)
    ec = r["equity_curve"]; astr = anchor.strftime("%Y-%m-%d")
    ea = next(p["equity"] for p in ec if p["date"] >= astr)
    return (ec[-1]["equity"] / ea - 1) * 100

def spy_seg(spy, anchor, end):
    a = float(spy.loc[spy.index <= anchor, "close"].iloc[-1]); e = float(spy.loc[spy.index <= end, "close"].iloc[-1])
    return (e / a - 1) * 100

WINDOWS = [
    ("steady bull Apr25->Jan26", pd.Timestamp("2025-03-15"), pd.Timestamp("2025-04-30"), pd.Timestamp("2026-01-30")),
    ("sharp rally Mar->May 26",  pd.Timestamp("2026-02-10"), pd.Timestamp("2026-03-31"), pd.Timestamp("2026-05-29")),
]
print(f"{'window':28} {'Maximizer':>10} {'Core t30v':>10} {'SPY':>8}  | Max alpha vs SPY / vs Core")
for name, warm, anchor, end in WINDOWS:
    m, spy = max_seg(warm, anchor, end)
    t = t30v_seg(warm, anchor, end)
    s = spy_seg(spy, anchor, end)
    print(f"{name:28} {m:>+9.1f}% {t:>+9.1f}% {s:>+7.1f}%  | {m-s:+.1f}pp / {m-t:+.1f}pp", flush=True)
print("DONE", flush=True)

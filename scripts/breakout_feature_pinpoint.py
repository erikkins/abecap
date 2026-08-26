#!/usr/bin/env python3
"""Pinpoint WHICH breakout feature/condition diverges. For top-disagreement symbols, compute
each sub-condition from research feats (shape_tpe.features) AND from production's inline
recompute (as in maximizer_sleeves.breakout_signal), on the SAME OHLCV. On cells where prod
fires but research doesn't, tally which condition flipped; also report per-feature max abs diff.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "backend"); os.environ.setdefault("LAMBDA_ROLE", "worker")
for _l in open(".env"):
    if _l.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = _l.strip().split("=", 1)[1]; break
from shape_tpe import load_data
from regime_allocator_v2 import BREAKOUT as P

FULL_S, END = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-05-29")
arrs, close, dvols, feats = load_data(FULL_S, END)

def prod_feats(o, h, l, c, vol):
    cs = pd.Series(c); vs = pd.Series(vol)
    return dict(
        ma50=cs.rolling(50, min_periods=10).mean().to_numpy(),
        ma200=cs.rolling(200, min_periods=50).mean().to_numpy(),
        hi50_1=cs.rolling(50, min_periods=15).max().shift(1).to_numpy(),
        vol50=vs.rolling(50, min_periods=10).mean().to_numpy(),
        mom126=(cs / cs.shift(126) - 1.0).to_numpy(),
        c1=cs.shift(1).to_numpy(), c=c, vol=vol)

def conds(f):
    return dict(
        uptrend=(f["ma50"] > f["ma200"]) & (f["c"] > f["ma50"]),
        broke=f["c"] > f["hi50_1"] * (1 + P["buffer"]),
        fresh=f["c1"] <= f["hi50_1"],
        volspike=f["vol"] > P["vol_mult"] * f["vol50"],
        lead=f["mom126"] > P["mom_min"],
        pxfilt=(f["c"] >= 15) & (f["vol"] >= 500_000))

FEATS = ["ma50", "ma200", "hi50_1", "vol50", "mom126", "c1"]
for sym in ["BIDU", "LVS", "CRM", "AAPL"]:
    if sym not in feats: continue
    o, h, l, c, vol, idx = arrs[sym]
    rf = {k: np.asarray(feats[sym].get(k), float) for k in FEATS}
    rf["c"] = c; rf["vol"] = vol
    pf = prod_feats(o, h, l, c, vol)
    n = len(c)
    print(f"\n=== {sym} ===", flush=True)
    for k in FEATS:
        a, b = rf[k][:n], pf[k][:n]
        m = ~(np.isnan(a) | np.isnan(b))
        md = np.nanmax(np.abs(a[m] - b[m])) if m.any() else float("nan")
        print(f"  feat {k:7} max|res-prod| = {md:.6g}", flush=True)
    rc, pc = conds(rf), conds(pf)
    rsig = rc["uptrend"] & rc["broke"] & rc["fresh"] & rc["volspike"] & rc["lead"] & rc["pxfilt"]
    psig = pc["uptrend"] & pc["broke"] & pc["fresh"] & pc["volspike"] & pc["lead"] & pc["pxfilt"]
    extra = psig & ~rsig  # prod fires, research doesn't
    print(f"  prod-only cells: {int(np.nansum(extra))} (res={int(np.nansum(rsig))} prod={int(np.nansum(psig))})", flush=True)
    for k in ["uptrend", "broke", "fresh", "volspike", "lead", "pxfilt"]:
        flipped = int(np.nansum(extra & pc[k] & ~np.nan_to_num(rc[k]).astype(bool)))
        print(f"    cond {k:8} True-in-prod-only among extra: {flipped}", flush=True)
print("DONE", flush=True)

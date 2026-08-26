#!/usr/bin/env python3
"""Bisect the breakout gap: compare the SIGNAL matrices directly (no backtest). Same OHLCV
arrays (from shape_tpe.load_data) fed to BOTH: research detect(feats,'breakout') vs production
maximizer_sleeves.breakout_signal. Count True cells + agreement. Differ -> feature/signal
computation bug in one. Identical -> it's the sleeve LOOP mechanics (simulate vs replay_sleeve).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "backend"); os.environ.setdefault("LAMBDA_ROLE", "worker")
for _l in open(".env"):
    if _l.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = _l.strip().split("=", 1)[1]; break
from shape_tpe import load_data, detect
from regime_allocator_v2 import BREAKOUT as RES_BK
from app.services.maximizer_sleeves import breakout_signal, BREAKOUT as PROD_BK

FULL_S, END = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-05-29")
print(f"research params: {RES_BK}")
print(f"prod params:     {PROD_BK}", flush=True)

arrs, close, dvols, feats = load_data(FULL_S, END)
research_sig = detect(feats, RES_BK, "breakout")   # {sym: bool Series}

tot_r = tot_p = both = r_only = p_only = 0
worst = []
for s in feats:
    o, h, l, c, vol, idx = arrs[s]
    prod = breakout_signal(o, h, l, c, vol, PROD_BK).astype(bool)
    res = research_sig[s].reindex(idx).fillna(False).to_numpy().astype(bool) if s in research_sig else np.zeros(len(idx), bool)
    n = min(len(prod), len(res)); prod, res = prod[:n], res[:n]
    tot_r += int(res.sum()); tot_p += int(prod.sum())
    both += int((res & prod).sum()); r_only += int((res & ~prod).sum()); p_only += int((prod & ~res).sum())
    d = int((res != prod).sum())
    if d:
        worst.append((s, d, int(res.sum()), int(prod.sum())))

print(f"\nTotal breakout signals: research={tot_r}  production={tot_p}", flush=True)
print(f"agree(both True)={both}  research-only={r_only}  production-only={p_only}", flush=True)
print(f"symbols with any disagreement: {len(worst)} / {len(feats)}", flush=True)
worst.sort(key=lambda x: -x[1])
print("top disagreements (sym, #diff-cells, res-True, prod-True):", flush=True)
for x in worst[:8]:
    print(f"  {x}", flush=True)
print("DONE", flush=True)

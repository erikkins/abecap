#!/usr/bin/env python3
"""WIDER confirmation sweep of ENTRY PACING on t30v.

Confirms whether pace=8 (Pareto-beat baseline on the 10-start test) holds across
MANY more start dates AND in a bull-heavy regime (not just the 2022-bear windows).

Monthly starts 2021-01 .. 2023-06, fixed 3-year windows (survivorship-free PITFWU
panel). Reports the distribution for ALL starts, plus BEAR-heavy (<2023) and
BULL-heavy (>=2023, windows mostly 2023-2026) subsets. Needs /tmp/sectors_cache.json.

    source backend/venv/bin/activate
    python3 scripts/entry_throttle_multistart.py
"""
import os, sys, time, statistics
from datetime import datetime
from dateutil.relativedelta import relativedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "backend"))
os.environ.setdefault("LAMBDA_ROLE", "worker")
_envp = os.path.join(HERE, "..", ".env")
if os.path.exists(_envp) and not os.environ.get("DATABASE_URL"):
    for line in open(_envp):
        if line.startswith("DATABASE_URL="):
            os.environ["DATABASE_URL"] = line.strip().split("=", 1)[1]; break

import pitfwu_wf as pwf

STARTS = []
d = datetime(2021, 1, 4)
while d <= datetime(2023, 6, 1):
    STARTS.append(d)
    d = d + relativedelta(months=1)
WIN_YEARS = 3
T30V = dict(vb_on=True, trail=0.30, max_pos=20, size=0.045)
CONFIGS = [
    ("baseline", dict()),
    ("pace=6",   dict(max_entries_per_rebalance=6)),
    ("pace=8",   dict(max_entries_per_rebalance=8)),
]
BULL_CUTOFF = datetime(2023, 1, 1)   # starts >= this → 3y windows are bull-dominated


def summarize(name, rows):
    if not rows:
        print(f"  {name:10} — no results"); return
    ann = [r["ann"] for r in rows]; shp = [r["sharpe"] for r in rows]; mdd = [r["mdd"] for r in rows]
    print(f"  {name:10} n={len(rows):2}  "
          f"ann med={statistics.median(ann):6.2f} min={min(ann):6.2f} | "
          f"sharpe med={statistics.median(shp):.2f} min={min(shp):.2f} | "
          f"mdd med={statistics.median(mdd):5.2f} max={max(mdd):5.2f}")


rows = {name: [] for name, _ in CONFIGS}   # each row: {start, ann, sharpe, mdd}
t0 = time.time()
for st in STARTS:
    en = st + relativedelta(years=WIN_YEARS)
    for name, kw in CONFIGS:
        try:
            m = pwf.run(st, en, **T30V, **kw); m["start"] = st
            rows[name].append(m)
            print(f"[{st:%Y-%m}->{en:%Y-%m}] {name:10} ann={m['ann']:6.2f} sharpe={m['sharpe']:.2f} mdd={m['mdd']:5.2f}", flush=True)
        except Exception as ex:
            print(f"[{st:%Y-%m}] {name}: ERROR {ex}", flush=True)

for label, filt in [("ALL", lambda r: True),
                    ("BEAR-heavy starts (<2023)", lambda r: r["start"] < BULL_CUTOFF),
                    ("BULL-heavy starts (>=2023)", lambda r: r["start"] >= BULL_CUTOFF)]:
    print(f"\n===== {label} =====", flush=True)
    for name, _ in CONFIGS:
        summarize(name, [r for r in rows[name] if filt(r)])
print(f"\nDONE  ({time.time()-t0:.0f}s total)", flush=True)

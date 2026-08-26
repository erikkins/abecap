#!/usr/bin/env python3
"""t30v sector-cap sweep — does capping help the CURRENT live strategy (t30v),
evaluated as a Preserver overlay (Core stays cap=0 for marketing-baseline parity).

Runs pwf.run at t30v config (trail=0.30 / max_pos=20 / size=0.045) across sector
caps over the survivorship-free PITFWU panel. Cap method = walk_forward_service's
validated list-order per-sector filter (needs /tmp/sectors_cache.json).

    source backend/venv/bin/activate
    python3 scripts/sector_cap_t30v_sweep.py
"""
import os, sys, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
os.environ.setdefault("LAMBDA_ROLE", "worker")
_envp = os.path.join(HERE, "..", ".env")
if os.path.exists(_envp) and not os.environ.get("DATABASE_URL"):
    for line in open(_envp):
        if line.startswith("DATABASE_URL="):
            os.environ["DATABASE_URL"] = line.strip().split("=", 1)[1]
            break

import pitfwu_wf as pwf

WINDOWS = [("2021-01-04", "2026-05-29")]   # ~5y canon window; add more start dates to confirm
CAPS = [0, 3, 6, 8]
T30V = dict(vb_on=True, trail=0.30, max_pos=20, size=0.045)

for a, b in WINDOWS:
    s, e = datetime.fromisoformat(a), datetime.fromisoformat(b)
    print(f"\n=== t30v (trail30/20pos/4.5%) {a} -> {b} ===", flush=True)
    print(f"{'cap':>4} {'ann%':>8} {'sharpe':>7} {'mdd%':>8} {'calmar':>7} {'total%':>9} {'trades':>7}", flush=True)
    for cap in CAPS:
        t = time.time()
        try:
            m = pwf.run(s, e, sector_cap=cap, **T30V)
            print(f"{cap:>4} {m['ann']:>8.2f} {m['sharpe']:>7.2f} {m['mdd']:>8.2f} "
                  f"{m['calmar']:>7.2f} {m['total']:>9.1f} {m['trades']:>7}   ({time.time()-t:.0f}s)", flush=True)
        except Exception as ex:
            print(f"{cap:>4}  ERROR: {ex}", flush=True)
print("\nDONE", flush=True)

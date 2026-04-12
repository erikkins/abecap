---
name: Strategy optimization — Apr 9-12, 2026 (FINAL: Trial 37)
description: Comprehensive strategy optimization. Final config: TPE Trial 37 — +240% avg, 24% MaxDD, ~28% annualized.
type: project
originSessionId: 7dc69abd-ade1-4ef8-b901-42d3cee7df53
---
## FINAL CONFIG: TPE Trial 37 (Deployed Apr 12, 2026)

```
dwap_threshold_pct: 6.5%  (was 5.0%)
trailing_stop_pct: 13%    (was 12%)
near_50d_high_pct: 5%     (unchanged)
max_positions: 8           (was 6)
position_size_pct: 17%     (was 15%)
bear_keep_pct: 0.1         (keep top 10% during regime exit)
pyramid_threshold_pct: 15% (add when up 15%)
pyramid_size_pct: 2.5%     (small add)
pyramid_max_adds: 2
profit_lock_pct: 15%       (tighten stop when up 15%)
profit_lock_stop_pct: 5%   (tightened stop)
```

### Results (8 start dates, 2021-2026):
- **Avg: +240%** (~28% annualized)
- **Sharpe: 0.89**
- **MaxDD: 24% avg** (worst 25.3%)
- **All start dates positive** (worst +83%)
- **10-Year: +603%**
- **2023: -4%** (known weakness — narrow leadership market)

### What Was Tested (Exhaustive)
- Position sizing: 6/8/10/12/15 positions at various sizes
- Breakout filter: 3%/5%/8% near_50d_high
- Regime exit: 200MA vs panic-only (200MA wins)
- Profit lock: 15→6%, 30→10% (hurts returns)
- Pyramiding: various combos (25/5/1 best in combo with other levers)
- Bear keep: gradual regime exit (was dead code — fixed)
- RS Leaders secondary strategy (ABANDONED — start-date sensitive)
- Megacap fallback: 52-week high entry, own_ma50 exit, regime delay (couldn't fix 2023)
- Regime confirmation delay: 5/10/15/20 days (makes it WORSE)
- TPE Run 1: 50 trials, 4 objectives, 11 params — found Trial 37
- TPE Run 2: 30 trials, megacap-specific for 2023 (marginal improvement only)

### 2023 Problem (Accepted)
- Narrow leadership (Mag 7) + SPY wobbling around 200MA = whipsaw exits
- Every attempt to fix 2023 either blows up 2022 protection or increases MaxDD
- Accepted as structural cost of sub-25% MaxDD
- Marketing: don't show year-by-year, lead with 5-year total

### Marketing (Current):
- No year-by-year tables anywhere
- Lead with: +240% / ~28% ann / 24% MaxDD
- 2022 highlight: +6% while SPY -20%
- No "never a losing year" claim

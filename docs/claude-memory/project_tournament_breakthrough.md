---
name: Tournament breakthrough — 19.9% avg annualized (Apr 3 2026)
description: Parameter tournament found near_50d_high_pct=3% is the key lever. 7/7 positive, 3/7 hit 20%+, avg 19.9% ann.
type: project
---

## The Breakthrough (Apr 3 2026)

**One param change:** `near_50d_high_pct` from 5% → 3% (require stocks within 3% of 50-day high instead of 5%)

### Results — 7 Start Dates
| Start | Return | Ann% | Sharpe | MaxDD |
|-------|--------|------|--------|-------|
| Jan 18 | +266.9% | 29.7% | 0.95 | -20.4% |
| Jan 25 | +164.6% | 21.5% | 0.88 | -23.0% |
| Feb 1 | +162.1% | 21.3% | 0.91 | -22.4% |
| Feb 8 | +130.2% | 18.1% | 0.81 | -23.9% |
| Mar 1 | +129.2% | 18.0% | 0.90 | -16.6% |
| Jan 4 | +117.0% | 16.8% | 0.82 | -18.9% |
| Feb 15 | +92.6% | 14.0% | 0.71 | -19.1% |

**Avg: 19.9% ann | 7/7 positive | 3/7 ≥20% | All beat SPY**

### vs Previous Best (Warmup 13)
- Old: 11.4% avg ann, 7/7 positive, 2/7 ≥20%
- **New: 19.9% avg ann, 7/7 positive, 3/7 ≥20%** — nearly doubled

### How We Found It
1. Parameter tournament: 16 single-param sweeps in parallel (6.5 min on M4 Max)
2. Identified top 6 params by impact
3. Tested combined tournament winners → +146% flat backtest but only +37% in WF (8% stop killed it)
4. Kept trailing_stop at 12% (proven in WF), applied only near_50d_high_pct=3% → **+162% on Feb 1**
5. Validated across 7 start dates ��� consistent

### Why It Works
Tightening from 5% to 3% of 50-day high = only entering stocks at genuine breakout points. The extra 2% filter eliminates "almost breaking out" stocks that often fail. Stronger entry quality → better win rate → better returns.

### Key Lesson
The optimizer was searching 17 dimensions when only 1 mattered for WF mode. The tournament approach (test each param independently) found this in 6.5 minutes vs weeks of WF runs.

### Config for Production
```
near_50d_high_pct: 3.0  (changed from 5.0)
trailing_stop_pct: 12.0  (unchanged)
dwap_threshold_pct: 5.0  (unchanged)
max_positions: 6  (unchanged)
position_size_pct: 15.0  (unchanged)
All other params: baseline (Job 224)
```

### Next Steps
- Test with 2% and 4% to confirm 3% is the sweet spot (or if even tighter is better)
- Update production strategy config
- Run live for 1-2 months to validate out-of-sample

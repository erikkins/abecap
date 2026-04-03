---
name: 100-trial local WF result (Apr 3 2026)
description: First local run with 100 Optuna trials — modest improvement over 30 trials, not the unlock to 20%
type: project
---

## 100-Trial Local Run (Apr 2-3 2026)
- **Config:** Feb 1 → Feb 1, v2m, 200 symbols, rp=0.8, 100 trials
- **Result:** +122.4% (17.3% ann), Sharpe 0.90, MaxDD -24.9%
- **Baseline (30 trials, Job 224):** +112.8% (16.3% ann), Sharpe 0.83, MaxDD -17.9%
- **Delta:** +1.0pp annualized, +0.07 Sharpe, but -7.0pp worse drawdown
- **Runtime:** ~8 hours on M4 Max

**Why:** 100 trials finds slightly better returns but at higher risk. The search space (17 params) is the constraint, not exploration depth.

**How to apply:** More trials isn't the path to 20%. Need to expand parameter space with new signals/indicators. 100 trials is useful once we have better params to optimize.

## Conclusion
The optimizer is NOT starved for exploration — the search space itself is the bottleneck. Path to 20% is through **new parameters/signals**, not more trials of the same 17 params.

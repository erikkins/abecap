---
name: Optimizer expansion — more trials, more params, local runs
description: Plans to run optimizer with 100+ trials locally and explore additional parameters beyond the current 17
type: project
---

## The Problem (Apr 2 2026)
The optimizer has been handicapped by Lambda's 900s timeout. Forced to use 30 trials (barely explores the search space). Fixed params (no optimizer, Job 224) at 16.3% ann still beats every AI optimizer config.

## More Trials
- **30 trials** (current): barely explores 17-param space. Local optima near warm start.
- **100 trials**: meaningful exploration. TPE sampler has enough data to model the objective.
- **500 trials**: near diminishing returns for 17 params but finds global optima reliably.
- **1000+**: probably overkill unless landscape is very noisy.
- Each trial = one 60-day backtest. ~10-15s per trial on Lambda, similar locally.
- **100 trials × 131 periods = 13,100 backtests per WF run. ~3-5 hours locally.**

## Run Locally
- Local machine has no 900s timeout constraint
- Backend venv has numpy 1.26.3 (matches Lambda), optuna installed
- 7y pickle downloaded to `backend/data/all_data.pkl.gz` (256 MB)
- 10y pickle available on S3 (`all_data_10y.pkl.gz`, 363 MB) — OOMs on Lambda (3008 MB) but fine locally
- Connect to prod RDS for result persistence
- Script: `scripts/local_wf_runner.py` (being built)

## Additional Parameters to Explore
Beyond the current 17 optimizer params:
1. **100-day MA** — intermediate trend filter between 50 and 200 DMA
2. **ADX (Average Directional Index)** — measures trend strength, could replace/supplement momentum composite
3. **ATR-based stops** — volatility-adaptive stops instead of fixed %. Stock-specific risk.
4. **Sector momentum** — rotate into hot sectors, not just hot stocks
5. **Earnings avoidance** — skip entries within N days of earnings (reduces blowup risk)
6. **Volume profile** — differentiate between volume spikes on up days vs down days
7. **Relative strength vs SPY** — RS line slope as entry filter

**Priority:** Run 100+ trials with existing params first (low-hanging fruit), then add new params one at a time to measure marginal value.

## How to Apply
- Build `scripts/local_wf_runner.py` that loads pickle, connects to prod DB, runs WF with configurable n_trials
- Run overnight on local machine (prevent sleep: `caffeinate -i python3 scripts/local_wf_runner.py`)
- Compare 100-trial results vs 30-trial to quantify improvement
- If 100 trials significantly improves consistency, consider EC2 spot instances for production WF runs

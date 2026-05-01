---
name: Wire Circuit Breaker / Cascade Guard into production
description: Production scanner does not pause after consecutive trailing-stop hits the way the WF backtester does. Subscribers don't get the protection that backtest numbers reflect. Schedule for early-week May 4-5 2026.
type: project
originSessionId: 1081317d-f863-470f-ab07-65ddc614e856
---
**The gap:** Circuit Breaker (a.k.a. Cascade Guard externally) lives only in:
- `backend/app/services/walk_forward_service.py`
- `backend/app/services/backtester.py`
- `backend/app/services/strategy_params_v2.py`
- `backend/app/services/strategy_analyzer.py`

It is NOT in:
- `backend/app/services/scanner.py` (production daily scan)
- `backend/app/services/model_portfolio_service.py` (production portfolio decisions)
- `backend/app/services/ensemble_signal_service.py` (production signal generation)

**Why this matters:** Per the [WF ↔ production parity rule](feedback_wf_prod_parity.md), every lever proven in backtest must exist in production. Marketing claims (+675% / 21.6% ann / 28% MaxDD) include CB-pause protection. Subscribers don't get that protection today. If a cascade event happens, production keeps trading where backtest would have paused.

**When:** Schedule for early week of May 4 2026, after this weekend's CloudFront API cutover.

**Scope:**
1. Read CB params from active strategy (`strategy_params_v2.circuit_breaker_stops`, `circuit_breaker_pause_days`, `circuit_breaker_tighten_pct`).
2. Track consecutive-stop count and pause-state in `model_portfolio_service` — likely needs a new DB column or S3 state file (state must persist across Lambda cold starts).
3. When threshold hits, scanner skips signal generation OR `model_portfolio_service` rejects new entries until pause expires.
4. CB tighten: when re-engaging post-pause, tighten trailing stop for first N positions.
5. Surface CB state in admin dashboard (already shown in WF runs — needs equivalent for live).

**Validation:**
- Trigger a synthetic CB event in dev (force consecutive stops) and verify pause behavior matches backtester.
- Re-run latest WF results post-deployment to confirm parity claim is now defensible.

**Related work:**
- Intraday SELL validation (kicked off Apr 30 2026) — closes the OTHER parity gap (intraday stop in production but not backtester).
- Both gaps need closing for the [WF ↔ prod parity rule](feedback_wf_prod_parity.md) to be honored.

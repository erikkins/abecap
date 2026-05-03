---
name: TPE research — find intraday-stop sub-strategies that DO help
description: b-full showed default intraday trailing stops cost 17 pp annualized over 7y. Production is now EOD-only by default. Open question: are there parameter combinations where intraday DOES help (high-vol regimes, post-peak only, multi-min confirm)? Run TPE.
type: project
originSessionId: 1081317d-f863-470f-ab07-65ddc614e856
---
**Why this exists:** Apr 30 - May 3 2026 b-full WF result (jobs 1226 vs 1227, 7y window) found intraday trailing stops cost the strategy ~17 pp annualized vs EOD-only. Production switched to EOD-only May 3 2026 via the `INTRADAY_TRAILING_STOP_ENABLED` kill switch.

**The remaining question:** the *default* intraday config loses, but is there a SUB-STRATEGY with parameters that flips it positive? If yes, we have a route to (a) re-enable intraday selectively for protection on real catastrophes without giving up bull-market upside, or (b) build a separate "Bear Ripper" style sub-strategy that uses intraday during specific regimes.

**What to test (TPE search space):**

1. **Lockout window** — no intraday fire for first N days of trade. Avoids cutting winners short.
   - Range: 0, 3, 5, 7, 10, 14 days from entry
2. **Multi-minute confirmation** — trigger must hold for N consecutive minutes.
   - Range: 1, 2, 3, 5, 10 minutes
3. **Wider intraday stop** — e.g., 12% EOD, 18% intraday.
   - Range: 12-25% intraday (bracket the EOD width)
4. **VIX-conditional** — only fire intraday when VIX > X.
   - Range: VIX > 20, 25, 30, off
5. **Regime-conditional** — only fire intraday in panic_crash regime.
   - Boolean per regime
6. **Time-of-day** — no intraday fire in last hour of trading (let EOD handle).
   - Boolean
7. **Drawdown-conditional** — only fire intraday when position is already underwater.
   - e.g., only fire intraday if pnl_pct < -X% (catches catastrophes, not noise)

**Methodology:**
- Use the same b-full code path (set `intraday_aware=True` in WF, but with sub-parameters tuned)
- Strict OOS: TPE optimizes on 2019-2022, scores on 2023-2026
- Compare against EOD-only baseline (the new production default)
- Goal: find a config where total return ≥ EOD baseline AND MDD ≤ EOD baseline

**Pre-requisites:**
- 11y validation should land first (gives broader sample including 2018 correction)
- Code in backtester.py needs new params (`intraday_lockout_days`, `intraday_min_confirm_minutes`, etc.) — currently the b-full implementation is bare (just on/off)
- Real data anomaly fix (`project_intraday_data_anomalies.md`) to avoid TPE optimizing against bad data

**Schedule:** mid-to-late May 2026, after CB-in-production wiring + ticker-rename plumbing land. ~2-day project (extend backtester params + run TPE + analyze + ship if positive).

**Connected:**
- [Bear Ripper strategy](project_bear_ripper_strategy.md) — intraday might be the right tool there
- [Intraday data anomalies](project_intraday_data_anomalies.md) — fix execution model first
- [b-full validation](project_intraday_validation_apr30.md) — the finding that motivated this

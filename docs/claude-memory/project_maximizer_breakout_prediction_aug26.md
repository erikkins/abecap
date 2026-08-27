---
name: project_maximizer_breakout_prediction_aug26
description: Aug 26 2026 Maximizer breakout prediction to grade against the next actual scan
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Maximizer breakout prediction — made 2026-08-26 (close), grade vs next scan

Prediction from read-only `{"maximizer_preview":{"run":true}}` on 2026-08-26 data (regime, sleeve, firing entries, approaching radar). First real test of the fixed universe + rebuilt calendar. See [[session-progress]]; concept ties to [[project_prediction_ledger_idea]].

## Called (as of 2026-08-26 close)
- **Regime = rotating_bull → breakout sleeve ACTIVE** (necessary condition met; other regimes = no breakouts).
- **WT (WisdomTree)** — FIRING: fresh cross above 50d-high trigger, $24.76, $56M daily $-vol, NOT held. Clean data (2677 bars, fresh, px/dwap 1.52). Book 12/15 → room for 3. → **expect WT entered on the next scan cycle** (breakout_signal fires on the CROSS day = today, so it's a next-cycle entry, not a fresh Aug-27 cross).
- **BHVN (Biohaven)** — ON CUSP: 0.4% below trigger ($17.03), vol×5.34, +47% 6mo. Clean (2341 bars, px/dwap 1.47). → most likely NEXT to fire (maybe Aug 27 if it ticks up).
- **Radar approaching (8):** BHVN 0.4%, GEN 1.2%→$29.98, DBRG 1.5%→$16.2, TECK 2.1%→$72.62, BOX 2.2%→$34.11, SYF 3.0%→$82.13, HPQ 4.0%→$31.74, SHEL 4.2%→$94.96.

## How to grade (next session)
- Run `{"maximizer_preview":{"run":true}}` again + check actual entries: `build_todays_actions` (maximizer TierFills dated the scan day) OR MaximizerBookSnapshot diff.
- WIN = WT entered the book on the next cycle; partial-WIN = any radar name (esp. BHVN) fired/entered; MISS = regime flipped out of rotating_bull (sleeve went dormant) or WT didn't enter despite room.
- Note anything that reveals a gap (e.g. WT gated by is_series_tradeable in prod — production applies that gate AFTER build_daily_signals, my preview did not).

## RESULT (fill in next session)
- [ ] regime still rotating_bull? __
- [ ] WT entered? __
- [ ] BHVN / other radar fired? __
- [ ] surprises? __

---
name: 3.5-week silent signal drought — two-layer indicator-pipeline failure
description: No signals fired for 3.5 weeks because fetch_incremental stripped indicator columns AND _ensure_indicators only checked column-missing not NaN-tail. Both fixed; safeguards added.
type: feedback
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
## What broke

1. `fetch_incremental()` appends new rows for existing symbols and legitimately strips indicator columns (dwap, ma_50, ma_200, vol_avg, high_52w) so they get lazy-recomputed — this is the design.
2. `_ensure_indicators()` checked only "are these columns present?" — columns existed (on the old rows), so it skipped recompute. New appended rows had NaN in those columns.
3. Scanner's entry gates (DWAP > threshold, near_50d_high, etc.) silently returned False on NaN → zero signals for 3.5 weeks. No error, no alarm.

## Fixes in place

- `_ensure_indicators()` now triggers recompute on NaN-tail detection, not just missing columns (commit `5c26998`)
- `fetch_incremental()` explicitly calls recompute after append (belt-and-suspenders)
- Three safeguards added (commit `4c25eac`): indicator-freshness check in daily scan, admin alert if zero signals for N days, per-symbol NaN-tail assertion in export step

**Why:** Silent data corruption is the worst class of bug. Now there are three independent alarms before users see it.

**How to apply:**
- Any new indicator column added must be covered by `_ensure_indicators` NaN-tail check
- Never assume "column exists → data valid" — always verify the tail row
- Daily scan zero-signal outcome should page an admin, not just log
- If a signal drought exceeds 3 trading days and SPY regime is bullish, investigate pipeline immediately

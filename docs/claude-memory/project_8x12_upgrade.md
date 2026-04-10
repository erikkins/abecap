---
name: NEXT SESSION — Upgrade to 8@12% position sizing
description: Switch production from 6@15% to 8@12% and update all public-facing numbers. First priority next session.
type: project
originSessionId: 7dc69abd-ade1-4ef8-b901-42d3cee7df53
---
## First Thing Next Session (Apr 10, 2026)

### 1. Change production config: 6 positions @ 15% → 8 positions @ 12%

**Validated results (8@12%, 200MA exit, 7 random start dates, 2021-2026):**
- Average: +160.6% total return
- Avg Sharpe: 0.92 (was 0.73 with 6@15%)
- Avg MaxDD: 23.8%
- 7/7 positive, all beat SPY (+78%)
- Range: +84% to +267%
- Worst case (+84%) massively better than old worst case (-14%)

**Why:** 8 positions at 12% each provides better diversification in Rotating Bull regimes (which dominate 4 of 5 years) while still capturing upside in trending markets. Best risk-adjusted returns across all configurations tested.

### 2. Update all public-facing references

Files to update with new numbers:
- `frontend/src/TrackRecordPage.jsx` — yearly data, headline metrics, benchmarks
- `frontend/src/LandingPage.jsx` — hero stats, pricing section claims
- `frontend/src/Blog2022StoryPage.jsx` — 5-year table, comparison numbers
- `frontend/src/BlogBacktestsPage.jsx` — any performance references
- `frontend/src/BlogMarketCrashPage.jsx` — any performance references
- `backend/app/core/config.py` — MAX_POSITIONS, POSITION_SIZE_PCT
- Social posts (scheduled 456-467) — check if any reference old numbers
- CLAUDE.md — strategy documentation

### 3. Keep 200MA exit rule (not panic-only)

Tested both. Panic-only had worse drawdowns (35-40%) with similar or lower returns. The 200MA rule protects capital better even if it sometimes exits too early on individual trades.

### 4. Don't forget: `panic_only` config flag is deployed but defaults to False (correct)

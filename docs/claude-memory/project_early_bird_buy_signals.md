---
name: Early-bird buy signals — pre-close (3:45 PM) signal generation for premium tier
description: Generate buy signals 15-20 min before market close instead of post-close. Premium tier subscribers act before close, base tier gets normal next-day signal. Technical feasibility study + materiality test.
type: project
originSessionId: 1081317d-f863-470f-ab07-65ddc614e856
---
**Why this exists:** Current scan fires 4:30 PM ET (post-close, daily bars). Subscribers see signals that night, act next-day open — which means they buy at a different price than the signal record (RIVN slippage example: signal $17.74 → Erik bought next day at $16.36, 7.8 pp slippage). Question: can we close that gap by signaling pre-close so subscribers can buy SAME-DAY?

**Technical feasibility — YES, straightforward:**

1. **Data**: Alpaca minute bars are real-time on Pro SIP. We already have the `IntradayBarCache` module from May 1. At 3:45 PM ET, we can pull minute bars 9:30 AM → 3:44 PM and treat the 3:44 close as "today's close."

2. **Scanner changes**: minimal. The scoring code uses daily OHLC. If we substitute the 3:44 minute close for "today's close" and use intraday-built H/L/V from the day's bars so far, the same scoring logic runs. Estimated work: ~2-4 hours.

3. **Schedule**: new EventBridge cron firing at 3:45 PM ET on top of the existing 4:30 PM cron. Output flows to a SEPARATE channel (Pro tier only).

4. **Tier differentiation**: signals fired at 3:45 PM go to Pro subscribers; base tier still gets the 4:30 PM canonical scan. (Both can converge if scoring agrees, which is the empirical question below.)

**The empirical question — does it actually help?**

Three sub-questions:
1. **Signal stability**: of stocks that score in the top-N at 4:30 PM, how many were ALREADY in the top-N at 3:45 PM? If high overlap (>80%), early-bird gives the same signal earlier — clear value. If low overlap, early-bird is just noise.
2. **Price slippage**: how much does the entry price improve from 3:45 close vs 4:30 next-day-open? On average over 11 years.
3. **Subscriber actually-acts-on-it rate**: harder to measure pre-launch. Assume 50% act in the 15-min window if Pro is structured for it.

**Method:**
1. Pick 100 random scan-dates from 11y history.
2. For each, run scanner with 3:44 close (intraday cache) → record top-N.
3. Run scanner with 4:00 close (daily cache) → record top-N.
4. Measure overlap, slippage, missed-signals (signals at 4:00 that weren't visible at 3:45).
5. Translate to "potential alpha" by simulating the early-bird subscriber buying at 3:55 close vs the standard subscriber buying next-day-open.

**Tier pricing implication**: if early-bird produces +3-5 pp annualized advantage, justifies a Pro tier at $179-199/mo. If <2 pp, not worth the operational complexity.

**Legal flags (Erik to check separately):**
- Selectively timing the same signal differently to different paying tiers ≠ market manipulation per se (publishers offer time-tiered access routinely), but could draw scrutiny if Pro buyers systematically front-run base subscribers. Need attorney sign-off before launching.
- Easier defense if early-bird is a structurally DIFFERENT product (e.g., scoring methodology slightly differs), not "same signal on a delay."

**Schedule:** ~2-day project. Run after 11y validation lands and TPE-on-intraday research. Mid-to-late May 2026.

**Connected:**
- [Signal slippage tracking](project_signal_slippage_tracking.md) — early-bird is the operational solution to the slippage gap RIVN exposed
- [TPE on intraday execution](project_intraday_tpe_research.md) — different problem (sells), but uses same intraday infrastructure

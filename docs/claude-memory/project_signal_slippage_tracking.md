---
name: Signal slippage tracking — post-publication price monitoring
description: Track actual achievable execution prices after signal publication to measure real-world slippage. Build data for methodology disclosure.
type: project
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
**Concept:** When a signal is published, automatically record the stock's price action for 15-60 minutes post-publication. Compare the published entry price to the VWAP or 30-minute post-publication median price.

**Why:** This isn't measuring any individual subscriber's execution — it's measuring what was *achievable* during the window subscribers were likely trying to execute. Builds real data to replace the "zero slippage" assumption in methodology with actual measured slippage.

**Implementation notes:**
- Trigger: signal publication event (daily scan completion → dashboard update → email send)
- Data collection: poll price every 1-5 min for 60 min post-publication via intraday monitor (already runs every 5 min during market hours)
- Store: S3 JSON per signal date, keyed by symbol
- Metrics: entry price vs 15-min VWAP, 30-min median, 60-min VWAP
- Report: average slippage by signal, rolling 30-day average

**Three metrics from one data collection:**
1. **Slippage**: published entry price vs achievable 30-min VWAP
2. **Capacity**: when subscriber volume starts moving price on thinner names (early warning to widen universe or cap subs)
3. **Signal decay**: how much of the edge remains by the time subscribers can act. If the move is 80% done by email delivery, signal has low actionability regardless of slippage.

**How to apply:** Once we have 30+ days of data, update methodology page with measured slippage instead of "zero slippage" assumption. Signal decay data informs whether to change publication timing (e.g., pre-market vs after-hours vs intraday). Capacity data tells us when to widen the universe or introduce subscriber caps.

Noted Apr 20, 2026. Not yet built.

---
name: Market data licensing — verify before charging subscribers
description: Need to confirm Alpaca Pro terms cover displaying live quotes to paying subscribers. Also evaluate websocket real-time as premium feature.
type: project
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
**Pre-launch question:** Does Alpaca Pro subscription cover displaying real-time quotes to authenticated paying subscribers via our web dashboard? Or do we need a separate market data redistribution agreement?

**Ask Alpaca sales:** "We display real-time quotes to authenticated paying subscribers through our web app. Do we need a separate market data agreement?"

**Also ask the securities attorney** — market data licensing is adjacent to the publisher's exemption analysis.

**Current setup:**
- Alpaca Pro SIP: EOD bars + 5-min quote polling for intraday monitor
- yfinance: VIX/index symbols + fallback (gray area for commercial use)

**Future consideration:**
- Alpaca websocket real-time streaming — could be premium feature
- Unknown cost — check Alpaca pricing for websocket/streaming tier
- FMP also offers real-time data in their commercial tiers

Noted Apr 24, 2026. Not yet resolved.

---
name: Newsletter has no signals or tickers
description: Market Measured is editorial only — same content for free and paid subscribers, no ticker symbols ever
type: feedback
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
The weekly newsletter (Market, Measured.) is purely editorial. It never contains ticker symbols, signal data, watchlists, or anything from the daily scan. Paid subscribers already get all of that in the daily digest email.

**Why:** The newsletter's job is voice, trust, education, and retention — not signal delivery. Splitting into free/paid versions with `show_symbols` would turn the editorial into a pitch vehicle, which kills the Matt Levine quality bar.

**How to apply:** One newsletter, one version, everyone gets it. Send to both `newsletter_preferences` (free list) AND active paid subscribers. Remove any `show_symbols` logic from the newsletter send flow. The `show_symbols` parameter in `send_market_measured` should always be False for the newsletter (or removed entirely from the newsletter path).

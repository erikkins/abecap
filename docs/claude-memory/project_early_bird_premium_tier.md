---
name: Early Bird premium tier — late-day signals for head start
description: Premium subscribers get signals at 3:45 PM instead of post-market. Requires FMP 1-min intraday data to validate.
type: project
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
**Concept:** Run the ensemble scan at 3:45 PM (15 min before close) and deliver "Early Bird" signals to premium subscribers. Standard tier sees signals at 6 PM post-market.

**Why:** Premium subscribers execute at 3:50 PM, capturing overnight gaps. Standard subscribers execute next morning at 9:30 AM, 16+ hours later. The head start is the premium value.

**Benefits:**
- Natural capacity solution — fewer people executing at the same time
- Concrete differentiator for premium tier ($249/mo vs $129/mo)
- Dilutes signal impact by spreading executions across two windows

**Validation needed (requires FMP 1-min data):**
1. Run walk-forward with 3:45 PM prices instead of 4:00 PM close
2. Confirm same signals fire at 3:45 as at 4:00
3. Compare returns: early entry vs EOD entry vs next-morning entry
4. If 3:45 PM entries perform better, the premium tier has a proven edge

**Pricing structure:**
- Core: $129/mo — signals at 6 PM (after market close)
- Premium: $249/mo — early bird signals at 3:45 PM + Core features
- Annual Premium: $2,399/yr (~$200/mo)

**Dependencies:** FMP subscription ($139/mo for 1-min historical bars, 30-year range)

Noted Apr 24, 2026. Not yet built.

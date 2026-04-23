---
name: Bear Ripper shelved — 5-year drag, promising on individual bears
description: BR consistently -10 to -18pp drag on 5-year 8-date average despite being positive in each individual bear period. Shelved pending 11y validation.
type: project
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
**Status:** Shelved as of Apr 22, 2026.

**What worked:**
- Approach J (must be positive absolute + positive RS, 2 pos, 10% size, 20% stop, Price>$50, Vol>1M, Vol<60%)
- Positive in ALL THREE individual bears: 2018 +3%, COVID +27.6%, 2022 +4.9%
- Open universe (no hardcoded stock list, no look-ahead bias)

**What didn't work:**
- 5-year 8-date average consistently -10 to -18pp below CB-only baseline
- One bear in the 5-year window isn't enough to overcome friction during bull periods
- Regime detector whipsaws cause churn (brief bear→bull→bear flips)
- Trailing stop exits lock in small losses on volatile defensive names

**Key findings:**
- BR positions were being carried into ensemble slots (fixed: separate pools)
- Capital was leaking at period boundaries (fixed: always-close)
- Requiring positive absolute return (not just RS > 0) was the breakthrough for individual bear performance
- The +80% COVID result was mostly ensemble recovery, not BR defensive picks

**Next steps (when revisited):**
- Wait for 11y TPE to finish — extract per-period params, replay with Approach J on 11y pickle
- If 11y shows net positive across all three bears with enough magnitude to overcome bull-period drag, BR is viable
- Consider as a premium add-on product rather than core strategy component

Noted Apr 22, 2026.

---
name: Per-regime sub-strategies — deferred (overfit risk too high)
description: Bear Ripper / range-bound / per-regime sub-strategy work was considered Apr 28 2026 and shelved. 4-5 historical bear events is too small a sample for optimization. Captures the two-track framework if we revisit.
type: project
originSessionId: 1081317d-f863-470f-ab07-65ddc614e856
---
## Decision

**Deferred.** Not actively building per-regime sub-strategies. Clean-data canonical (+160% / 0.92 Sharpe / 20.4% MaxDD on 8-date 5y carry-on) is a defensible product as-is. Over-engineering with overfit sub-strategies dilutes the story.

**Why:** Across 11 years we have ~13 vol shocks, but only **4-5 true bear regimes (panic_crash trigger)**. TPE or any parameter optimization on that sample size will overfit catastrophically — the ±50-60pp swings we already see from a single binary CB-pause-carry lever flip are evidence we're at the simulation noise floor.

## How to apply

If a future session proposes "let's build a Bear Ripper / range-bound / per-regime sub-strategy with TPE," push back: 4-5 events doesn't support optimization. The threshold for revisiting is either (a) materially more bear events have occurred, or (b) a clearly-universal rule has been observed that doesn't require per-event fitting.

## The two-track framework (if we ever revisit)

**Track 1 — Disclosure (regime-PnL attribution), not optimization.**
- Tag every WF trade with the regime active at entry date.
- Publish per-regime stats: trades, win rate, avg return, hold duration, contribution to total PnL, vs SPY same windows.
- Story angle: *"We trade aggressively in strong_bull/weak_bull. Cascade Guard pauses us in panic_crash. Range_bound is statistically a soft spot — here are the numbers."*
- Why this is the right tool: trust > clever optimization. No competitor in retail signals publishes per-regime stats.

**Track 2 — Universal-rule enhancements only (no parameter fitting).**
Any addition must pass three tests:
1. Rule is statable in one sentence (no fitted parameters).
2. Evidence base extends beyond bear events alone.
3. Direction of effect confirmed on held-out data, magnitude not claimed.

**Candidate that passes:** Mega-cap basket during CG pause window. Across 12 vol shocks (clean 11y data, manual inspection 2026-04-28): mega-caps drop similar to SPY (-8 to -21%) but **post-shock recovery beats SPY in every cap**. Avg post-shock 14d returns: NVDA +10.99%, TSLA +9.75%, AAPL +7.93%, MSFT +6.67%, AMZN +6.29%, GOOGL +5.29%, META +1.82% vs SPY +6.25%. Realistic addition: ~+2-5pp lift on 5y avg if implemented as fixed-basket-during-pause. Simple rule, 12-event evidence, no fitting.

**Candidates that FAIL the test (do not pursue):**
- TPE-optimized bear sub-strategy
- TPE-optimized range-bound sub-strategy
- Per-regime parameter tuning of any kind on current sample size

## What killed enthusiasm

Two compounding issues:
1. **Sample size.** 4-5 bears is too few for any optimization to mean anything.
2. **Story is already strong without it.** Clean-data 20.4% MaxDD beats the 32% sim claim by 36%; +109% worst-case start beats the +86% published worst by 23pp; 21.1% ann lands within rounding of the 21.5% friction-adjusted claim. Adding sub-strategies risks contaminating a clean honest product with overfit noise.

## Implication for marketing

Today's marketing claims "7-regime intelligence" but the actual response logic is binary (Cascade Guard pauses on panic_crash; trade normally otherwise). That's defensible — but the narrative has more granularity than the code. **Don't escalate the regime-intelligence claim** without a real per-regime response system to back it. Track 1 (disclosure) would let us back the claim honestly. Without Track 1, soften the language ("regime-aware Cascade Guard" rather than "7-regime adaptive strategy").

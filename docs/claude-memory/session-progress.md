---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 2dce3134-d861-45c4-a371-80378750f8c0
---

# Session snapshot — Jul 30 2026

## Frozen spec (load-bearing)
- Tiers: everyone served **Preserver** floor (never raw t30v/Core). **Maximizer** = paid `has_maxpp_addon` OR admin `compmax`; = Preserver except rotating_bull → breakout book (N=15, vol-target). Flag-gated `TIER_SERVING` (both api + worker Lambdas).
- **NEVER expose publicly:** "t30v"/"Core"/"Option-B"/"N=15"/DWAP/capitulation overlay. Never "tape" (enforce via voice_filters). Never render "NaN". (Admin-only tabs may show internal labels.)
- Deploy = push main → wait for **"Deploy RigaCap"** workflow COMPLETE before resending emails (`gh run list --workflow "Deploy RigaCap"`; worker image can lag a beat after "success"). Long ops → invoke `rigacap-prod-worker` directly. Admin email/db targets erik@rigacap.com.

## Done this session — tier-book equity bugs FIXED + verified
Correct numbers now (as of 2026-07-29): **Core $91,543 · Preserver $91,543 · Maximizer $97,487**.
- **Maximizer**: was fake +42% ($142.8k) — Jul-24 rewrite double-counted cash loading an old-format snapshot. Rebased to clean $100k base (returns scale-invariant) → true −2.5%. Loader hardened. `{"maximizer_rebase"}` handler.
- **Preserver**: drifted ~0.7% above Core (parallel return-chain). Rewrote to equity = Core×factor (factor moves ONLY on real capitulation days) → penny-locked to Core when overlay dormant. `{"preserver_rederive"}` locked 30 historical snapshots. Lesson: derive from source of truth, don't run own equity chain.
- Polish: admin Strategies/Lab/Auto-Pilot hidden (`?tab=` still reaches); Simulated Portfolio shows full-cycle beside rolling-365; This Week widget scoped to served tier's capped book; admin Tier Books STR now shows Days+P&L + realtime intraday (polls /api/quotes/live); email rows restructured for phone + sentence-cased. Per-tier capital CLOSED (one tier/user).
- Commits: 56d3ec0 79b726e 798ac6b 35d6ee9 c8691a7 1a7068e e6b2a85 bd15be4 2460e11.

## Next / queued (Erik's call which first)
- Real beta **tier-announcement** blast (send_tier_announcement, knob email-knob-v3.png) → let beta list pick tier. Then Google Ads → buyers.
- Next-wave Maximizer widgets: book equity curve vs SPY; day-29 exit alerts (email/push); closed-trades STR; book stats.
- Subscriber-facing tier alerts (day-29 exit etc.).

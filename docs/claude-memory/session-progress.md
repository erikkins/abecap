---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 14–15 2026

## Frozen spec
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC. PRINT DOCS=ink-on-white+US-English; WEB=claret/paper. perf_numbers SSOT. Web deploy=push main→"Deploy RigaCap" GHA + smoke curl api.rigacap.com/api/market-data-status→200. Migration-first DB. NEVER lambda update-function-configuration --environment / terraform apply w/o plan. mobile-admin/=Expo EAS-OTA channel=preview. TIER_SERVING=true; 0 real external subs. Admin/test emails→erik@rigacap.com.

## ✅ SHIPPED (all live, deploys GREEN) — served-Maximizer "both books" portal + email
- **Portal layout FINAL:** [Preserver book | Maximizer book + Rotation watch] side-by-side, books-on-top, ONE capital control, Market Read PIXEL-aligned (JS measure `[data-market-read]`), whole shares NO ≈ ($ = exact target), candidates FULL-WIDTH below (no canyon). Date tag + "Last updated" on ONE row (two-book view). Rotation watch fills DYNAMICALLY (measured `rotRows` useLayoutEffect → rows reach `[data-books-left]` Preserver bottom; `data-rot-list/-max/-row`). Erik: NO wasted space.
- **Rotation watch** = 5+ nearest 29-day time-stops, urgency-sorted (book is weight-sorted), live hold-clocks, no WF. **Recently-closed** ("previously sold") = wired + DORMANT (prod: 15 maximizer BUYS Jul15-28, 0 sells; first exits ~mid-Aug) → auto-splits the space when sells log, both portal + email. No deploy needed.
- **Maximizer EMAIL:** breakout book renders from build_tier_book('maximizer') (portal parity) → per-name weight·whole-shares·$ scaled to recipient capital + vol-target exposure line; shared `_holding_row`; `breakout_tier_book` threaded through scheduler (built $100k, rescaled). Sent SAMPLE to erik@rigacap.com — Erik verifying.
- **Banner bug FIX:** /api/auth/refresh now mirrors /me subscription resolution (admin synthesis+override) — 11-min background refresh had been wiping admin's synthesized sub → SubscriptionBanner popped at top on idle.
- Earlier: Stripe-sourced admin stats (0 paid/$0 MRR, internal excl); admin app rebuilt+OTA'd.

## ▶ STILL OPEN
- DOCS refresh: signal-intel + tech-arch edits UNCOMMITTED; PDF re-export + investor/marketing/sales sweep + 3 Qs. Commit ONLY design/documents (scratch scripts/ untracked).
- ADS: watch search terms + cost-per-signup.

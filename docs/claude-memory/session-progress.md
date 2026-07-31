---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 2dce3134-d861-45c4-a371-80378750f8c0
---

# Session snapshot — Jul 31 2026

## Frozen spec
- Public copy: say "walk-forward" NOT "backtest"; never t30v/Core/sleeve/DWAP; never "tape"/"NaN"; "warts" OK in blog editorial voice but NOT in ads. Survivorship-free NOW OK in customer copy (within How-We-Test). STEER subs toward Maximizer (+$100/mo upsell). Deploy=push main→"Deploy RigaCap".

## OVERLAY CANONICAL (shipped) — SSOT = frontend/src/perf_numbers.js + backend/app/services/perf_numbers.py + docs/numbers-citations-registry.md §1.
- Preserver 21yr 7.7/0.87/−13.7, 5yr 13.0/1.28/−12.9, typical-yr +10.7%. Maximizer 21yr 13.5/0.93/−20.8, 5yr 31.4/1.51/−14.9, typical-yr +26.5%, 2022 −6.4% (the upsell hero). S&P 21yr 9.8/−55, 5yr 14.2/−25.4. Raw-mom 13.2/0.69/−57.

## ✅ GATE B PUBLIC PAGES DONE + PUSHED LIVE (commit f51bbac, "Deploy RigaCap" deploying ~3:10pm).
- All 7 wired to overlay: LandingPageV2, TrackRecordPageV2, MethodologyPageV2, ForAdvisersPage, 3 blogs (Blog2022 pivoted to Maximizer −6.4 hero). Nothing else public is stale.
- ⬜ FAST-FOLLOW (not pushed-blocking): backend emails (email_service ≥6 / newsletter_generator:121 / ai_content:56 → hardcode overlay, plain HTML strings so no f-string interp); SocialTab post library (needs 2-tier REWRITE, still single-strategy 8.3/19); IMAGE REGENS og-card.png ("backtest"+"19%") + launch-1..5.png (knob email-knob-v3.png = FINE, no perf numbers). fuller Maximizer-satellite narrative on /for-advisers.

## ▶ NOW: GOOGLE ADS SETUP — walking Erik through the UI step-by-step (no Ads API; he clicks, screenshots, I confirm). Doc = design/documents/google-ads-2tier-campaign.md (refreshed Jul 31, paste-ready copy+keywords+negatives).
- CONFIRMED: conversion tracking already live — **Sign-up = Active + Primary** (+ Purchases Active). So old campaign's 0-conv was a REAL land→signup leak (Erik right), NOT a tracking gap. Begin-checkout = Misconfigured but unused (ignore).
- Campaign so far: name `rigacap-signals-2tier`; objective Sales→Search; Website visits + rigacap.com; bidding = **Clicks + $6.50 max-CPC cap** (NOT Max Conversions — 0 history); customer-acquisition OFF; Networks = Search Partners OFF + Display OFF; Locations = US; Language = English; **AI Max = OFF** (keep control/clean data).
- NEXT UI STEPS: keyword/ad-group step → build Ad Group 1 "Preserve" (landing /track-record?utm_campaign=preserve) then Ad Group 2 "Maximize" (landing /?utm_campaign=maximize) + RSAs (paste headlines/descriptions from doc — lead Maximize with the 2022 −6% hook), then shared negatives list `rigacap-negatives`, then assets (sitelinks/callouts/structured snippet Preserve/Maximize), then review+publish. Budget $25/day.
- LAUNCH GATE: $350/2-week kill-or-scale; if clicks but ~0 signups at ~$150 → funnel/price leak, stop+fix (the real risk — cheap clicks were never the problem). Switch to Max Conversions/tCPA after ~15-30 conversions.

## CLEANUP temp scratch: scripts/{tier_vintages_today.py, tier_vintages_daily_today.py, recompute_canonical.py, canonical_recompute.json, tier_curves_21y_today.json, tier_curves_21y.json.bak-may29} + the breakout_*.py scratch. KEEP scripts/overlay_canonical.json.

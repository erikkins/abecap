---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 14–17 2026

## Frozen spec
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC; never say "tape" for the market ([[feedback_no_tape_brand_voice]]). PRINT DOCS=ink-on-white+US-English; WEB=claret/paper. perf_numbers SSOT. Web deploy=push main→"Deploy RigaCap" GHA + smoke api.rigacap.com/api/market-data-status→200. Migration-first DB. NEVER lambda update-function-configuration --environment / terraform apply w/o plan. Admin app=mobile-admin/ Expo EAS OTA (`eas update --channel preview`; Erik reopens app). TIER_SERVING=true; 0 real external subs. Admin/test emails→erik@rigacap.com. Email=SMTP/aiosmtplib; diagnose sends via email_events table. NO Google Ads API (Erik applies ad changes in UI).
- **RigaCap = PUBLISHER** (signals-only, no custody). No "your portfolio value" screen. Book equity = model MTM from $100k CAP0.

## ✅ SHIPPED this session (all live)
- Served-Maximizer PORTAL (2 books side-by-side, pixel-aligned reads, whole shares, Rotation watch measured-fill, candidates full-width). Recently-closed DORMANT (~mid-Aug first sells).
- Maximizer EMAIL: Maximizer-first order, scaled shares + vol-target, gauges (day-clock + cushion table bars), breakout radar replaces ensemble "Approaching". Weight rounds WHOLE %.
- ADMIN app: weight-sorted positions + equity "model book · from $100k · ±X%" + Recent transactions (OTA'd).
- **NEW /momentum door (c8b481b, live)**: Maximizer persona, hold-period-first (filters day-traders), "momentum with a floor", sets rigacap_want_maximizer, prerendered. Same funnel events as /should-i-sell → mom_funnel + sis_funnel both in /api/admin/pageviews/summary; web TrafficTab + mobile Ads tab render BOTH (OTA group dec5c012). **ExploreMore** band on both doors (fixes dead-end nav; explore_* events).
- Fixes: refresh-endpoint admin banner; weekend test-email gate; Stripe stats; "tape" removed everywhere (63e459b).

## ▶ AWAITING ERIK
- Google Ads: open "momentum trading signals" (Apply remove neg + add `live signals`/`scalping`/`options signals`) and point that ad group at **/momentum** (door now exists) — OR I point the ad group; Erik doing UI-side. Keep negative until he's ready.
- SIS funnel WATCH ([[project_sis_funnel_watch]]): baseline fold-through 36%; ~1wk/150 landers → redesign above-the-fold if still mid-30s. Now also watch /momentum funnel + explore_* clicks.

## ▶ STILL OPEN
- DOCS refresh: signal-intel + tech-arch UNCOMMITTED; PDF re-export + investor/marketing/sales sweep + 3 Qs. Commit ONLY design/documents.

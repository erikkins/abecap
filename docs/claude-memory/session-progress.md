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
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC; never say "tape" for the market ([[feedback_no_tape_brand_voice]]). PRINT DOCS=ink-on-white; WEB=claret/paper. perf_numbers SSOT. Web deploy=push main→"Deploy RigaCap" GHA + smoke api.rigacap.com/api/market-data-status→200. Admin app=mobile-admin/ EAS OTA (`eas update --channel preview`; Erik reopens). NO Google Ads API (Erik does ad UI). NEVER lambda update-function-configuration --environment / terraform apply w/o plan. RigaCap = PUBLISHER (no custody, no "your portfolio value" screen).

## ✅ SHIPPED this session (all live)
- Served-Maximizer PORTAL (2 books side-by-side, gauges, whole-shares, Rotation-watch fill). Maximizer EMAIL (Maximizer-first, scaled shares + vol-target, gauges, radar). Weight rounds WHOLE %. Admin app: weight-sorted positions + labeled equity + Recent transactions.
- Fixes: /api/auth/refresh admin banner; weekend test-email gate; Stripe stats; "tape" removed.
- **TWO AD DOORS**: /should-i-sell (Preserver) + NEW /momentum (Maximizer, hold-period-first, "momentum with a floor", sets want_maximizer, prerendered). Both pointed w/ UTM, plumbing verified. ExploreMore band on both (explore_* events). mom_funnel + sis_funnel both in /api/admin/pageviews/summary; web TrafficTab + mobile Ads tab show both (data-driven).
- **FUNNEL BLIND-SPOT FIXED (c88bc8c)**: ad-landing checkout goes via App.jsx auto-checkout effect (not AuthContext) which now fires `checkout_redirect` — before, ad-door conversions never logged "Reached Stripe" (the should-i-sell signup_open→0-Stripe = MEASUREMENT GAP, not dead funnel; historical 0 stays blind). Added `signup_submit`+`signup_success` (LoginModal: email/Google/Apple register) so signup_open→Stripe drop is localizable: open≫submit=modal friction; submit≫success=registration fail; success≫redirect=Stripe handoff. admin _funnel_order updated.

## ▶ WATCH / NEXT
- After a day+ of live door clicks: pull BOTH funnels (all steps) — see if 0-Stripe was a mirage or find the real leak step, then fix that one. Compare /momentum vs /should-i-sell. SIS fold-through: baseline 36%, today 52% (redesign call open, [[project_sis_funnel_watch]]).
- ADS (Erik UI): shared neg list CLEAN; ad-group intent split correct; "trading signals" removed; DON'T switch to conversion bidding (~0 conv); prune search-terms reactively.
- Recently-closed (email+portal) auto-activates on first real Maximizer sells (~mid-Aug).
- DOCS refresh STILL OPEN: signal-intel + tech-arch UNCOMMITTED; PDF re-export + investor/marketing/sales sweep + 3 Qs. Commit ONLY design/documents.
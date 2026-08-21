---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — thru Aug 21 2026

## Frozen spec
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC; never say "tape" ([[feedback_no_tape_brand_voice]]). PRINT DOCS=ink-on-white; WEB=claret/paper. perf_numbers SSOT. Web deploy=push main→"Deploy RigaCap" GHA + smoke api.rigacap.com/api/market-data-status→200. Admin app=mobile-admin/ EAS OTA. NO Google Ads API (Erik does ad UI). RigaCap=PUBLISHER (no custody). Research MUST map to prod penny-to-penny; verify before concluding; no fabrication.

## ✅ SHIPPED (bb3c9be, deploy GREEN + api 200, VERIFIED by me) — Maximizer vol-brake cold-start fix, option (c)
- FOUND (verified): live MaximizerBook vol-brake OFF entire life (bk_eq_hist<21→vol_scale=1.0), never warm-started; launched Jul8, rebased Jul24, now −1.69%/$98,311. This uncovered by the start-date-sweep question. Erik picked (c).
- FIX = COLD-WINDOW ESTIMATOR in _vol_scale: WARM (bk_eq_hist≥21) = certified return-stream computation BYTE-IDENTICAL (verified in diff; marketed WF penny-to-penny intact). COLD (<21) with holdings = value-weighted real held-names' returns from data_cache closes (√252, lagged/today-excluded, min(1,VOL_TARGET/rv)); empty/no-data→1.0 (no fabrication). recent_closes threaded run_shadow_day→advance_day→_vol_scale, built ONLY when cold. Forward-looking, current book self-warms ~1 day, NO live mutation, no retroactive equity change. Every FUTURE launch now brake-live day 1.
- Start-date SWEEP: now meaningful for future launches; historical sweep of CURRENT book still DEFERRED until warm+reproducible (~days) — offered to run then ([[project_maximizer_coldstart_jul21]]).
- FYI: design/documents/* still UNCOMMITTED (separate docs-refresh; fork didn't touch).

## ✅ SHIPPED recently (live)
- Market reads SPY-trend-aware + mandatory 3+ session streak mention (real facts only, verified live). Two ad doors /should-i-sell + /momentum + ExploreMore + dual funnel tracking. Funnel blind-spot fixed (checkout_redirect + signup_submit/success). Served-Maximizer portal + email (gauges, scaled shares). Admin app weight-sorted books. Whole-share display. Personal social launch posts (design/documents/personal-launch-social-posts.txt).

## ▶ WATCH / OPEN (Erik-side or wait)
- Funnel: wait a few days → compare /momentum vs /should-i-sell; /should-i-sell offer→CTA leak (2 of 19), signup modal (single-field + one-tap OAuth idea) ([[project_sis_funnel_watch]]). Ads: shared-neg clean, intent split correct, don't switch to conversion bidding (~0 conv).
- DOCS refresh STILL OPEN: signal-intel + tech-arch uncommitted; PDF re-export + investor/marketing/sales sweep. Commit ONLY design/documents.
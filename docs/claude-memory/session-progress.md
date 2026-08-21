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
- Start-date SWEEP now RUNNING (fork a48a7dd7d8de0060b, bg): uses the CERTIFIED WF engine (replay_sleeve('breakout')+vol_scaled_returns = marketed construction), WARM each variant (replay from ~30td before each launch date so brake warm by launch), measure ret L→today. ~28 launch dates Jun23–Aug1. ANCHOR GATE: must reproduce compute_tier_walkforward trailing-365 Maximizer number before trusting. Read-only handler {"maximizer_start_sweep":...}, no live mutation, doesn't touch WF/marketing code. Reports spread + where Jul-8 falls + TRANCHING takeaway (stagger onboarding?). Measures the STRATEGY's warm-braked launch-date sensitivity (how a real launch behaves post-(c)-fix), NOT the buggy cold live book (−1.69%, already characterized). Verified earlier: marketing WF numbers CORRECT (WF brakes properly; its ~21d cold-start is negligible over 5-21yr; (c) fix didn't touch WF).
- FYI: design/documents/* still UNCOMMITTED (separate docs-refresh; fork didn't touch).
- TIMING (verified Aug 21): bk_eq_hist=20 on Aug 20 (Thu); threshold=21. Certified own-history brake first engages MONDAY Aug 24 scan (loads len-21 saved after tonight's Fri scan). Today's scan still loads 20. BUT (c) cold-window estimator brakes off REAL holdings vol NOW (today + weekend), hands off to certified path Mon (tiny expected step at boundary).

## ✅ SHIPPED recently (live)
- Market reads SPY-trend-aware + mandatory 3+ session streak mention (real facts only, verified live). Two ad doors /should-i-sell + /momentum + ExploreMore + dual funnel tracking. Funnel blind-spot fixed (checkout_redirect + signup_submit/success). Served-Maximizer portal + email (gauges, scaled shares). Admin app weight-sorted books. Whole-share display. Personal social launch posts (design/documents/personal-launch-social-posts.txt).

## ▶ WATCH / OPEN (Erik-side or wait)
- Funnel: wait a few days → compare /momentum vs /should-i-sell; /should-i-sell offer→CTA leak (2 of 19), signup modal (single-field + one-tap OAuth idea) ([[project_sis_funnel_watch]]). Ads: shared-neg clean, intent split correct, don't switch to conversion bidding (~0 conv).
- DOCS refresh STILL OPEN: signal-intel + tech-arch uncommitted; PDF re-export + investor/marketing/sales sweep. Commit ONLY design/documents.
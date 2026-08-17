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

## ▶ ADS (Aug 17 — Erik doing UI-side; campaign=rigacap-signals-2tier, ad groups Maximize + Preserve)
- Both doors LIVE + pointed with UTM (each ad group → its own landing page). PLUMBING VERIFIED via db_read: /momentum logs pageview+scroll+cta+utm (paid=0, just pointed); /should-i-sell 25 landed/22 paid today, **fold-through 52% today vs 36% baseline** (encouraging — may NOT need redesign; keep watching).
- Advised Erik: (1) NOW OK to remove "trading signals" negative (door exists) BUT add specifics `live signals`/`scalping`/`options signals` + keep day trading/forex/crypto/free; put "momentum trading signals" in Maximize group; watch search terms. (2) DON'T switch to conversion bidding (Google rec) — ~0 conversions to train on; stay Maximize Clicks/Manual CPC for discovery; revisit at ~15+ conv/mo AFTER pre-Stripe leak fixed; also verify conversion action = sign_up/purchase not begin_checkout. (3) Ad-preview showed cross-ad-group keyword OVERLAP (momentum search matched both groups) → add cross-group negatives (Preserve: momentum/breakout/growth; Maximize: sell/protect/crash) + reconsider "stock buy sell signals" in Preserve, so the door A/B stays clean.

## ▶ WATCH / OPEN
- Compare /momentum vs /should-i-sell funnels once momentum has paid clicks (gclid>0) — ping to pull head-to-head. SIS redesign call still open ([[project_sis_funnel_watch]], baseline 36%, today 52%).
- DOCS refresh: signal-intel + tech-arch UNCOMMITTED; PDF re-export + investor/marketing/sales sweep + 3 Qs. Commit ONLY design/documents.

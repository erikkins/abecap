---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 14–19 2026

## Frozen spec
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC; never say "tape" ([[feedback_no_tape_brand_voice]]). PRINT DOCS=ink-on-white; WEB=claret/paper. perf_numbers SSOT. Web deploy=push main→"Deploy RigaCap" GHA + smoke api.rigacap.com/api/market-data-status→200. Admin app=mobile-admin/ EAS OTA (channel preview; Erik reopens). NO Google Ads API (Erik does ad UI). RigaCap=PUBLISHER (no custody, no "your portfolio value" screen). NO HALLUCINATION: AI copy cites only real Python-computed facts.

## ✅ SHIPPED this session (all live)
- Served-Maximizer PORTAL (2 books, gauges, whole-shares, Rotation-watch fill) + Maximizer EMAIL (Maximizer-first, scaled shares+vol-target, gauges, radar, weight WHOLE %). Admin app: weight-sorted positions + labeled equity + Recent transactions.
- Fixes: /api/auth/refresh admin banner; weekend test-email gate; Stripe stats; "tape" removed.
- TWO AD DOORS: /should-i-sell (Preserver) + /momentum (Maximizer, hold-period-first). ExploreMore band both (explore_* events). Both funnels (sis_funnel+mom_funnel) in admin summary; web TrafficTab + mobile Ads tab show both.
- FUNNEL BLIND-SPOT FIXED: App.jsx auto-checkout now fires checkout_redirect (ad-door path); added signup_submit+signup_success sub-steps (LoginModal email/Google/Apple). admin _funnel_order updated.
- MARKET READS now SPY-trend-aware (a8674f0): market_regime.spy_trend_facts() = deterministic streak + 5-session return + from-20d-high (real closes, '' if missing → no fabrication). Base read + Maximizer briefing both fed it + "cite only provided numbers" prompt rule. Anti-repeat 5→8. Takes effect NEXT daily scan.

## ▶ WAIT-AND-WATCH (Erik: wait a couple days, revisit) — [[project_sis_funnel_watch]]
- Ads targeting WORKING (crash/protection intent, ~$2.70 CPC, on-thesis). But 0 conversions on $367 = PAGE problem not targeting. Funnel: /should-i-sell fold-through recovered to 46% (was 36%); real leak = offer→CTA (2 of 19 offer-viewers clicked, 11%). signup→Stripe leg TOO NEW to judge (events 2 days old, 3 opens, 1 maybe Erik). /momentum starved (~0 paid, needs volume).
- NEXT (in a few days): redesign /should-i-sell OFFER/CTA/trust block (not the hero); Erik's signup idea = single-field email-first + prominent one-tap Google/Apple (revisit w/ sub-funnel data). Momentum needs impression volume to evaluate.

## ▶ STILL OPEN
- DOCS refresh: signal-intel + tech-arch UNCOMMITTED; PDF re-export + investor/marketing/sales sweep + 3 Qs. Commit ONLY design/documents.
- Recently-closed (email+portal) auto-activates on first real Maximizer sells (~now/mid-Aug).
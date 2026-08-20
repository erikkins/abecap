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

## 🚨 AWAITING ERIK DECISION (Aug 20) — Maximizer sweep hit validation gate → uncovered LIVE cold-brake ([[project_maximizer_coldstart_jul21]])
- Fork correctly STOPPED (no spread, nothing deployed) — live book can't be reproduced penny-to-penny (old-format Jul8–23 → rebase Jul24 → cold brake). VERIFIED by my own db_read: launched Jul 8, rebased Jul 24, current −1.69%/$98,311 (Aug19; the −0.9%/$99,125 was Aug14). **Vol-target brake OFF entire life** (bk_eq_hist 19<21 → vol_scale=1.0), never warm-started (contra our own rule); self-warms ~Aug21.
- Answer to "maximize customer output": launch-timing risk real + safeguard inert. LEVERS (in order): (1) WARM-START the brake (seed bk_eq_hist from backtested breakout equity + re-seed on rebase) — load-bearing change, AWAITING ERIK SIGN-OFF; (2) tranche/stagger customer entry. Then faithful sweep validated against Jul-24+ era becomes meaningful. Asked Erik: fix warm-start first / run validated sweep / both.

## ✅ Market reads: 3+ session SPY streak = MANDATORY mention (809bbc0, live)
- Confirmed Aug 18 WAS down 3 straight (777.88→767.45) but read didn't say so — pre-deploy old code + prompt led with cross-asset. Added hard rule to BOTH reads (signals.py base system_prompt + maximizer_service base_system): if SPY facts show 3+ straight sessions (up/down), MUST reference (≥a clause) even alongside cross-asset lead; 1-2 day wiggle stays optional. Still hallucination-safe (streak = Python-computed real fact; rule only forces USE, never invent). Net market-read state: SPY-trend-aware + mandatory 3+ streak + cite-only-provided + anti-repeat 8.
- ✅ VERIFIED LIVE: Maximizer read now cites real SPY facts ("down 0.4% over five sessions... 1.1% off its 20-day high" = exact spy_trend_facts output, 769.06 vs 777.88 high) + on-voice + correctly NO streak claim (none today). Blind→aware confirmed. Last box = see it surface a real 3+ session streak.

## ▶ PERSONAL SOCIAL LAUNCH (Aug 19, copy delivered — no code)
- Erik promoting RigaCap on his OWN socials (rare poster = high impact). Drafted "it's live" posts: Facebook (warm story-first, behavioral why, soft ask, link OK), Instagram (punchy + link-in-bio + brand launch card), Story (IG+FB, link sticker). Framing: founder-authentic not ad-copy; lead with WHY (investors lose to own panic not bad stocks); numbers light (compliance+braggy); [X years] placeholder for Erik. NOTE: LinkedIn Stories RETIRED 2021 → suggested a follow-up "founder's note" post instead (main LinkedIn post already done). OFFERED: spin a fresh launch card (social-launch-cards.html pipeline) + match Erik's voice if he pastes a past post.

## ▶ STILL OPEN
- DOCS refresh: signal-intel + tech-arch UNCOMMITTED; PDF re-export + investor/marketing/sales sweep + 3 Qs. Commit ONLY design/documents.
- Recently-closed (email+portal) auto-activates on first real Maximizer sells (~now/mid-Aug).
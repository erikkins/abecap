---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — thru Aug 24 2026

## Frozen spec
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC; never "tape"; NO fabrication (AI cites only real computed facts). Web deploy=push main→"Deploy RigaCap" GHA + smoke api.rigacap.com/api/market-data-status→200. Admin app=mobile-admin/ EAS OTA. NO Google Ads API (Erik does ad UI). RigaCap=PUBLISHER. perf_numbers SSOT. Research maps to prod penny-to-penny; verify before concluding. Admin/test emails→erik@rigacap.com.

## 🔴 LIVE THREAD — /should-i-sell conversion (DISCUSS mode, no changes yet)
- 7-day funnel: /should-i-sell 123 landed (121 paid) → fold 48% (hero FINE) → offer 35% → **CTA 1 of 43 (~2%) = THE LEAK** → signup_open 2 → 0 submit/stripe. /momentum starved (14 landed, unreadable). So: hero recovered (redesign-hero trigger OFF); funnel dies at OFFER→CTA; signup→Stripe leg can't be judged (no volume reaches it).
- GROUNDED offer facts: /should-i-sell CTA = just "Start your 7-day trial" (no price/proof/founder shown there). Modal = "7-day free trial · Credit card required" (CC confirmed). Founding/First-100 framing lives on LandingPageV2 (not should-i-sell). "Direct line to founder" = ADVISER tier only.
- DIAGNOSIS: leak = (a) CC-required trial is a wall for cold cautious 60-something mobile traffic; (b) CTA too thin (no price/proof/trust); (c) **7-day trial TOO SHORT for a weeks-to-months product (29d holds, ~monthly signals) — trial ends before the system does anything.**
- OPEN DECISIONS (Erik weighing, discuss→then change): (1) **30-day MONEY-BACK GUARANTEE** vs longer free trial [my rec: guarantee — matches value horizon + trust + committed payer + category norm (Motley Fool etc.); also = the de-risk-the-ask lever]; (2) no-card vs card; (3) **Founding→"Introductory pricing"** (keep 12mo rate-lock, drop FOMO/empty "first 100" — right for cautious preservation persona) — Erik's instinct; (4) trust/price-transparency on /should-i-sell CTA; (5) two-step signup (email/social→password, "card required" to step 2) = secondary (only 2 reach modal). Offered to pull Stripe trial/checkout config to scope the 30-day-guarantee change.
- Ads context reinforces: audience OLD+MALE (55+≈65%) = capital-preserver fit; buys TRUST not FOMO; 83% mobile → mobile UX critical.

## 🧭 FREE-FIRST PIVOT (design sketched Aug 24, DISCUSS/no build) — earn-the-card-later
- Rationale: Fool/SA can ask for a card because TRUSTED; RigaCap unknown → card ask is a wall. Erik: "those are trusted names." So unknown brand must be MORE generous: no-card free front door → product/proof builds trust → paid ask LATER (30-day money-back guarantee, not 7-day CC trial). 7-day trial ALSO too short (value horizon = weeks-months). Drip must be rewritten to match (currently trial-clock).
- LADDER: cold ad→/should-i-sell w/ 2 no-card doors (free account "see what system says now" + newsletter) → FREE tier (limited/DELAYED signal view = can verify system was right, can't act realtime) → nurture → upgrade (card, money-back) → paid (full realtime: live tickers/mirror-book/alerts). Lapsed→drop to free view (not locked).
- ✅ STATE MODEL DECIDED: add NEW `free` state, RETIRE `trial` (money-back guarantee replaces free-trial; don't overload trial = opposite semantics). Model simplifies: free→active (+ canceled/expired/past_due DROP TO free view). Give free users explicit status='free' record at register (cohort/drip/analytics). REDEFINE `is_valid()` = paid full-access (active+grace) ONLY; free sits below valid (entitled limited tier, not "locked out"). Migration: existing trial→active if carded else free (~6 subs).
- Current infra (grounded): binary gating require_valid_subscription→full-or-403; get_current_user_optional EXISTS; SubscriptionBanner has trial-day/expiry state logic (repurpose as upgrade layer); no-card-trial fields exist in model. BUILD gap: free-tier limited/delayed view + tier-aware serving via optional-auth; register→free (stop auto-Stripe); graceful drop-to-free; drip rewrite; upgrade nudges + money-back offer; Founding→Introductory.
- OPEN DECISION: the FREE/PAID LINE (proposed: delayed-proof free vs realtime-paid). Offered to write data-model spec + confirm free/paid line. Multi-piece pivot — sequence, start w/ free/paid line + tier-aware serving.

## ✅ SHIPPED recently (all live)
- Maximizer vol-brake cold-start fix (bb3c9be, cold-window estimator; warm path byte-identical; certified path + gauge self-heal on Aug24 scan). Start-date sweep (c622e3e; ~$7.4k/$100k launch-swing; tranche onboarding). Newsletter weekly-pack + regime-run + tier rule (f127d0b; draft to erik). SPY-trend market reads + 3+ streak. 2 ad doors + ExploreMore + dual funnel tracking + checkout_redirect/signup sub-steps. Personal social posts.

## ✒️ Seeded: "Pascal's Portfolio" newsletter topic w/ READY DRAFT ([[project_newsletter_pascal_topic]]) — generator does NOT auto-grab; place by hand. Verified 1pm Maximizer social post (13.5 vs 13.2 matched; DD cut ~64%).

## ▶ OTHER OPEN: DOCS refresh (design/documents/* uncommitted). Paste ad negatives. Verify brake/gauge after Aug24 4pm scan.
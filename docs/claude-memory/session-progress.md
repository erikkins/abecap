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
- Never expose publicly: t30v/Core/Ensemble/Option-B/N=15/DWAP/overlay-internals/sleeve-internals. Say "walk-forward" NOT "backtest". Never "tape"/"NaN". PITFWU survivorship-free 2016+ (pre-2016=yfinance survivor-only, disclosed). Survivorship-free NOW OK in customer copy within "How We Test" (Erik reversed old rule Jul 31 — lends credence). Deploy=push main→"Deploy RigaCap".

## CANONICAL = OVERLAY. SSOT = frontend/src/perf_numbers.js + backend/app/services/perf_numbers.py (mirror) + scripts/overlay_canonical.json + docs/numbers-citations-registry.md §1.
- Preserver: 21yr **7.7/0.87/0.56/−13.7**, 5yr 13.0/1.28/1.01/−12.9, rolling-12mo-avg **+10.7%** (77% pos)/24mo 20.8, plan 11-13. Maximizer: 21yr **13.5/0.93/0.65/−20.8**, 5yr 31.4/1.51/2.10/−14.9, rolling-12mo **+26.5%**/24mo 57.5, plan 13-17. S&P 5yr 14.2/0.87/0.56/−25.4, 21yr 9.8/−55. Raw-mom 21yr 13.2/0.69/−57.
- Supporting (overlay, in SSOT): 2008 both +0.1 (SPY −37.7); 2019 +6.1/+1.2 (SPY +28.6); 2020 +9.2/+34.9; 2022 −11.2/−6.4; down-month capture Pres −1.05 (SPY −3.85), up +1.59; longest-underwater 2.2/3.4/5.4; beats-SPY 36/17/14 & 50/45/31; corr 0.55/0.39.
- DECISIONS: recent=ROLLING avg (drop single window); haircut planning-assumptions=DEPTH only; LEAD modern/KEEP 21yr foundation; returns=HERO, Sharpe/Calmar/MDD=supporting; Maximizer=growth SATELLITE; voice=honest-BOLD. NARRATIVE SHIFT: Preserver now MATCHES market at half drawdown (not "beats").

## ▶ GATE B — NOTHING PUSHED/LIVE. All commits LOCAL (6ef01c0, c3897f4, +2 more).
- ✅ DONE (build clean, local): SSOT+registry; **LandingPageV2, TrackRecordPageV2 (incl regime table), MethodologyPageV2, ForAdvisersPage** — the 4 ad-critical/funnel pages, fully wired to overlay. THIS UNBLOCKS ADS (ads land on landing + track-record).
- ⬜ REMAINING: 3 blogs (BlogWalkForwardResultsPage ~14 spots / BlogHonestBacktestPage ~4 / Blog2022StoryPage ~4 — number swaps, live public); components/SocialTab.jsx post library (NEEDS 2-TIER REWRITE not swap — still single-strategy 8.3/19/"backtest", ~10 posts, admin templates); backend text (email_service ≥6 spots / newsletter_generator_service:121 / ai_content_service:56 → import perf_numbers, subscriber prose); IMAGE REGENS: og-card.png ("backtest"+"19%" wrong) + launch-1..5.png. (Knob email-knob-v3.png = FINE, no perf numbers.) Also nice-to-have: fuller Maximizer-satellite narrative block on /for-advisers.
- RECOMMENDATION (given "I NEED subs"): finish blogs (last public inconsistency) → push whole public-page set → LAUNCH ADS → emails/social/images as fast-follow.

## NEXT: finish blogs → push → GOOGLE ADS two-tier (Preserver=capital-preservation $250k+ / Maximizer=aggressive-growth satellite; reuse stability-search-test-a; fix GA4→Ads conversion import). Plan: /Users/erikkins/.claude/plans/unified-sauteeing-whale.md.
## CLEANUP temp scratch: scripts/{tier_vintages_today.py, tier_vintages_daily_today.py, recompute_canonical.py, canonical_recompute.json, tier_curves_21y_today.json, tier_curves_21y.json.bak-may29}. KEEP scripts/overlay_canonical.json.

---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 2dce3134-d861-45c4-a371-80378750f8c0
---

# Session snapshot — Aug 5 2026 (Jul 31 ads content below still current)

## ▶ AUG 5 IN FLIGHT
- **EMAIL↔PORTAL per-tier CONSISTENCY dig — DONE (2 Explore agents traced both paths).** Findings: (1) ✅ Jul-7 sell-alert 12%-trail parity bug is FIXED — signals.py:1926 now passes live regime-adjusted 30% (comment :1920-22 documents it); config default TRAILING_STOP_PCT=12.0 still exists but is no longer used in sell path. → UPDATE project_sell_alert_parity_bug_jul7 to RESOLVED. (2) ⚠️ Portal "Simulated Portfolio" card serves OLD CERTIFIED_WF (tier_serving.py:38-49): Preserver 89.2%/0.97/−20.2, Maximizer 301.4%/1.47/−15.5 — DIVERGES from re-baselined overlay SSOT (Preserver −12.9/1.28, Maximizer −14.9/1.51). Returns reconcile (~29%/12.5% CAGR) but Preserver DD off by ~7pp. = Gate B task #13, awaiting Erik's call (fold into perf_numbers vs label window). (3) Daily email shows NO positions/P&L (positions=[]); portal does — likely by design.
- **▶ ACTIVE: MARKET-CONTEXT ("New Today" blurb vs list) per-tier fix.** Erik's screenshot: Preserver email blurb named HPQ/BMY/BAC/PYPL (day-10 anchors) + MSFT "day 3" — NONE match the "New Today (6)" list (AMZN/GOOG/FIG/PLTR/MSFT/XOM). ROOT: blurb built from FULL buy_signals+continuity (incl held) = shared/book-level & tier-agnostic; email list = buy_signals MINUS held; portal list = all incl held; + count-only cache key (signals.py:1510-13, no symbol-set check) let stale HPQ survive. Erik chose **Option A (tight)** + wants each email to surface correct per-tier signals. 4-change plan presented: (1) per-tier grounded blurbs cached as market_context_preserver/_maximizer, AI cites only shown names; (2) cache key += symbol set; (3) ONE "signals" definition everywhere; (4) fix "New Today" label (only truly-new). **AWAITING Erik A/B decision:** A=signals means new-not-held everywhere (recommended); B=all still-qualifying incl held. Then implement all 4 + test-send to erik@rigacap.com before ship. Tier selection itself already correct (Preserver t30v / Maximizer breakout, scheduler.py:1828-38).
- **✅ CONFIRMATION EMAIL WIRED (not yet deployed).** email_service.send_newsletter_confirmation() (paper-brand, reuses segmented one-click newsletter-unsub token) + fire-and-forget _fire_newsletter_confirmation() in signals.py subscribe endpoint, fires on new-subscribe + resubscribe (not on already-subscribed). Needs deploy; Erik may want a self-test send first. Trigger: 1 new Market Measured signup off /track-record (Preserve ad LP) + a suspected duplicate (2 emails).
- **NEWSLETTER main story PRELOADED** — "One engine, two settings: Preserve & Maximize" draft written inline (honest-bold, F/A self-select, Maximize steer, survivorship-free close). NOT saved to locked draft — awaiting Erik review. For THIS weekend's send.
- **ADS:** `rigacap-signals-2tier` in LEARNING (~1-2 days left). CTR 8.5%, CPC ~$1.94 (well under $6.50 cap), 0 account-signups yet (newsletter signup is a SOFT conv, not the tracked GA4 sign_up). HANDS OFF except negatives — ADDED off-thesis negatives Aug 5: quotex, rsi, indicator, dma, golden cross, bot, binary, elitesignals (search-terms showed day-trader/bot/binary-options junk eating Preserve budget). Judge only after learning + $350/2wk gate.



## Frozen spec
- Public copy: "walk-forward" not "backtest"; never t30v/Core/sleeve/DWAP; "warts" OK in blog voice NOT ads. Survivorship-free OK in customer copy now. STEER toward Maximizer (+$100/mo upsell). Deploy=push main→"Deploy RigaCap".

## OVERLAY CANONICAL (SSOT: frontend/src/perf_numbers.js + backend/app/services/perf_numbers.py + docs/numbers-citations-registry.md §1)
- Preserver 21yr 7.7/0.87/−13.7, 5yr 13.0/1.28/−12.9, typical-yr +10.7%. Maximizer 21yr 13.5/0.93/−20.8, 5yr 31.4/1.51/−14.9, typical-yr +26.5%, 2022 −6.4% (upsell hero). S&P 21yr 9.8/−55. Raw-mom 13.2/0.69/−57.

## ✅ PUBLIC PAGES CONVERTED + LIVE (f51bbac deployed SUCCESS): Landing, TrackRecord, Methodology, ForAdvisers, 3 blogs (Blog2022→Maximizer hero). 
## ⬜ FAST-FOLLOW (not blocking ads): backend emails (email_service/newsletter_generator:121/ai_content:56 → hardcode overlay), SocialTab post library (2-tier REWRITE, still 8.3/19), og-card.png regen ("backtest"+"19%") + launch-1..5.png. Knob PNG fine.

## ✅ GOOGLE ADS FULLY LAUNCHED (Jul 31) — `rigacap-signals-2tier` LIVE. Doc = design/documents/google-ads-2tier-campaign.md.
- Settings: Search only (Display+Search-partners OFF), US, English, **$25/day campaign-level** (Google pushed $60; held 25), bidding=**Max Clicks + $6.50 CPC cap**, AI Max OFF. Conversion tracking already live (**Sign-up=Active+Primary**; old 0-conv was a REAL land→signup leak, not tracking).
- Ad Group "Preserve" → `/track-record?...utm_campaign=preserve` (our phrase/exact kws, 15 HL + 4 desc, strength "Average"). Ad Group "Maximize" → `/?...utm_campaign=maximize` (lead "In 2022 We Lost 6%" + "Beat the S&P, Less Risk"). BOTH SHARE the $25 budget → if it favors Preserve, split into 2 campaigns to force Maximize spend.
- Campaign-level extensions DONE: sitelinks (/track-record, /methodology, /for-advisers, /), negatives list `rigacap-negatives` (bare "volatile" kept OUT), 6 callouts (trimmed "Execute at your broker" to ≤25 char), structured snippet Types=Preserve/Maximize.
- LAUNCH GATE: **$350 / 2-week kill-or-scale — watch COST-PER-SIGNUP, not clicks.** If clicks but ~0 signups by ~$150 → land→signup/price leak (the real risk) → STOP + fix funnel, don't feed budget. Raise budget / switch to Max Conversions or tCPA only after ~15–30 real signups.

## ⬜ TOMORROW (Aug 1) — Erik deferred: (1) backend EMAIL number conversion (email_service ≥6 spots, newsletter_generator:121, ai_content:56 → hardcode overlay); (2) regen OG assets (og-card.png says "backtest"+"19%", launch-1..5.png) + SocialTab post library 2-tier rewrite (still 8.3/19). Ad clickers may hit these → clean before traffic compounds.

## CLEANUP temp scratch: scripts/{tier_vintages_today.py, tier_vintages_daily_today.py, recompute_canonical.py, canonical_recompute.json, tier_curves_21y_today.json, tier_curves_21y.json.bak-may29, breakout_*.py}. KEEP scripts/overlay_canonical.json.

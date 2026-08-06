---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 6 2026

## Frozen spec (load-bearing)
- Public copy: "walk-forward" not "backtest"; never expose t30v/Core/sleeve/DWAP/Ensemble internals; "warts" OK in blog/newsletter voice NOT ads. Survivorship-free OK in customer copy. STEER toward Maximizer (+$100/mo upsell). Deploy = push main → wait "Deploy RigaCap" (gh run watch) → prod (S3/CloudFront + Lambda). Admin/test emails target erik@rigacap.com (in ADMIN_EMAILS → bypasses active-sub filter). NEVER `aws lambda update-function-configuration --environment` (wipes env). Manual worker invoke = rigacap-prod-worker, AWS_PROFILE=rigacap.
- OVERLAY CANONICAL SSOT (frontend/src/perf_numbers.js + backend/app/services/perf_numbers.py + docs/numbers-citations-registry.md §1): Preserver 21yr 7.7/0.87/−13.7, 5yr 13.0/1.28/−12.9, typ-yr +10.7%. Maximizer 21yr 13.5/0.93/−20.8, 5yr 31.4/1.51/−14.9, typ-yr +26.5%, 2022 −6.4%. SPY 21yr 9.8/−55. Raw-mom 13.2/0.69/−57. (Portal Simulated-Portfolio card still on OLD CERTIFIED_WF tier_serving.py:38-49 — Gate B task #13, unresolved.)

## ✅ DONE THIS SESSION (all deployed + live)
- **2-TIER GOOGLE ADS `rigacap-signals-2tier` LIVE** ($25/day, Max-Clicks+$6.50 cap, Preserve→/track-record + Maximize→/). Aug 6: Erik confirmed `negatives` exclusion list NOW ATTACHED (Tools→Exclusion Lists, "1 campaign"; was UNattached Aug 5 → junk traffic unfiltered, explains 0-conv). Negatives: quotex,rsi,indicator,dma,golden cross,bot,binary,elitesignals. Recheck Search-terms ~Aug 8. $350/2wk gate; newsletter signup = SOFT conv (not tracked GA4 sign_up).
- **DAILY REPORT redesign (book-first) — SHIPPED + 6PM digest LIVE.** Blurb grounded in served list (no ghost tickers) + symbol-set cache key. Signal actionability: entry_status fresh/actionable/extended (Preserver=move-since-signal >8%=chase; Maximizer=days_left<10=Late). BOOK-FIRST everywhere (Erik's call: keep valid signals for self-directed subs = diversification): "Our Book" (mirror) leads → "Other Signals — not in our book". Email: _book_section + merged Other Signals; Maximizer always shows full book. Portal: "Other Signals" section under TierBookView. Today's Actions = today-dated fills only. Newsletter confirmation email wired.
- **PRESERVER BOOK GAUGE WIDGET (Aug 6, just deployed 7cb2e69)** — mobile-first (76% mobile!). TierBookView.jsx: Preserver holdings now STACKED CARDS (was 7-col scroll table), each with HoldingGauge = stop→HWM band, green fill=cushion above trailing stop, entry tick + today marker. Data from build_tier_book (entry_price/price/high_water_mark/trailing_stop_level). Maximizer keeps its table+hold-clock. AWAITING ERIK phone review + tweak.

## ✅ DONE Aug 6 (cont.)
- **NUMBER ACCURACY fast-follows SHIPPED (commit e484c6d, deployed).** Backend customer text → overlay SSOT: email_service (welcome/onboarding/trial-ending/winback), newsletter_generator, ai_content, engagement_service. 7.7/0.87/−13.7 + 13.5/0.93/−20.8, 2019 idx +29%, "backtest"→"walk-forward". Pure value swaps (subagent did it, I reviewed diff — clean).
- **MOBILE regime bar fix SHIPPED (commit e484c6d, deployed).** App.jsx advanced-mode regime bar (~3350-3429): header now flex-col on mobile/row on sm+, inline metrics + chips flex-wrap, outlook_detail forecast full-width left on mobile (was cramped right column = the "scrunch" from Erik's phone screenshot). Awaiting Erik phone re-check.
- **MARKETING-PAGE mobile clipping = FALSE ALARM.** My headless screenshots (--window-size 390, headless=new) don't do true device emulation → rendered wide + clipped; Erik's real phone shows NO clipping. V2 pages ARE responsive (grid-cols-1, max-w, overflow-x-clip). Lesson: headless window-size ≠ mobile; trust Erik's phone. DON'T re-chase.

## ▶ AWAITING ERIK DECISION (number-pass flags)
1. Trailing-"last 2 yrs" +31%/+49% (ai_content/newsletter/email) — don't map to overlay 24-mo (+20.8/+57.5 cumulative). Keep/drop/replace?
2. Portal Simulated-Portfolio card CERTIFIED_WF (tier_serving.py:38-49) — old 2021-26 totals + BANNED "Ensemble" label. = Gate B #13. Reconcile to overlay?
3. Launch-card PNGs (design/brand/social-launch-cards.html L261-262 → launch-1..5.png) still 8.6/14.5 — regen needed.
4. Email prose "quarter of rolling 3yr windows" → SSOT 17% (~a sixth). Reword?

## ▶ NEXT / OPEN
- **Erik reviewing the gauge widget on his phone** → iterate on marker clarity/labels/colors at 360px. Task #20 (mirror-book actionability) largely SUBSUMED by the gauge (no separate "extended" tag unless Erik wants one).
- 💡 gauge idea came from Erik Aug 5 EOD; now built.
- **Fast-follows (still pending, customer-facing accuracy):** backend email number conversion to overlay SSOT (email_service ≥6 spots, newsletter_generator:121, ai_content:56) + regen OG/social cards (og-card.png "backtest"/"19%", launch-1..5.png) + SocialTab 2-tier rewrite (still 8.3/19).
- Gate B task #13: fold CERTIFIED_WF → overlay SSOT (portal Simulated-Portfolio card).
- CLEANUP scratch: scripts/{breakout_*,maximizer_*,tier_*_today,*.bak,shapes_tpe.db, scratch json}. KEEP scripts/overlay_canonical.json.

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

## ✅ NUMBER-PASS ITEMS 1/3/4 + Ensemble-label DONE (commit 7ff33fb; deploy QUEUED on GH runner backlog — will land, verify later)
1. ✅ Trailing-24mo +31%/+49% → overlay typical 2yr +20.8%/+57.5% (ai_content, newsletter, welcome-email). Reframed "held-out"→"typical". ⚠️ FLAGGED to Erik: lost the out-of-sample "held out of all tuning" proof-point framing; offered to recompute true held-out-24mo on overlay curve if he wants it back — AWAITING.
3. ✅ Launch-card PNGs regen (card1+card2 → launch-1/2.png): table 7.7/0.87/−13.7 + 13.5/0.93/−20.8, bar-viz, card1 2yr line. Procedure: extract JS → wrapper HTML all-5-canvases (target display:block) → headless=new --force-device-scale-factor=1 --window-size=1080,1350 → copy to frontend/public/launch-cards/. card_N→launch-N.
4. ✅ Email "quarter of rolling 3yr windows" → "about 17%".
- ✅ Item2 LABEL: CERTIFIED_WF "Preserver/Maximizer Ensemble" → dropped "Ensemble".

## ✅ GATE B #13 DONE (Erik chose Option A; "leave ssot typical" on item-1 held-out — no recompute)
- Portal Simulated-Portfolio card now = public /track-record overlay, driven by SINGLE SSOT: tier_serving._tier_card()/tier_backtest() import perf_numbers.PERF → 5yr annualized headline (Pres 13.0/1.28/−12.9, Max 31.4/1.51/−14.9) + 21yr foundation full-cycle (Pres 7.7/0.87/−13.7, Max 13.5/0.93/−20.8). CERTIFIED_WF now a derived alias; dropped rolling-cache path. App.jsx: card labeled "Annualized" (+/yr), full-cycle marked /yr; legacy non-tier "Return" path untouched. Verified locally: py_compile OK, npm build OK, PERF resolves correct. Portal↔site↔emails now single-source, can't drift.

## 🟡 DEPLOY BLOCKED BY GITHUB ACTIONS OUTAGE (Aug 6, confirmed via githubstatus — jobs failing/delayed/timing out, Actions API erroring; GH fix rolling out). NOT our code (verified locally). Erik: "wait it out."
- Commit **43d6127** (cumulative: items 1/3/4 + Ensemble-label + PNGs + Gate B) can't deploy — test job never gets a runner → cancelled ~15min (no timeout/concurrency config; startedAt==createdAt). Runs 31117918919 (7ff33fb) + 31121908355 (43d6127) both cancelled this way.
- **WATCHER LOOP STOPPED (was whack-a-mole: env reaped bg watchers within minutes, run stuck "queued" so watcher couldn't rerun it — zero value).** Run 31121908355/43d6127 stuck-queued 3.5h+ ("stuck runner retrying" per incident; won't self-resolve). Actions API also inconsistent (cancel said "completed" while list said "queued"). **PLAN: wait for GitHub to post incident RESOLVED (Erik relays status updates = the signal), THEN one clean re-trigger: `gh run cancel 31121908355` (if needed) → `gh run rerun 31121908355` (or push) → verify success → verify portal card "+13.0%/yr annualized".** Prior watcher IDs: bob1l0pmc/bxdhyn1rf/b2fdduc4k/b3vawlai8/bosji9714. (zsh reserves `status` var → use `st` if re-scripting.) re-triggers the Deploy run every ~12min, exits+notifies ONLY on success (~6h max). GH incident recovering (20:34 UTC: ~65% of queued jobs now succeeding, up from 30-40%; webhook triggers throttled to 15% so PUSHES may not trigger runs — `gh run rerun` via Actions API is the right path, bypasses webhook). Run 31121908355 / 43d6127 queued ~3.5h+ (long queue = expected per incident). NOTE: can't `gh run rerun` a run while it's still "queued" (only when completed/terminal); if it's stuck-queued for hours consider it may be a "stuck runner retrying" per incident. When it fires success → VERIFY portal card shows "+13.0%/yr annualized" + confirm to Erik. If watcher gives up (still down after ~6h), re-trigger manually once githubstatus.com is green (gh run rerun <latest Deploy run id>).
- Manual-deploy option (scripts/deploy-container.sh + frontend S3/CloudFront) REJECTED by Erik — non-urgent accuracy fixes, not worth deploy-footgun risk.

## (history) DEPLOY STUCK — GH ACTIONS RUNNER BACKLOG (not a failure)
- Commit **43d6127** (contains items 1/3/4 + Ensemble-label + PNGs + Gate B — git is cumulative) QUEUED ~15+min on GitHub runner availability. Earlier run 7ff33fb showed "test: cancelled / deploy: skipped" = concurrency-cancel by my newer push, NOT a real failure. Earlier today runs deployed fine → transient GH infra. Will deploy automatically when a runner frees. **VERIFY IT LANDS** (gh run list --workflow "Deploy RigaCap") + spot-check portal card shows +13.0%/yr annualized once live.

## ▶ NEXT / OPEN
- **Erik reviewing the gauge widget on his phone** → iterate on marker clarity/labels/colors at 360px. Task #20 (mirror-book actionability) largely SUBSUMED by the gauge (no separate "extended" tag unless Erik wants one).
- 💡 gauge idea came from Erik Aug 5 EOD; now built.
- **Fast-follows (still pending, customer-facing accuracy):** backend email number conversion to overlay SSOT (email_service ≥6 spots, newsletter_generator:121, ai_content:56) + regen OG/social cards (og-card.png "backtest"/"19%", launch-1..5.png) + SocialTab 2-tier rewrite (still 8.3/19).
- Gate B task #13: fold CERTIFIED_WF → overlay SSOT (portal Simulated-Portfolio card).
- CLEANUP scratch: scripts/{breakout_*,maximizer_*,tier_*_today,*.bak,shapes_tpe.db, scratch json}. KEEP scripts/overlay_canonical.json.

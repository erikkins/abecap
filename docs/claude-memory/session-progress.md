---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 10–14 2026

## Frozen spec
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC (signal-intel=INTERNAL). PRINT DOCS=ink-on-white+US-English; WEB=claret/paper. perf_numbers SSOT. Web deploy=push main→GHA+smoke. Migration-first DB. TIER_SERVING=true; 0 Maximizer subs, 7 Preserver served.
- **ADMIN APP = mobile-admin/ (Expo, EAS Update OTA, channel=preview, project rigacap-admin, authed as erikkins). NOT web-CI.** Ship changes: commit + `cd mobile-admin && eas update --channel preview -m "..."` → Erik fully-closes/reopens app to pull. Social router mounted at **/api/admin/social** (not /api/social!).

## ✅ SHIPPED (Aug 14, live)
- Served dashboard redesign COMPLETE (books-on-top, 2 mirror books side-by-side, one capital control, signals=deviation layer, +Entry retired, cushion gauge, books always-visible). Web.
- Web book polish: Maximizer capital row moved ABOVE vol-target meter (aligns w/ Preserver); plain-language vol-target explainer added.
- ADMIN APP rebuilt + 2 OTAs to preview: Portfolio=Preserver+Maximizer tier books + Cascade Guard card (Core dropped); Ads=Google summary + Traffic + /should-i-sell funnel; NEW Social queue tab; home freshness chip (services.market_data.status+last_fetch). Backend: Cascade Guard folded into /api/admin/tier-books. Fixed Social tab 404 (path→/api/admin/social/posts). Erik confirmed Google Ads Script runs+populates.

## ▶ AWAITING ERIK DECISION (2 dashboard items, then build in one pass)
- **#3 both market reports:** currently Maximizer AI briefing REPLACES market_context (Preserver read hidden). Rec: backend return BOTH (preserver measured read + maximizer breakout briefing) → show a "market read" line under EACH book. Confirm to build (small backend+frontend).
- **#4 Other Signals attribution:** the Other Signals list = Preserver base momentum names (breakout candidates live in Maximizer radar, not this list). Options: (a) label section "Preserver" (accurate/minimal) or (b) two groups: Preserver momentum + Maximizer breakout candidates (from radar). Erik to pick a/b.
- Also verify OTA social-queue fix worked after Erik reopens app.

## ▶ DOCS REFRESH — NOT COMMITTED (signal-intel pen-marked; 6 html+pdf modified; scratch scripts/ untracked—do NOT commit). PDF re-export + investor/marketing/sales sweep + 3 Qs pending → commit design/documents only.

## ▶ ADS — running (broad TA/tool negatives; check last-1-2-day search terms; cost-per-signup). QUEUED: Maximize recovery landing; "A holding week"; testimonials/churn; unified-sauteeing-whale.md.

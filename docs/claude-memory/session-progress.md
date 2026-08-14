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
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC (signal-intel=INTERNAL). PRINT DOCS=ink-on-white+US-English; WEB=claret/paper. perf_numbers SSOT. Deploy=push main→GHA+smoke-test (web only; mobile-admin is Expo, NOT web-CI — reload on phone). Migration-first DB. STORAGE(live)=PARQUET (PITFWU raw+adjust-at-read). TIER_SERVING=true; 0 Maximizer subs, 7 Preserver served.

## ✅ SHIPPED (Aug 14, all live web): additive Maximizer + served-dashboard redesign COMPLETE (books-on-top, 2 side-by-side mirror books, one capital control, signals=deviation layer "Not in our book — but could be in yours", +Entry retired both tiers, cushion gauge w/ entry tick, books always-visible). Erik: "looks good." Commits 3c529b5→927823a.

## ▶ ACTIVE — ADMIN APP (mobile-admin/, Expo) UPGRADE (Erik: "more info the better")
- Tabs today: index/ads/users/portfolio. Portfolio was Core (getModelPortfolio); Ads was google-only (getAdsSummary). Backend ALL endpoints exist: /api/admin/tier-books, /ads/summary (google-ads-ingest.template.js Ads-Script → /ads/ingest → S3), /pageviews/summary (traffic + sis_funnel), /social/posts?status=, /service-status.
- **BACKEND DONE:** folded Cascade Guard into /api/admin/tier-books response (cascade_guard: {enabled,paused,pause_until,pause_source,last_triggered_at,threshold_stops,pause_days,last_stopped_symbols}). Compiles. (Deploy web next push.)
- **RN BUILD DELEGATED to fork ad29730f46e6f7022:** Portfolio→Preserver+Maximizer books (drop Core)+Cascade Guard card; Ads→keep Google summary + add Traffic + /should-i-sell conversion funnel; NEW Social tab (read-only queue: draft reply approvals + scheduled autoposts); index→data-freshness scan indicator. admin.ts wrappers getTierBooks/getTrafficSummary/getSocialPosts. mobile-admin/ only, no backend, no commit. Awaiting completion.
- **NEXT:** review fork diff → commit backend (tier-books CB) + deploy web → give Erik Expo reload steps. NOTE: Google-Ads half of Ads tab only populates if his hourly Ads Script is running (ADS_INGEST_SECRET); Traffic+funnel works regardless.

## ▶ DOCS REFRESH — NOT COMMITTED (signal-intel pen-marked; 6 html+pdf modified; scratch scripts/ untracked—do NOT commit). PDF re-export + investor/marketing/sales sweep + 3 Qs (pricing figures, BacktesterService name, investor logo) pending → then commit design/documents only.

## ▶ ADS — running (broad TA/tool negatives; check last-1-2-day search terms; cost-per-signup). QUEUED: Maximize recovery landing; "A holding week" label; testimonials/churn; unified-sauteeing-whale.md.

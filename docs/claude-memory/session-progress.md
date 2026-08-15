---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 14 2026

## Frozen spec
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC (signal-intel=INTERNAL). PRINT DOCS=ink-on-white+US-English; WEB=claret/paper. perf_numbers SSOT. Web deploy=push main→"Deploy RigaCap" GHA + smoke curl api.rigacap.com/api/market-data-status→200. Migration-first DB. NEVER lambda update-function-configuration --environment / terraform apply w/o plan. ADMIN APP=mobile-admin/ Expo EAS-OTA channel=preview. TIER_SERVING=true; 0 real external subs.

## ✅ SHIPPED TODAY (live, commit 3696ba6, deploy GREEN + api 200)
- Stripe-sourced admin stats (paid+MRR from Stripe API, internal accts excluded, 5-min cache, safe fallback → today 0 paid/$0 MRR). Email self-sufficiency (capital-scaled shares/$ on Preserver book, "Today's moves" banner, Cascade-Guard notice, both market reads). Portal: each book's candidates moved INTO its column.

## ▶ IN FLIGHT (NOT yet committed/deployed)
- **#1 Market Read equal-height** — DONE in code (TierBookView.jsx: marketNote block got `md:min-h-[112px]` so both side-by-side books' Your-Capital ribbons align). Build clean. NOT deployed yet — batching with the gap fix.
- **#2 Gap under Maximizer book** — AWAITING ERIK. Asked 3 options (AskUserQuestion), he chose "clarify" — I asked what he wants to clarify, awaiting reply. Options: (A) equal-height cols, grow shorter col's candidate card to fill [my rec — keeps all holdings visible]; (B) top-N by weight + "show all" expander; (C) fill Maximizer col w/ breakout-mechanics panel (29d hold/vol-target/Cascade Guard). Grid is App.jsx ~4156 (`items-start` md:grid-cols-2), Preserver col div 4158, Maximizer col div 4179.
- Told Erik: **book ordering = by weight (−implied_value desc) in BOTH books** (build_tier_book, tier_serving.py 436/457) — confirmed to him it's the right default (broker/fund convention, best for mirroring; NEW badge handles recency). Offered optional sort toggle later.

## ▶ STILL DEFERRED (Erik's "just do them all" not 100% done)
- 2 Maximizer EMAIL items: breakout-book per-name capital-scaled shares/$ + vol-target exposure % → need build_maximizer_breakout_view (tier_serving.py ~140) to emit weight_pct + implied shares/value + book-level vol_scale, then render in email_service.py _breakout_book_section + thread vol_scale via scheduler.py.

## ▶ DOCS REFRESH — signal-intel + tech-arch edits UNCOMMITTED; PDF re-export + investor/marketing/sales sweep + 3 Qs pending. Commit ONLY design/documents (scratch scripts/ untracked—never commit).
## ▶ ADS — running; watch search terms + cost-per-signup.

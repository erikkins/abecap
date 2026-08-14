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
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC (signal-intel=INTERNAL). PRINT DOCS=ink-on-white+US-English; WEB=claret/paper. perf_numbers SSOT. Deploy=push main→GHA+smoke-test. Migration-first DB. STORAGE(live)=PARQUET primary (PITFWU raw+adjust-at-read). TIER_SERVING=true live. Subs: 6 active+1 trial (all Preserver-served), 0 Maximizer. Preview via admin ?preview_tier=maximizer / ?preview_tier=preserver.

## ✅ SHIPPED (Aug 14) — MAXIMIZER/served DASHBOARD REDESIGN, COMPLETE & LIVE (commits 3c529b5, ab59423, bc57f5d, 927823a)
- Model = "just mirror the book." +ENTRY / manual Record-Entry RETIRED both tiers (Erik: broken concept). Books = canonical holdings; Signals = deviation layer.
- Backend: Maximizer sub gets BOTH mirror books (tier_book=breakout + preserver_book=base; same capital). Frontend App.jsx: signal_source==='both' → "Your Books" = 2 TierBookViews side-by-side (Preserver left / Maximizer breakout right w/ Today's-Actions+radar). ONE shared capital control (setSharedCapital rescales both). Signals below = "Other Signals · Not in our book — but could be in yours" (informational, click-to-chart, no +Entry). BOOKS ALWAYS VISIBLE for served users (quiet-day gate fixed); signals empty-state added. Legacy/unserved path untouched.
- CUSHION PARITY: Preserver in two-book view renders SAME compact table as Maximizer (TierBookView `compact` prop) — Exit col = thin cushion gauge (stop=left edge, HWM=right edge, green fill=cushion-to-now, entry tick, now marker, stop/high labels). Standalone Preserver keeps full HoldingGauge cards. Fixed the lopsided-height issue. Removed unused v1 BreakoutBookCard.
- AWAITING Erik final look: does entry tick read clearly / need bolder? Otherwise redesign DONE. Minor deferred: 2-book mode shows EOD marks (no live intraday reprice) — low.

## ▶ DOCS REFRESH — NOT COMMITTED (Erik pen-marking signal-intel; 6 html+pdf modified; scratch scripts/ untracked—do NOT commit). Section 09 "receives both books" now TRUE (additive shipped). PDF re-export + investor/marketing/sales sweep + 3 Qs (pricing figures, BacktesterService name, investor logo) pending → then commit design/documents only.

## ▶ ADS — running (broad TA/tool negatives; check last-1-2-day search terms; cost-per-signup). QUEUED: Maximize recovery landing; "A holding week" label; testimonials/churn; unified-sauteeing-whale.md.

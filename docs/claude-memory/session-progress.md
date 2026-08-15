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
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC. PRINT DOCS=ink-on-white+US-English; WEB=claret/paper. perf_numbers SSOT. Web deploy=push main→"Deploy RigaCap" GHA + smoke curl api.rigacap.com/api/market-data-status→200. Migration-first DB. NEVER lambda update-function-configuration --environment / terraform apply w/o plan. mobile-admin/=Expo EAS-OTA channel=preview. TIER_SERVING=true; 0 real external subs.

## ✅ SHIPPED TODAY (all live, deploys GREEN, api 200) — served-Maximizer portal ("both books")
- **Layout NOW:** [Preserver book | Maximizer book + Rotation watch] side-by-side, books-on-top, ONE capital control, market reads PIXEL-aligned → full-width BELOW: Preserver signals, Maximizer breakout candidates, empty-state. (+Entry retired; cushion gauge; always-visible.)
- **Whole shares, NO ≈** (Erik hated squigglies): sh() rounds, `<1` flag; $ value = exact target; email matched. Ordering = by weight both books (broker convention; NEW badge = recency).
- **Market Read pixel-perfect equal height** via JS measure (Dashboard useLayoutEffect measures both `[data-market-read]`, pins to taller; md+ only + resize) — NOT min-height.
- **Rotation watch** (App.jsx, Maximizer col only per Erik "same TD"): 5 nearest 29-day time-stops, re-sorted by URGENCY (book is weight-sorted), 100% live hold-clocks. RESOLVED (commit 432e13f): NATURAL card height + small honest gap (Erik picked this over stretch-fill which left an empty box; books genuinely 20 vs 15). items-start, no flex-stretch.
- **Recently-closed = wired, DORMANT.** signals.py serves `maximizer_recent_exits` from tier_fills sells (empty: prod has 15 maximizer BUYS Jul15–28, ZERO sells; first day-29 exits ~mid-Aug). Auto-splits "Rotation watch | Recently closed" when sells land — no deploy needed.
- Earlier today: Stripe-sourced admin stats (0 paid/$0 MRR, internal excl); email self-sufficiency (Cascade notice, both reads, Today's moves); admin app rebuilt+OTA'd.

## ▶ STILL OPEN ("just do them all")
- Maximizer EMAIL: breakout-book per-name scaled shares/$ + vol-target % (need build_maximizer_breakout_view emit weight_pct/implied + book vol_scale).
- DOCS refresh: signal-intel + tech-arch edits UNCOMMITTED; PDF re-export + investor/marketing/sales sweep + 3 Qs. Commit ONLY design/documents (scratch scripts/ untracked).
- ADS: watch search terms + cost-per-signup.

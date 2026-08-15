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

## ✅ SHIPPED (Aug 14 late pm — portal layout iteration, all deployed GREEN + api 200)
- **dd4187a**: whole shares NO squiggly (Erik: "don't put squigglies before them!") — sh() rounds, `<1` flag; email matched. Market Read PIXEL-PERFECT equal height via JS MEASUREMENT (Dashboard useLayoutEffect measures both `[data-market-read]` blocks → pins to taller; md+ only; resize listener) — replaced the min-h-112px guess.
- **9ab62d1**: FIXED the Maximizer canyon — in-column candidate cards made left col ~4x taller. Pulled Preserver-signals + breakout-candidates OUT to FULL-WIDTH below the two-book grid (removed 3 `signal_source!=='both'` guards at the full-width sections; KEPT guard on the full-width single-book TierBookView so no dup). Rotation watch STAYS in Maximizer column (Erik: "same TD as maximizer").
- **89c3377**: Rotation watch fills column bottom — grid items-stretch + right col flex-col + rotation outer `flex-1 auto-rows-fr`; show up to 5 nearest (was 3). NOTE: if 5 names don't reach Preserver book bottom, card absorbs residual as internal padding. Offered Erik dials: fewer names (more pad) vs natural small gap. AWAIT reaction.
- LAYOUT NOW: [Preserver book | Maximizer book + Rotation watch] side-by-side (books-on-top, one capital control, market reads pixel-aligned) → full-width below: Preserver signals, Maximizer breakout candidates, empty-state.

## ✅ SHIPPED earlier (Aug 14 pm, commit 46bc695)
- **Market Read equal-height** (TierBookView marketNote block `md:min-h-[112px]` → Your-Capital ribbons align in the two-book view).
- **Whole-share ≈** in both portal books + Maximizer email (`≈14 sh`, `<1` flag; $ value stays exact = dollar-first). sh() helper in TierBookView; email_service.py ~410.
- **Rotation watch** under Maximizer book (App.jsx ~4216) — holdings nearest 29-day time-stop, re-sorted by URGENCY (book table is weight-sorted so soonest exit buried); 100% live hold-clocks, no WF. Fills the gap w/ real pertinent content (Erik's steer: "fill, don't overfill; useful+pertinent").
- **Recently-closed = wired but DORMANT.** signals.py serves `maximizer_recent_exits` from tier_fills sells (last 5, pnl_pct = realized/(gross−realized)). Confirmed via prod db_read: tier_fills has ONLY 15 maximizer BUYS (Jul15–28), ZERO sells any tier. So it's [] today → Rotation watch shows solo. When Jul-15 buys hit day29 (~mid-Aug, imminent) block AUTO-SPLITS "Rotation watch | Recently closed" — no further deploy. Frontend split already coded (App.jsx recentlyClosed grid sm:grid-cols-2).
- Book ordering = by weight (−implied_value desc) BOTH books — confirmed to Erik as correct default (broker/fund convention; NEW badge handles recency). Offered optional sort toggle later (declined for now).

## ▶ STILL DEFERRED (Erik's "just do them all" not 100% done)
- 2 Maximizer EMAIL items: breakout-book per-name capital-scaled shares/$ + vol-target exposure % → need build_maximizer_breakout_view (tier_serving.py ~140) to emit weight_pct + implied shares/value + book-level vol_scale, then render in email_service.py _breakout_book_section + thread vol_scale via scheduler.py.

## ▶ DOCS REFRESH — signal-intel + tech-arch edits UNCOMMITTED; PDF re-export + investor/marketing/sales sweep + 3 Qs pending. Commit ONLY design/documents (scratch scripts/ untracked—never commit).
## ▶ ADS — running; watch search terms + cost-per-signup.

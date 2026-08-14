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
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC (signal-intel=INTERNAL). PRINT DOCS=ink-on-white+US-English; WEB=claret/paper. perf_numbers SSOT. Web deploy=push main→GHA+smoke. Migration-first DB. ADMIN APP=mobile-admin/ Expo EAS-OTA channel=preview (eas update --channel preview; Erik reopens app). Social router=/api/admin/social. TIER_SERVING=true; 0 real external subs.

## ✅ SHIPPED TODAY (Aug 14, live): additive Maximizer + served-dashboard redesign COMPLETE (books-on-top, 2 mirror books side-by-side, one capital control, +Entry retired, cushion gauge, always-visible). Plus polish: capital-row-above-vol-target + vol-target explainer; per-book Market Read (#3); two-group Signals Preserver+Maximizer-breakout-candidates (#4); regime chip removed from book headers; Maximizer top daily-report → slim date line (Preserver keeps full); This Week widget pulled for served. Admin app rebuilt+OTA'd (Portfolio tier books+Cascade Guard, Ads+traffic funnel, Social queue [path fixed], freshness). Admin stats: guard so no-Stripe rows don't count as paid (was showing 2 paid/$258 MRR = all internal/self accounts; erikkins@gmail has real sub, amberland+jglynn null stripe_subscription_id).

## ▶ AWAITING ERIK DECISIONS
- **Maximizer daily EMAIL audit done** (email = beta users' ONLY channel; they don't login). HAS: 1 market read, Preserver mirror book (sym/price/%-of-book/exit/PnL), Preserver Other Signals, Breakout Book (day X/29). MISSING (priority): (1) **capital-scaled shares+$ per holding** — can't mirror from email without math (needs per-user portfolio_size in scheduler's build_tier_book); (2) **explicit "Today's moves"** SELL(day29)/NEW callout; (3) **Cascade Guard paused** notice; (4) **both market reads** (scheduler passes max_context OR market_context — Preserver read dropped); (5) vol-target exposure %; (6opt) radar candidates, data date. Rec: build #1+#2 first. Proposed. AWAIT go + subset.
- **Empty space under Maximizer book (portal):** Preserver 20 holdings ≫ Maximizer 15 → right column dead-ends w/ gap. Proposed fix: move each book's candidates INTO its own column under it (kills gap + clearer "which book"); trade-off = re-layout of the just-built full-width Signals. AWAIT.
- **Admin stats real fix (Erik's Q "should come from Stripe API"):** build get_stats to source paid+MRR from Stripe API (kills local drift + fake ×$129 MRR). AWAIT go. Also offered to COMP internal accounts (erikkins@gmail/amberland/jglynn).

## ▶ DOCS REFRESH — NOT COMMITTED (signal-intel pen-marked; 6 html+pdf modified; scratch scripts/ untracked—do NOT commit). PDF re-export + investor/marketing/sales sweep + 3 Qs pending.
## ▶ ADS — running (broad negatives; watch search terms + cost-per-signup).

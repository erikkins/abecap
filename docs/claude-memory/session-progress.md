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
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC (signal-intel dossier=INTERNAL). PRINT DOCS=ink-on-white+US-English; WEB=claret/paper. perf_numbers SSOT. Deploy=push main→GHA+smoke-test. Migration-first DB cols. STORAGE (live): PARQUET primary (PITFWU raw bars+corp-actions, adjust-AT-READ; pickle legacy). TIER_SERVING=true (live, both lambdas). 0 Maximizer-entitled subscribers (has_maxpp_addon) → tier changes have ZERO real blast radius; preview via admin ?preview_tier=maximizer.

## ✅ SHIPPED (Aug 14) — ADDITIVE MAXIMIZER (deployed, live, commit 3c529b5)
- Maximizer subscriber now sees BOTH books (Preserver base + breakout book), delineated — replaces old regime-swap. Backend: tier_serving.apply_tier_serving returns buy_signals=Preserver base + new breakout_book array (always), signal_source='both'; signals.py passes it through. Frontend App.jsx: when signal_source==='both', responsive 2-col grid (grid-cols-1 md:grid-cols-2) — Preserver left / claret Maximizer breakout book right (BreakoutBookCard w/ day X/29 + radar strip), stacks mobile; TierBookView suppressed in 'both'. Email digest + scheduler: maximizer gets Preserver base + delineated breakout block; Preserver path unchanged (verified). Reversible via TIER_SERVING. Fork ae16973c built FE+email.
- **AWAITING ERIK PREVIEW + DECISION:** view rigacap.com/app?preview_tier=maximizer (admin). Flag #1 DESIGN CALL: right column is currently SIMPLE breakout cards + radar (dropped TierBookView's mirror holdings / Set-capital / Today's-Actions) — keep simple, OR swap right col to full TierBookView (richer, tighter on mobile). Minor flags: plain-text email lacks breakout block; push fresh_count=base not breakout-new. Offered to fire sample maximizer email to erik@rigacap.com.

## ▶ ADS — running, watching (broad TA/tool negatives added; check last-1-2-day search terms; funnel filling; judge cost-per-signup).

## ▶ DOCS REFRESH — NOT COMMITTED (Erik pen-marking signal-intel; 6 html+pdf modified; scratch scripts/ untracked — do NOT commit). Aug14 fixes: storage flip, adjust-at-read, NEW Section 03 PITFWU (renumbered to 16), Production-params both-tiers (2 stat rows), Venn nudge. PDF re-export PENDING (batch when Erik done). Once additive ships, Section 09 "receives both books" line is now TRUE — reconcile. OPEN: sweep investor/marketing/sales; 3 Qs (pricing figures, BacktesterService name, investor logo). Then re-export + commit design/documents only.

## ▶ QUEUED: Maximize recovery landing page; "A holding week" label; testimonials/churn; unified-sauteeing-whale.md.

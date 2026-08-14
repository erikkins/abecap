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
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP PUBLIC (signal-intel=INTERNAL). PRINT DOCS=ink-on-white+US-English; WEB=claret/paper. perf_numbers SSOT. Deploy=push main→GHA+smoke-test. Migration-first DB. STORAGE(live)=PARQUET primary (PITFWU raw+adjust-at-read). TIER_SERVING=true live; 0 Maximizer subs → tier UI = ZERO real blast radius; preview via admin ?preview_tier=maximizer.

## ✅ SHIPPED (Aug 14): Additive Maximizer v1 (commit 3c529b5, live) — 2-col Preserver signals | breakout book. Erik previewed → found real IA problems (below), so v1 is a stepping stone; redesigning.

## ▶ ACTIVE — MAXIMIZER DASHBOARD REDESIGN (concept approved, building next)
- DECISION (Erik): **Paradigm A = "just mirror the book"** canonical. RETIRE manual "+Entry"/Record-Entry for served tiers (may reinstate later if users ask to track own positions). KEEP "Other Signals" as an OPPORTUNITY/deviation layer (users can choose to not be exactly the book). "Has to look good."
- Explore agent ab1697 mapped IA. Root cause of v1 problems: additive layout put a SIGNAL list (left) next to a HELD BOOK (right) = apples/oranges → lopsided 12-vs-20; AND suppressing TierBookView + the !tier_book gate orphaned BOTH holdings views + buried day-29 sells (Today's Actions lived inside TierBookView).
- **APPROVED CONCEPT (presented, awaiting Erik's 2 answers):** Top-level = "Your Books" (canonical, side-by-side Preserver mirror-book | Maximizer breakout-book, BOTH holdings so symmetric; each with a "Today ▸" moves line surfacing rotations/day-29 sells + capital-scaled holdings + exit guidance) THEN "Signals" section below (fresh + Other Signals, ●=in-your-book marker, informational, NO +Entry). Books-first, one shared Set-capital control (recommended). Mobile stacks.
- **OPEN Qs to Erik:** (1) Books-on-top vs Signals-first (rec: books first); (2) shared capital control vs per-book (rec: shared). Then BUILD behind flag → live preview.
- **BUILD IMPLICATIONS:** backend must serve BOTH books' holdings for a maximizer sub (tier_book currently single-tier) + each book's today's-moves; frontend = two TierBookView-style panels side-by-side + signals section below + remove +Entry in served mode. tier_serving.py apply_tier_serving + build_tier_book; App.jsx ~4137 grid + ~4188 TierBookView + ~4560 positions table + BreakoutBookCard ~1574.

## ▶ DOCS REFRESH — NOT COMMITTED (Erik pen-marking signal-intel; 6 html+pdf modified; scratch scripts/ untracked—do NOT commit). PDF re-export + investor/marketing/sales sweep + 3 Qs pending → then commit design/documents only.

## ▶ ADS — running (broad TA/tool negatives; check last-1-2-day search terms; cost-per-signup). QUEUED: Maximize recovery landing; "A holding week"; testimonials/churn; unified-sauteeing-whale.md.

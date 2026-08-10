---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Mon Aug 10 2026

## Frozen spec (load-bearing)
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP. Deploy=push main→"Deploy RigaCap" GHA (deploy.yml builds BACKEND Lambda container + FRONTEND S3/CloudFront in one run). **SMOKE-TEST curl api.rigacap.com/api/market-data-status→200 after EVERY deploy.** NEVER terraform apply w/o plan / NEVER lambda update-function-configuration --environment. Worker=rigacap-prod-worker, AWS_PROFILE=rigacap. Worker events: `{"db_read":"SQL"}`(::text), `{"run_migration":true,"sql":...}`(commits), `{"scan_replies":{...}}`. Mass prod DELETE = auto-mode blocked w/o explicit Erik consent.

## ✅ THIS SESSION (Aug 10) — reply engine + tier-book data/UX
- Reply engine de-mechanized (VARY-SHAPE + intra-batch anti-repeat + Maximizer thesis-rotation + anti-judgy). Repeat tickers OK, each a distinct take.
- **Reply engine ANTI-FABRICATION (506dd21):** Erik caught it posting fake wins — all were WalkForwardSimulation rows, not live book (SNDK sim +34% but live LOST -23%; AMD/AMZN were rebalance_exit OPEN-MARKS at sim edge, batch also held -15% losers). Fix: prefer real live/STR closed wins; WF only if CLOSED-ON-A-RULE (not rebalance_exit); `_contradicted_symbols()` hard-blocks names the live/STR book lost/is-underwater on; FACTUAL_ACCURACY_RULES (exact #s, no invented dates, WF must say "in walk-forward testing", else SKIP). Dry-run: 7 fabrications→1 honest cite. **7 bad drafts DELETED.**
- **BUG2 Maximizer STR 8 vs 15 (5a9d2d6):** logging went live Jul 24; 7 pre-Jul-24 breakout entries (S/ERAS/BNY/XYZ/CFG/CSX/VG) never logged. Backfilled from snapshot → tier_fills 15/15. Endpoint (admin.py:4752 get_tier_books) now self-heals (synth display row for any held position missing a fill).
- **Tier-books admin UI (d03cc75):** Core STR collapsed by default (show/hide toggle); endpoint adds current_price+pnl_pct per open row; TierBooksTab.jsx row now shows Entry + Current (live-quote first, EOD fallback, live dot) + P&L %. TierBookView.jsx = the SUBSCRIBER mirror (different file).
- Erik morale note: books flat/down = duration not algo (oldest maximizer pos 18 trading days; live Core since ~Jun 15). Walk-forward = multi-year proof; live record = months/years build (Paul canon). Reassured, grounded in data.

## ▶ NEXT (queued, none started)
- DOCS REFRESH ([[project_docs_refresh]]) — the main planned task. plan unified-sauteeing-whale.md (verify+centralize public return #s → perf_numbers SSOT, Gate A read-only first). "A holding week" Maximizer label. GROWTH: testimonials/churn/ads.

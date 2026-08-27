---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 27 2026 (data-integrity DONE; CHART overlay + previous-holds + admin demo widget SHIPPED)

## ▶▶ GO SLOW — verify don't assume. PARQUET not pickle. NO "DWAP"/"Wtd Avg" customer-facing ([[feedback_no_dwap_customer_facing]]). LONG JOBS: invoke ASYNC (--invocation-type Event); 2-min bash SIGTERMs sync invokes.

## ✅ DATA-INTEGRITY (Aug 26, live): frozen-universe fix (`universe_refresh_v2`, cron SAT20:00); merger-aware heal; calendar completeness + cron SAT18:00 scope=full.

## ✅ CHART + PREVIOUS-HOLDS (Aug 27, all shipped/deploying)
- Backend `GET /api/stock/{symbol}/previous-holds`: Preserver=ModelPosition + Maximizer=TierFill(maximizer) + t30v WF (WalkForwardSimulation.trades_json is_daily_cache) + **Maximizer breakout WF** (reads `signals/maximizer_wf_trades.json` = 392 trades/256 syms, is_walkforward+tier=maximizer). gain/loss=(exit/entry-1)*100.
- `maximizer_wf_trades.json` built via instrumented replay_sleeve (collect side-channel, prod-safe) piggybacked on build_signaled_symbols_5y. Breakout=Maximizer-only → no dedup. Rich data: SMCI +124%, NVDA multi, PLTR/MSTR/APP/VRT 5 each.
- Frontend StockChartModal: killed +20%; labels Average price/Entry trigger + Crossed trigger/Buy signal; PREVIOUS-HOLDS overlay (bands + entry filled/exit ring circle dots + vertical-stagger P&L labels + legend solid=live·dashed=backtested, toggle); MOVED overlay to FOREGROUND (after price Area) so volume bars don't cover dots. Added **5Y range** (breakout holds are 2023-24, were clamped at 2Y).
- Trade History: rows clickable→chart; ticker LOOKUP; Days column fixed.
- **ADMIN-ONLY "Where Your Stocks Sit" widget** (App.jsx `WhereStocksSit`, mounted `{isAdmin && ...}` top of HISTORY tab): paste portfolio → per-name table (holds/Preserver/Maximizer M-badge+best%/best) → row click opens chart+overlay. Paid-customer DIRECTION demo while counsel reviews. "Information only" disclaimer.

## ⏭️ NEXT / OPEN
- IN-CHART "M" badge distinguishing Maximizer WF bands from t30v WF bands (only open polish; table has M-badge, chart doesn't yet).
- Paid widget future: inline mini-charts, SnapTrade/CSV import (Jacob thread), counsel greenlight for real customers.
- 🎯 GRADE after 4pm scan: [[project_maximizer_breakout_prediction_aug26]] WT/BHVN → maximizer_preview + build_todays_actions.
- OTHER: scrub DWAP perf_numbers.js; retire get_universe(); nasdaqtraded.txt ETF rule; rename_continuity(SUNB); remove diag handlers.

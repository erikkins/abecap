---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 27 2026 (data-integrity DONE; CHART overlay shipped + iterating on UI)

## ▶▶ GO SLOW — verify don't assume. PARQUET not pickle. NO "DWAP"/"Wtd Avg" customer-facing ([[feedback_no_dwap_customer_facing]]). Empty {} payloads FALSY→mangum→worker-errors alarm; pass truthy.

## ✅ DATA-INTEGRITY (Aug 26, live): frozen-universe fix (`universe_refresh_v2`, cron SAT 20:00); merger-aware heal; calendar completeness + cron `rigacap-prod-calendar-rebuild` SAT 18:00 scope=full. ASST radar was stale dashboard.json → patched live.

## ✅ CHART FEATURE SHIPPED (Aug 27, deploying — commits through e317284)
- Backend NEW `GET /api/stock/{symbol}/previous-holds` (main.py ~12430): unions Preserver=ModelPosition(live closed) + Maximizer=TierFill(tier=maximizer,paired) + WF=WalkForwardSimulation.trades_json (is_daily_cache canonical, EXCLUDE is_nightly_missed_opps). gain/loss=(exit/entry-1)*100. VERIFIED shapes.
- Frontend StockChartModal (App.jsx): killed +20% line; labels "Average price"/"Entry trigger", verticals "Crossed trigger"/"Buy signal" (no DWAP/ensemble). PREVIOUS-HOLDS overlay: shaded entry→exit bands + gain/loss% (labels bottom, alt L/R to fix collisions) + ▲/▼ prior-hold markers (faded, P&L-colored) + legend "solid=live · dashed=backtested" (was cryptic "(bt)"). Toggle "Prior holds (N)" default-on.
- TRADE HISTORY fixes: rows now clickable→open chart; added ticker LOOKUP search (any symbol, /history yfinance-fallback); Days column showed just "d" (days_held missing)→compute from dates. Test: lookup IREN (WF +110%).
- DATA NOTE: live holds SPARSE (young book); WF is the rich source.

## 🔜 OPEN DECISION — TIER display on charts (Erik leaning teaser, my rec agreed)
- Tiers SHARE base engine; Maximizer only diverges in rotating_bull (breakout sleeve). So showing ALL tiers' holds = ~75% DUPLICATES = clutter. Teaser value = the DIVERGENCE only.
- REC (mine, Erik warm): show viewer's tier normally + overlay ONLY divergent holds from the OTHER tier, BADGED ("M" chip / "Maximizer add-on: +X%"), dedup shared, behind a toggle. Compliance: label as the other PRODUCT's result (esp backtested), never imply user got it.
- BLOCKER for teaser: endpoint pulls only t30v(Preserver) WF; need MAXIMIZER WF trades too (separate artifact/tier vintage) — offered to scope that data source next. Plain tier BADGES on existing data (Preserver live + t30v WF) can come first.

## 🎯 GRADE after 4pm scan: [[project_maximizer_breakout_prediction_aug26]] WT/BHVN → maximizer_preview + build_todays_actions.
## ⏭️ OTHER: scrub literal DWAP in perf_numbers.js; retire get_universe(); nasdaqtraded.txt ETF rule; rename_continuity(SUNB); remove diag handlers; X reply fix.

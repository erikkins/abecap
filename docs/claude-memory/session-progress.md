---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 27 2026 (data-integrity DONE; CHART cleanup + previous-holds overlay SHIPPED)

## ▶▶ GO SLOW — verify don't assume. PARQUET not pickle. NO "DWAP"/"Wtd Avg" customer-facing ([[feedback_no_dwap_customer_facing]]). Empty {} payloads FALSY→mangum→worker-errors alarm; pass truthy.

## ✅ DATA-INTEGRITY (Aug 26, live): frozen-universe fix (`universe_refresh_v2`, weekly cron SAT 20:00); merger-aware heal (classify_short_symbols; yfinance splice removed); calendar completeness fixed + weekly cron `rigacap-prod-calendar-rebuild` SAT 18:00 scope=full. Ready for 4pm scan.
## ✅ ASST on Maximizer radar was STALE dashboard.json (pre-fix 4:30pm scan) → fixed live via patch_breakout_radar (backed up). Live compute clean.

## ✅ CHART WORK SHIPPED (Aug 27, deploying via CI — commits a89c35e/f828326/3280064)
- Backend: NEW `GET /api/stock/{symbol}/previous-holds` (main.py ~12430) — unions Preserver=ModelPosition(live closed) + Maximizer=TierFill(tier=maximizer, paired buy→sell) + WF=WalkForwardSimulation.trades_json. NO double-count. gain/loss = (exit/entry-1)*100. VERIFIED: WF trades_json shape matches (symbol/entry_date/exit_date/prices/pnl_pct/exit_reason). FIXED to use is_daily_cache WF (canonical holds), EXCLUDE is_nightly_missed_opps (opposite of holds).
- Frontend App.jsx StockChartModal (build clean): KILLED +20% line (+orphaned gain20 vars); relabeled "Average price"/"Entry trigger" (was Wtd Avg/DWAP, Breakout), verticals "Crossed trigger"/"Buy signal" (was BREAKOUT/ENTRY); PREVIOUS-HOLDS overlay = shaded entry→exit ReferenceArea bands + gain/loss% label, WF=lighter+dashed+"(bt)", clamped to visible window; "Prior holds (N)" toggle default-on; imported ReferenceArea.
- DATA NOTE: live holds SPARSE (young book, ~1 each CIFR/CORZ/INTC/MU/MRVL); WF is the rich source (e.g. IREN +110%). Test charts: IREN/INTC/MU/MRVL.

## ⏭️ CHART TODO (asked Erik): TIER-GATING not wired — overlay shows ALL sources regardless of viewer tier; each hold tagged w/ tier already, but StockChartModal has NO tier prop → needs viewer tier plumbed in. Erik decided chart SHOULD reflect viewer's tier. Also: DWAP×1.05 IS a LIVE entry gate (signals.py:1021, verified) — line is real, just relabeled.

## 🎯 GRADE after 4pm scan: [[project_maximizer_breakout_prediction_aug26]] WT firing/BHVN cusp → maximizer_preview + build_todays_actions.
## ⏭️ OTHER (none urgent): retire get_universe() (~80); nasdaqtraded.txt ETF rule; rename_continuity carry-forward (SUNB); remove diags read_perf_test/fetch_scope_test/history_source_probe; X reply fix; scrub literal DWAP in perf_numbers.js.
